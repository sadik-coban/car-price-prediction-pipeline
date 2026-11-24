"""
Eksiklik: birlikte eksik kolon blokları ve kb/gb ikiz çiftleri.
Modülü import etmek hiçbir dosya okumaz, yazmaz.
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def co_missing_share(table, columns):
    """Birlikte eksik payı: kolonların hepsinin boş olduğu satırların, en az birinin boş olduğu satırlara oranı."""
    empty = table[columns].isna()
    either = empty.any(axis=1).sum()
    return float(empty.all(axis=1).sum() / either) if either else 0.0


def comissing_blocks(table, columns, max_dist):
    """
    Kolonları birlikte boş kaldıkları bloklara ayırır. İki kolonun uzaklığı = 1 - birlikte eksik payı; ortalama
    bağlantılı hiyerarşik kümeleme max_dist'te kesilir. Döndürür: en az iki kolonlu bloklar (kolon listeleri),
    büyükten küçüğe; blok içinde kolonlar verildikleri sırada.
    """
    if len(columns) < 2:
        return []
    dist = np.zeros((len(columns), len(columns)))
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            dist[i, j] = dist[j, i] = 1 - co_missing_share(table, [columns[i], columns[j]])
    labels = fcluster(linkage(squareform(dist, checks=False), method="average"), t=max_dist, criterion="distance")
    groups = {}
    for column, label in zip(columns, labels):
        groups.setdefault(label, []).append(column)
    return sorted([g for g in groups.values() if len(g) >= 2], key=len, reverse=True)


def twin_table(table, used, by_hand):
    """
    kb/gb ikiz çiftleri (kb_x ile gb_x; ayrıca kb_is_heavy_damaged ile is_heavy_damaged). Her çift için boş payları,
    farklı değer sayıları, iki tarafın aynı olduğu satırların payı (ikisi de boşsa aynı), tahmin modelinin tuttuğu
    taraf (used içinde olan), atılan taraf ve gerekçesi: by_hand içindeyse 'by_hand', iki taraf en az %99.9 aynıysa
    'identical', atılan taraf %40'tan fazla boşsa 'empty'. Gerekçesi olmayan atılmış bir taraf kalırsa durur.
    """
    pairs = [(c, "gb_" + c[3:]) for c in table.columns if c.startswith("kb_") and "gb_" + c[3:] in table.columns]
    pairs.append(("kb_is_heavy_damaged", "is_heavy_damaged"))
    rows = []
    for kb, gb in pairs:
        same = ((table[kb].astype(str) == table[gb].astype(str)) | (table[kb].isna() & table[gb].isna())).mean()
        kept = kb if kb in used else (gb if gb in used else None)
        dropped = gb if kb in used or gb not in used else kb
        empty = table[dropped].isna().mean()
        reason = ("by_hand" if dropped in by_hand else "identical" if 100 * same >= 99.9
                  else "empty" if 100 * empty > 40 else None)
        if reason is None:
            raise ValueError(f"atılan ikiz {dropped} için gerekçe yok (aynı %{100 * same:.1f}, boş %{100 * empty:.1f})")
        rows.append({"kb": kb, "gb": gb, "kb_missing_pct": 100 * table[kb].isna().mean(),
                     "gb_missing_pct": 100 * table[gb].isna().mean(), "kb_unique": table[kb].nunique(),
                     "gb_unique": table[gb].nunique(), "same_pct": 100 * same, "kept": kept, "dropped": dropped,
                     "dropped_missing_pct": 100 * empty, "reason": reason})
    return pd.DataFrame(rows)
