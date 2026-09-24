"""
09_drift.py
EN: Technical report §9 — did the market move between snapshots? The price distribution of every live
    snapshot (all listings on screen that day, not deduplicated) is compared with the first one and with every
    other (KS, PSI, EMD). Consecutive snapshots share most listings and KS assumes independent samples, so each
    pair is also measured on the disjoint part (shared listings removed). The market level shift is the median
    change of (model, year) cell medians between the first and each later snapshot (cells with ≥3 listings on
    both sides; composition held fixed, no model). Plus the KDE/histogram points of the figures.
TR: Teknik rapor §9 — piyasa taramalar arasında kaydı mı? Her canlı taramanın (o gün ekrandaki bütün ilanlar,
    tekilleştirilmemiş) fiyat dağılımı ilk taramayla ve her biriyle karşılaştırılır (KS, PSI, EMD). Ardışık
    taramalar ilanların çoğunu paylaşır ve KS bağımsız örneklem varsayar; bu yüzden her çift ayrık kısımda da
    (ortak ilanlar çıkarılarak) ölçülür. Piyasa seviyesi kayması: ilk tarama ile her sonraki arasında (model,
    yıl) hücre medyanlarının medyan değişimi (iki tarafta ≥3 ilanlı hücreler; bileşim sabit, model yok).
    Ayrıca figürlerin KDE/histogram noktaları.
Output / Çıktı: metrics/09_drift.json
"""

# %% [1] Setup | Kurulum
import numpy as np
from scipy import stats

from lib.common import load_clean, save_metrics

PSI_BINS = 10
GRID = 120
HIST_EDGES = np.linspace(0, 6.5e6, 40)      # 39 bins up to the 6.5M cap | tavana kadar 39 kova
MIN_CELL = 3


# %% [2] Analysis functions | Analiz fonksiyonları — pure: no file I/O, they only return values
def psi(expected, actual, bins=PSI_BINS):
    """
    EN: Population Stability Index of actual against expected, on expected's deciles (shares floored at 1e-4).
    TR: actual'ın expected'a göre Popülasyon Kararlılık Endeksi; expected'ın ondalıklarında (paylar en az 1e-4).
    """
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.clip(np.histogram(expected, edges)[0] / len(expected), 1e-4, None)
    a = np.clip(np.histogram(actual, edges)[0] / len(actual), 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def compare(a, b):
    """
    EN: KS statistic and p, PSI and EMD (₺) between two price samples. Returns: [KS, p, PSI, EMD].
    TR: İki fiyat örneklemi arasında KS istatistiği ve p, PSI ve EMD (₺). Döndürür: [KS, p, PSI, EMD].
    """
    ks, p = stats.ks_2samp(a, b)
    return [round(float(ks), 4), float(p), round(psi(a, b), 4), round(float(stats.wasserstein_distance(a, b)), 0)]


def drift_tables(rows, snaps):
    """
    EN: Each later snapshot against the first; every ordered pair; and every pair on the disjoint part
        (shared ad_ids removed) with the shared share. Returns: (table, all_pairs, overlap).
    TR: Her sonraki tarama ilkine karşı; her sıralı çift; ve her çift ayrık kısımda (ortak ad_id'ler
        çıkarılmış) ortak payıyla. Döndürür: (tablo, all_pairs, örtüşme).
    """
    by = {s: rows[rows["snap"] == s] for s in snaps}
    table = [[s[5:]] + compare(by[snaps[0]]["price"].values, by[s]["price"].values) for s in snaps[1:]]
    pairs, overlap = [], []
    for i in range(len(snaps)):
        for j in range(i + 1, len(snaps)):
            A, B = by[snaps[i]], by[snaps[j]]
            name = snaps[i][5:] + "→" + snaps[j][5:]
            pairs.append([name] + compare(A["price"].values, B["price"].values))
            shared = np.intersect1d(A["ad_id"].values, B["ad_id"].values)
            a_d, b_d = A.loc[~A["ad_id"].isin(shared), "price"].values, B.loc[~B["ad_id"].isin(shared), "price"].values
            overlap.append([name, round(len(shared) / len(A) * 100, 1)] + compare(a_d, b_d) + [int(len(a_d)), int(len(b_d))])
    return table, pairs, overlap


def density_points(rows, snaps, listing_price):
    """
    EN: KDE of every snapshot's price on a 120-point grid over the listings' price range (raw and log1p), and
        density histograms on fixed edges. Returns: (kde_raw, kde_log, hist) with the grid/edges included.
    TR: Her taramanın fiyat KDE'si, ilanların fiyat aralığında 120 noktalı ızgarada (ham ve log1p), ve sabit
        kenarlarda yoğunluk histogramları. Döndürür: (kde_raw, kde_log, hist) ızgara/kenarlar dahil.
    """
    x_raw = np.linspace(listing_price.min(), listing_price.max(), GRID)
    x_log = np.linspace(np.log1p(listing_price).min(), np.log1p(listing_price).max(), GRID)
    kde_raw, kde_log, hist = {"x": x_raw.tolist()}, {"x": x_log.tolist()}, {}
    for s in snaps:
        p = rows.loc[rows["snap"] == s, "price"].values
        kde_raw[s] = stats.gaussian_kde(p)(x_raw).tolist()
        kde_log[s] = stats.gaussian_kde(np.log1p(p))(x_log).tolist()
        hist[s] = np.histogram(p, bins=HIST_EDGES, density=True)[0].tolist()
    hist["edges"] = HIST_EDGES.tolist()
    return kde_raw, kde_log, hist


def cell_shift(rows, snaps, min_n):
    """
    EN: Median % change of (model, year) cell median prices between the first snapshot and each later one,
        over cells with ≥min_n listings on both sides. Returns: [[MM-DD, median change %, cells], ...].
    TR: İlk tarama ile her sonraki arasında (model, yıl) hücre medyan fiyatlarının medyan % değişimi; iki
        tarafta ≥min_n ilanlı hücrelerde. Döndürür: [[AA-GG, medyan değişim %, hücre], ...].
    """
    agg = lambda s: rows[rows["snap"] == s].groupby(["model", "gb_year"])["price"].agg(["median", "size"])  # noqa: E731
    base, out = agg(snaps[0]), []
    for s in snaps[1:]:
        j = base.join(agg(s), lsuffix="_a", rsuffix="_b", how="inner")
        j = j[(j["size_a"] >= min_n) & (j["size_b"] >= min_n)]
        out.append([s[5:10], round(float(((j["median_b"] / j["median_a"] - 1) * 100).median()), 2), int(len(j))])
    return out


def holm(overlap, alpha=0.05):
    """
    EN: Significance of the disjoint pairs: how many have KS p < alpha, and how many survive a Holm correction
        over all pairs (and which). Returns: {"n_sig", "n_tests", "n_holm", "pairs"}.
    TR: Ayrık çiftlerin anlamlılığı: kaçında KS p < alpha ve bütün çiftler üzerinden Holm düzeltmesinden
        kaçı (hangileri) geçiyor. Döndürür: {"n_sig", "n_tests", "n_holm", "pairs"}.
    """
    ps = sorted(r[3] for r in overlap)
    n_holm = 0
    for i, p in enumerate(ps):
        if p > alpha / (len(ps) - i):
            break
        n_holm += 1
    return {"n_sig": sum(1 for r in overlap if r[3] < alpha), "n_tests": len(ps), "n_holm": n_holm,
            "pairs": [r[0] for r in sorted(overlap, key=lambda r: r[3])[:n_holm]]}


# %% [3] Metrics assembly | Metrik derleme — naming and rounding only | yalnız adlandırma ve yuvarlama
def to_metrics(res):
    """
    EN: Published in the site tree (domain.drift), the report inputs (error_drivers.donem_kaymasi) and
        report.drift_holm.
    TR: Site ağacında (domain.drift), rapor girdilerinde (error_drivers.donem_kaymasi) ve report.drift_holm'da
        yayımlanır.
    """
    return {"domain": {"drift": {
                "table": res["table"], "all_pairs": res["pairs"], "kde_raw": res["kde_raw"], "ortusme": res["overlap"],
                "ortusme_not": ("[cift, ortak ilan %, KS_ayrik, p_ayrik, PSI_ayrik, EMD_ayrik, n_a, n_b]. "
                                "Ayrik = iki taramada da gorulen ilanlar cikarildiktan sonra."),
                "kde_log": res["kde_log"], "hist": res["hist"],
                "not": ("KS=maks dağılım farkı, PSI<0.10 güvenli/>0.25 retrain, EMD=kayma mesafesi (₺). "
                        "hist.edges = bin kenarları (39 bin, 40 kenar); hist[snapshot] = yükseklikler.")}},
            "error_drivers": {"donem_kaymasi": {"min_hucre_n": MIN_CELL, "canli": res["shift"]}},
            "report": {"drift_holm": res["holm"]}}


# %% [4] Load | Yükle — the only cells that read files | dosya okuyan tek hücreler
rows = load_clean(all_snapshots=True)              # every snapshot row, not deduplicated | tekilleştirilmemiş
listings = load_clean()

# %% [5] Compute | Hesapla — look at the results here | sonuçlara burada bak
snaps = sorted(listings["snap"].unique())
table, pairs, overlap = drift_tables(rows, snaps)
kde_raw, kde_log, hist = density_points(rows, snaps, listings["price"].values.astype(float))
res = {"table": table, "pairs": pairs, "overlap": overlap, "kde_raw": kde_raw, "kde_log": kde_log, "hist": hist,
       "shift": cell_shift(rows, snaps, MIN_CELL), "holm": holm(overlap)}
print("max PSI:", max(p[3] for p in pairs), "· cell shift | hücre kayması:", res["shift"])

# %% [6] Save | Kaydet — the only cell that writes the JSON | JSON'u yazan tek hücre
print("written | yazıldı:", save_metrics("09_drift", to_metrics(res)))
