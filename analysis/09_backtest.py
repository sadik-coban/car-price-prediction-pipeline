"""
09_backtest.py
EN: Technical report §9 — does a model trained on the past price the next snapshot? Four arms, all with a
    lighter LightGBM than the headline (model/series as raw categories, no TF-IDF/SVD, no early stopping):
      - single: train on one snapshot, test on the NEW ad_ids of every later snapshot (no leakage);
      - cumulative: train on all snapshots up to t, same test rule (its first block is the single arm's);
      - insample: plain 5-fold OOF on the listings accumulated up to t (no time dimension);
      - per_snapshot: plain 5-fold OOF inside each snapshot on its own (all listings live that day).
TR: Teknik rapor §9 — geçmişle eğitilen model bir sonraki taramayı fiyatlıyor mu? Dört kol; hepsi manşetten
    hafif bir LightGBM ile (model/seri ham kategori, TF-IDF/SVD yok, erken durdurma yok):
      - single: tek taramada eğit, sonraki her taramanın YENİ ad_id'lerinde test et (sızıntı yok);
      - cumulative: t'ye kadarki bütün taramalarda eğit, aynı test kuralı (ilk bloğu single'ınkiyle aynı);
      - insample: t'ye kadar biriken ilanlarda düz 5-fold OOF (zaman boyutu yok);
      - per_snapshot: her taramanın kendi içinde düz 5-fold OOF (o gün yayında olan bütün ilanlar).
Output / Çıktı: metrics/09_backtest.json
"""

# %% [1] Setup | Kurulum
import lightgbm as lgb
import numpy as np
import pandas as pd

from lib.common import CAT, FEATURES, NUM, TEXT, load_clean, save_metrics
from lib.cv import N_JOBS, PRICE_CAP, SEED, make_folds

LIGHT = dict(learning_rate=0.05, num_leaves=128, random_state=SEED, verbose=-1, deterministic=True, force_row_wise=True)
FORWARD_TREES, OOF_TREES = 800, 500
MIN_TEST, MIN_OOF = 30, 100


# %% [2] Analysis functions | Analiz fonksiyonları — pure: no file I/O, they only return values
def model_matrix(rows):
    """
    EN: Features with categories built from these rows only, log price and price.
    TR: Yalnız bu satırlardan kurulan kategorilerle öznitelikler, log fiyat ve fiyat.
    """
    d = rows.copy()
    for c in CAT + TEXT:
        d[c] = d[c].fillna("missing").astype("category")
    for c in NUM:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d[FEATURES], np.log1p(d["price"].values), d["price"].values.astype(float)


def latest_per_ad(rows):
    """
    EN: One row per ad_id: its latest snapshot within rows. / TR: ad_id başına tek satır: rows içindeki son taraması.
    """
    return rows.sort_values(["ad_id", "search_date"], kind="stable").drop_duplicates("ad_id", keep="last")


def mape(actual, pred_log):
    """
    EN: MAPE (%) of log predictions clipped to [0, log1p(15M)]. / TR: [0, log1p(15M)]'ye kırpılmış log tahminlerin MAPE'si (%).
    """
    p = np.expm1(np.clip(pred_log, 0, np.log1p(PRICE_CAP)))
    return round(float(np.mean(np.abs((actual - p) / actual)) * 100), 2)


def train_forward(rows, snaps):
    """
    EN: Fit on the latest row per ad_id over the given snapshots. Returns: (model, train matrix, ad_ids seen).
    TR: Verilen taramalarda ad_id başına son satırla eğit. Döndürür: (model, eğitim matrisi, görülen ad_id'ler).
    """
    tr = latest_per_ad(rows[rows["snap"].isin(snaps)])
    X, yl, _ = model_matrix(tr)
    mdl = lgb.LGBMRegressor(n_estimators=FORWARD_TREES, n_jobs=N_JOBS, **LIGHT)
    mdl.fit(X, yl, categorical_feature=CAT + TEXT)
    return mdl, X, set(tr["ad_id"])


def test_forward(rows, mdl, X_train, seen, snap):
    """
    EN: MAPE on the listings of snap that the model never saw (new ad_ids), categories aligned to training.
        Returns: (MAPE, n) or None if fewer than 30 test listings.
    TR: snap'in modelin hiç görmediği ilanlarında (yeni ad_id) MAPE; kategoriler eğitime hizalı.
        Döndürür: (MAPE, n), 30'dan az test ilanı varsa None.
    """
    te = rows[(rows["snap"] == snap) & (~rows["ad_id"].isin(seen))].drop_duplicates("ad_id", keep="last")
    if len(te) < MIN_TEST:
        return None
    X, _, y = model_matrix(te)
    for c in CAT + TEXT:
        X[c] = pd.Categorical(X[c], categories=X_train[c].cat.categories)
    return mape(y, mdl.predict(X)), len(te)


def plain_oof_mape(sub):
    """
    EN: Plain 5-fold OOF MAPE inside a set of listings (no time dimension). Returns: (MAPE or None, n).
    TR: Bir ilan kümesinin içinde düz 5-fold OOF MAPE (zaman boyutu yok). Döndürür: (MAPE ya da None, n).
    """
    if len(sub) < MIN_OOF:
        return None, len(sub)
    X, yl, y = model_matrix(sub)
    pred = np.zeros(len(X))
    for tr, va in make_folds(len(X)):
        m = lgb.LGBMRegressor(n_estimators=OOF_TREES, n_jobs=N_JOBS, **LIGHT)
        m.fit(X.iloc[tr], yl[tr], categorical_feature=CAT + TEXT)
        pred[va] = m.predict(X.iloc[va])
    return mape(y, pred), len(sub)


def backtest(rows, snaps):
    """
    EN: The four arms. Returns: {"single", "cumulative", "insample", "per_snapshot"} as row lists.
    TR: Dört kol. Döndürür: satır listeleri olarak {"single", "cumulative", "insample", "per_snapshot"}.
    """
    single, cumulative = [], []
    for i, s in enumerate(snaps[:-1]):
        mdl, X, seen = train_forward(rows, [s])
        for t in snaps[i + 1:]:
            r = test_forward(rows, mdl, X, seen, t)
            if r:
                single.append([s[5:], t[5:], r[0], r[1]])
    for i in range(1, len(snaps)):
        mdl, X, seen = train_forward(rows, snaps[:i])
        for t in snaps[i:]:
            r = test_forward(rows, mdl, X, seen, t)
            if r:
                cumulative.append(["→" + snaps[i - 1][5:], t[5:], r[0], r[1]])
    insample = []
    for i in range(1, len(snaps) + 1):
        mp, n = plain_oof_mape(latest_per_ad(rows[rows["snap"].isin(snaps[:i])]))
        insample.append(["→" + snaps[i - 1][5:] if i > 1 else snaps[0][5:], mp, n])
    per_snapshot = [[s[5:], *plain_oof_mape(rows[rows["snap"] == s])] for s in snaps]
    return {"single": single, "cumulative": cumulative, "insample": insample, "per_snapshot": per_snapshot}


# %% [3] Metrics assembly | Metrik derleme — naming and rounding only | yalnız adlandırma ve yuvarlama
def to_metrics(res):
    """
    EN: Published in the site tree (methodology.backtest).
    TR: Site ağacında yayımlanır (methodology.backtest).
    """
    return {"methodology": {"backtest": {
        **res,
        "protocol": {"leak_free": ["single", "cumulative"], "plain_kfold": ["insample", "per_snapshot"],
                     "same_experiment": "cumulative ilk blok (<= ilk tarama) single ilk blokla ayni",
                     "light_model": ("tum kollar ana modelden hafif: TF-IDF/SVD yok (ad ham kategorik), erken durdurma yok; "
                                     "single/cumulative 800 agac, insample/per_snapshot 500 agac")},
        "note": ("single: tek taramada eğit, sonraki taramaların yalnız YENİ ad_id'lerinde test et (sızıntısız). "
                "cumulative: t'ye kadarki taramalarda eğit, aynı kural; ilk bloğu single ile aynı deneydir. "
                "insample: kümülatif birikimde düz 5-fold OOF, zaman boyutu yok. per_snapshot: her tarama "
                "izole, düz 5-fold OOF. Dört kol da ana modelden hafif: TF-IDF/SVD yok, erken durdurma yok "
                "(single/cumulative 800, insample/per_snapshot 500 ağaç).")}}}


# %% [4] Load | Yükle — the only cells that read files | dosya okuyan tek hücreler
rows = load_clean(all_snapshots=True)              # every snapshot row, ORDER BY ad_id, search_date

# %% [5] Compute | Hesapla — look at the results here | sonuçlara burada bak
snaps = sorted(rows["snap"].unique())
res = backtest(rows, snaps)
print("single:", res["single"][:3], "· insample:", res["insample"][-1])

# %% [6] Save | Kaydet — the only cell that writes the JSON | JSON'u yazan tek hücre
print("written | yazıldı:", save_metrics("09_backtest", to_metrics(res)))
