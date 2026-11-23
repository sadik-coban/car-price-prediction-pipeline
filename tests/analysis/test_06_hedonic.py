"""
test_06_hedonic.py
EN: Invariants of metrics/06_hedonic.json (technical report §6 hedonic model): the OLS sample is the listings minus
    the dropped ones (hp or cc missing, both counted once); both columns carry the same ten terms; every row is
    consistent (interval ordered around the point, the "contains zero" flag matches it, the % effect and its bounds
    are exp(β)−1); the headline effects are the model column's; the intervals are clustered by model; adding C(model)
    never lowers the in-sample R²; the largest VIF is ≥ 1 and centring lowers it.
TR: metrics/06_hedonic.json'un değişmezleri (teknik rapor §6 hedonik model): OLS örneklemi ilanlar eksi atılanlar
    (hp ya da cc eksik, ikisi bir kez sayılır); iki sütun aynı on terimi taşıyor; her satır tutarlı (aralık noktanın
    iki yanında sıralı, "sıfırı içeriyor" bayrağı ona uyuyor, % etki ve sınırları exp(β)−1); manşet etkiler model
    sütununun; aralıklar modele göre kümeli; C(model) eklemek örneklem içi R²'yi düşürmüyor; en yüksek VIF ≥ 1 ve
    ortalamak onu düşürüyor.
EN (2026-11-23): the ladder's out-of-fold predictions never see the scored fold; its rungs are nested as they should
    be and include the published columns; the column without identity controls has the same terms; the age steps and
    the parts-by-count rows are well formed.
TR (2026-11-23): merdivenin kat dışı tahminleri puanlanan katı hiç görmez; basamakları olması gerektiği gibi iç içe ve
    yayımlanan sütunları içerir; kimlik kontrolsüz sütun aynı terimleri taşır; yaş basamakları ve sayıya göre parça
    satırları düzgün kurulu.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
TERMS = ["age", "age_sq", "km100k", "km_sq", "age_x_km", "heavy_damage", "painted", "changed", "hp100", "litre"]


@pytest.fixture(scope="module")
def doc(metrics):
    """EN: The published file. / TR: Yayımlanan dosya."""
    return metrics("06_hedonic")


@pytest.fixture(scope="module")
def hr(doc):
    """EN: The reliability block. / TR: Güvenilirlik bloğu."""
    return doc["domain"]["hedonic_reliability"]


def test_sample_size(doc, hr, metrics):
    """EN: n = listings − dropped; dropped = hp + cc − both. / TR: n = ilan − atılan; atılan = hp + cc − ikisi."""
    dr = doc["error_drivers"]["hedonic_dropped"]
    assert dr["total"] == dr["hp"] + dr["cc"] - dr["both"] and dr["both"] <= min(dr["hp"], dr["cc"])
    assert hr["n"] == metrics("01_dedup_leakage")["meta"]["n_dedup"] - dr["total"]


def test_coefficient_rows(hr):
    """
    EN: Same ten terms in both columns; ci_lo ≤ point ≤ ci_hi; contains_zero ⇔ ci_lo ≤ 0 ≤ ci_hi; % effect and bounds
        = 100·(e^x − 1) (x is rounded).
    TR: İki sütunda aynı on terim; ci_lo ≤ nokta ≤ ci_hi; contains_zero ⇔ ci_lo ≤ 0 ≤ ci_hi; % etki ve sınırlar
        = 100·(e^x − 1) (x yuvarlanmış).
    """
    assert set(hr["columns"]) == {"segment", "model"}
    for col in hr["columns"].values():
        assert [r["term"] for r in col["coefficients"]] == TERMS
        for r in col["coefficients"]:
            assert r["ci_lo"] <= r["point"] <= r["ci_hi"] and r["se"] > 0, r["term"]
            assert r["contains_zero"] == (r["ci_lo"] <= 0 <= r["ci_hi"]), r["term"]
            for x, p in (("point", "pct_effect"), ("ci_lo", "pct_lo"), ("ci_hi", "pct_hi")):
                assert r[p] == pytest.approx(100 * (math.exp(r[x]) - 1), abs=0.02), (r["term"], p)


def test_headline_is_the_model_column(doc, hr):
    """
    EN: The headline R² and effects are the model column's (the decision note prints them).
    TR: Manşet R² ve etkiler model sütununun (karar notu bunları basıyor).
    """
    hd = doc["domain"]["hedonic"]
    col = hr["columns"]["model"]
    effect = {r["term"]: r["pct_effect"] for r in col["coefficients"]}
    assert hd["control"] == "model" and hd["r2"] == col["r2"]
    for key, term in {"age_pct": "age", "damage_pct": "heavy_damage", "km100k_pct": "km100k", "hp100_pct": "hp100",
                      "cc_litre_pct": "litre"}.items():
        assert hd[key] == effect[term], key


def test_clustered_by_model(hr, metrics):
    """
    EN: Clustered by model; a cluster count between 2 and the listings, not above the models in the data.
    TR: Modele göre kümeli; küme sayısı 2 ile ilan sayısı arasında, verideki model sayısını aşmıyor.
    """
    n_models = len(metrics("08_large_errors")["error_drivers"]["per_model_error"])
    assert hr["cluster"] == "model" and 2 <= hr["n_clusters"] <= min(hr["n"], n_models)


def test_model_identity(hr):
    """EN: Adding C(model) never lowers the in-sample R². / TR: C(model) eklemek örneklem içi R²'yi düşürmez."""
    assert hr["columns"]["model"]["r2"] >= hr["columns"]["segment"]["r2"]


def test_vif(hr):
    """EN: Largest VIF ≥ 1; centring does not raise it. / TR: En yüksek VIF ≥ 1; ortalamak onu yükseltmez."""
    v = hr["vif"]
    assert v["term"] in TERMS and v["raw_term"] in TERMS
    assert 1 <= v["value"] <= v["raw_value"]


@pytest.fixture(scope="module")
def functions():
    """
    EN: The script's pure functions (cells 1–3), without loading any data.
    TR: Betiğin saf işlevleri (1–3. hücreler), veri yüklemeden.
    """
    # EN: "lib" is also db/lib (other tests import it): load analysis/lib for this exec only, then put things back
    # TR: "lib" aynı zamanda db/lib (başka testler import eder): yalnız bu exec için analysis/lib yüklenir, sonra geri konur
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "lib" or k.startswith("lib.")}
    sys.path.insert(0, str(ROOT / "analysis"))
    try:
        head = (ROOT / "analysis" / "06_hedonic.py").read_text(encoding="utf-8").split("# %% [4]")[0]
        ns = {"__name__": "hedonic_functions"}
        exec(compile(head, "06_hedonic.py", "exec"), ns)
    finally:
        sys.path.remove(str(ROOT / "analysis"))
        for k in [k for k in sys.modules if k == "lib" or k.startswith("lib.")]:
            del sys.modules[k]
        sys.modules.update(saved)
    return ns


def test_within_equals_dummies(functions):
    """
    EN: On made-up data with nested dummies (brand fixed by model, one segment fixed by model, the others varying
        inside models), the within-model column gives the same tracked estimates as OLS on explicit model dummies.
    TR: İç içe kuklalı uydurma veride (marka modele bağlı, bir segment modele bağlı, ötekiler model içinde değişken) model içi
        sütun, açık model kuklalı OLS ile aynı izlenen tahminleri verir.
    """
    rng = np.random.default_rng(0)
    n, n_model = 3000, 60
    model = rng.integers(0, n_model, n)
    d = pd.DataFrame({"model": model.astype(str), "brand": np.where(model < 30, "a", "b"),
                      "segment": np.where(model % 3 == 0, "D", np.where(rng.random(n) < .5, "E", "F")),
                      "kb_fuel": rng.choice(["Benzin", "Dizel"], n), "kb_transmission": rng.choice(["Manuel", "Otomatik"], n),
                      "age": rng.normal(0, 4, n), "km10": rng.normal(0, 1, n), "dmg": rng.integers(0, 2, n),
                      "painted": rng.integers(0, 4, n), "changed": rng.integers(0, 3, n),
                      "hp100": 1.5 + model / 50 + rng.normal(0, .1, n), "cc_L": 1.6 + model / 80 + rng.normal(0, .1, n)})
    d["log_price"] = 14 - .07 * d.age - .15 * d.km10 + .2 * d.hp100 + model / 40 + rng.normal(0, .1, n)
    fit, _ = functions["fit_column"](d, "model", "model")
    lsdv = smf.ols(functions["FORMULA"] + "+C(model)", data=d).fit()
    for k in functions["TERM_ID"]:
        assert fit.params[k] == pytest.approx(lsdv.params[k], abs=1e-8), k


def test_difference_is_segment_minus_model(hr):
    """
    EN: diff = segment − model on the log scale; t = diff / se; significant ⇔ |t| above the t(G−1) 97.5% point.
    TR: fark = segment − model (log ölçek); t = fark / sh; anlamlı ⇔ |t| t(G−1) %97,5 noktasının üstünde.
    """
    seg = {r["term"]: r["point"] for r in hr["columns"]["segment"]["coefficients"]}
    mod = {r["term"]: r["point"] for r in hr["columns"]["model"]["coefficients"]}
    crit = stats.t.ppf(0.975, hr["n_clusters"] - 1)
    assert [d["term"] for d in hr["difference"]] == TERMS
    for d in hr["difference"]:
        assert d["diff"] == pytest.approx(seg[d["term"]] - mod[d["term"]], abs=2e-4)
        assert d["t"] == pytest.approx(d["diff"] / d["se"], abs=0.02)
        assert d["significant"] == (abs(d["diff"] / d["se"]) > crit)


def test_sensitivities(hr, metrics):
    """
    EN: The series clustering has fewer clusters and names known terms; the spec-outlier refit drops at most §8's
        flagged listings; power varies in at most every model and less within models than overall.
    TR: Seri kümelemesinde küme sayısı daha az ve bilinen terimleri anıyor; tutarsızlar olmadan yeniden kurulum en çok
        §8'in işaretlediği ilanları atıyor; güç en çok her modelde değişiyor ve model içinde genelden az.
    """
    sc = hr["series_cluster"]
    assert 2 <= sc["n_clusters"] < hr["n_clusters"]
    assert all(set(v) <= set(TERMS) for v in sc["zero_in_ci"].values()) and set(sc["zero_in_ci"]) == {"segment", "model"}
    ws = hr["without_spec_outliers"]
    assert 0 < ws["n_dropped"] <= metrics("08_large_errors")["error_drivers"]["spec_outliers"]["n"]
    assert [r["term"] for r in ws["coefficients"]] == ["hp100", "litre"]
    hw = hr["hp_within_model"]
    assert hw["models"] == hr["n_clusters"] and 0 < hw["models_varying"] <= hw["models"]
    assert 0 < hw["sd_within"] < hw["sd_overall"]


# ── Ladder, no-identity column, age steps, parts by count (2026-11-23) ─────────────────────────────
@pytest.fixture(scope="module")
def rh(doc):
    """EN: The report-only hedonic block. / TR: Yalnız raporun kullandığı hedonik blok."""
    return doc["report"]["hedonic"]


def test_oof_predictions_never_see_the_scored_fold(functions):
    """
    EN: Changing the prices of one fold leaves that fold's out-of-fold predictions as they were and moves the others.
    TR: Bir katın fiyatlarını değiştirmek o katın kat dışı tahminlerini olduğu gibi bırakır, ötekileri değiştirir.
    """
    rng = np.random.default_rng(0)
    n = 500
    X = np.column_stack([np.ones(n), rng.normal(size=(n, 3))])
    y = X @ np.array([1.0, 2.0, -1.0, 0.5]) + rng.normal(0, .1, n)
    fold = np.arange(n) % 5
    folds = [(np.flatnonzero(fold != k), np.flatnonzero(fold == k)) for k in range(5)]
    a = functions["oof_predictions"](X, y, folds)
    y2 = y.copy()
    y2[fold == 0] += 10
    b = functions["oof_predictions"](X, y2, folds)
    assert np.array_equal(a[fold == 0], b[fold == 0]) and not np.allclose(a[fold != 0], b[fold != 0])


def test_ladder(rh, hr):
    """
    EN: Six regressions; the two published columns and the no-identity column are rungs of it (same in-sample R²);
        in-sample R² never falls when terms are added (condition ⊂ condition + engine ⊂ segment control ⊂ car-model
        control; segment identity ⊂ segment control; car-model identity ⊂ car-model control); out of fold a
        regression is not better than in sample (beyond rounding).
    TR: Altı regresyon; yayımlanan iki sütun ve kimliksiz sütun onun basamakları (aynı örneklem içi R²); terim
        eklendikçe örneklem içi R² düşmez (durum ⊂ durum + motor ⊂ segment kontrolü ⊂ araç modeli kontrolü; segment
        kimliği ⊂ segment kontrolü; araç modeli kimliği ⊂ araç modeli kontrolü); kat dışında bir regresyon örneklem
        içinden iyi değil (yuvarlama dışında).
    """
    rows = {r[0]: r for r in rh["ladder"]["rows"]}
    assert list(rows) == ["identity_segment", "identity_model", "condition", "condition_engine", "segment_control",
                          "model_control"] and len(rh["ladder"]["columns"]) == 4
    assert rows["segment_control"][2] == hr["columns"]["segment"]["r2"]
    assert rows["model_control"][2] == hr["columns"]["model"]["r2"]
    assert rows["condition_engine"][2] == rh["no_identity"]["r2"]
    for small, big in (("condition", "condition_engine"), ("condition_engine", "segment_control"),
                       ("segment_control", "model_control"), ("identity_segment", "segment_control"),
                       ("identity_model", "model_control")):
        assert rows[small][2] <= rows[big][2] and rows[small][1] < rows[big][1], (small, big)
    assert all(0 < r[3] <= r[2] + 1e-4 for r in rows.values())


def test_no_identity_column(rh):
    """
    EN: The column without identity controls carries the same ten terms, with consistent rows.
    TR: Kimlik kontrolsüz sütun aynı on terimi, tutarlı satırlarla taşır.
    """
    assert [r["term"] for r in rh["no_identity"]["coefficients"]] == TERMS
    for r in rh["no_identity"]["coefficients"]:
        assert r["ci_lo"] <= r["point"] <= r["ci_hi"] and r["se"] > 0, r["term"]
        assert r["pct_effect"] == pytest.approx(100 * (math.exp(r["point"]) - 1), abs=0.02), r["term"]


def test_age_steps_and_counts(rh, hr):
    """
    EN: Age steps: increasing age pairs, each with listings and an interval around the point. Parts by count: counts
        1 … top for painted and changed, each with listings (never more than the sample) and an ordered interval.
    TR: Yaş basamakları: artan yaş çiftleri, her birinde ilan ve noktanın iki yanında aralık. Sayıya göre parça:
        boyalı ve değişen için 1 … top; her birinde ilan (örneklemden fazla olamaz) ve sıralı aralık.
    """
    steps = rh["age_steps"]["rows"]
    assert len(rh["age_steps"]["columns"]) == 6 and all(a < b for a, b, *_ in steps)
    assert [s[0] for s in steps] == sorted(s[0] for s in steps)
    assert all(0 < s[2] <= hr["n"] and s[4] <= s[3] <= s[5] for s in steps)
    bc = rh["by_count"]
    for name in ("painted", "changed"):
        assert [r[0] for r in bc[name]] == list(range(1, bc["top"] + 1))
        assert all(r[3] <= r[2] <= r[4] and r[1] > 0 for r in bc[name]) and sum(r[1] for r in bc[name]) < hr["n"]
