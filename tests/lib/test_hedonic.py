"""
test_hedonic.py
notebooks/lib/hedonic.py küçük, uydurma verilerle: grup içi dönüşüm açık grup kuklalarıyla aynı katsayıları verir;
aralıklar küme sayısı eksi bir serbestlik dereceli t'den; tasarımda kalmayan izlenen terim, boş değer ve tam ranklı
olmayan tasarım durdurur; katsayı tablosu tutarlı; fark testinin varyansı statsmodels'in kümeli varyansıyla aynı
formülden; kat dışı tahminler puanlanan katı hiç görmez. Gerçek veri okunmaz.
"""
import math

import numpy as np
import pandas as pd
import patsy
import pytest
from scipy import stats

from lib import hedonic

TERMS = "age + I(age ** 2) + km100k + I(km100k ** 2) + age:km100k + heavy_damage + painted + changed + hp100 + litre"
CONTROLS = "C(segment) + C(brand) + C(kb_fuel) + C(kb_transmission)"
FORMULA = f"log_price ~ {TERMS} + {CONTROLS}"
TERM_NAMES = ["age", "I(age ** 2)", "km100k", "I(km100k ** 2)", "age:km100k", "heavy_damage", "painted", "changed",
              "hp100", "litre"]


@pytest.fixture(scope="module")
def made_up():
    """
    Uydurma ilanlar: marka araç modeline bağlı, bir segment araç modeline bağlı, öteki segmentler araç modeli içinde
    değişken; güç ve hacim araç modeliyle birlikte değişiyor.
    """
    rng = np.random.default_rng(0)
    n, n_model = 3000, 60
    model = rng.integers(0, n_model, n)
    d = pd.DataFrame({
        "model": model.astype(str), "series": (model // 10).astype(str), "brand": np.where(model < 30, "bmw", "audi"),
        "segment": np.where(model % 3 == 0, "D", np.where(rng.random(n) < .5, "E", "F")),
        "kb_fuel": rng.choice(["Benzin", "Dizel"], n), "kb_transmission": rng.choice(["Düz", "Otomatik"], n),
        "age": rng.normal(0, 4, n), "km100k": rng.normal(0, 1, n), "heavy_damage": rng.integers(0, 2, n),
        "painted": rng.integers(0, 4, n), "changed": rng.integers(0, 3, n),
        "hp100": 1.5 + model / 50 + rng.normal(0, .1, n), "litre": 1.6 + model / 80 + rng.normal(0, .1, n)})
    d["log_price"] = 14 - .07 * d["age"] - .15 * d["km100k"] + .2 * d["hp100"] + model / 40 + rng.normal(0, .1, n)
    return d


def test_pct_effect():
    assert hedonic.pct_effect(0) == 0
    assert hedonic.pct_effect(math.log(1.1)) == pytest.approx(10)


def test_within_equals_dummies(made_up):
    fit, r2 = hedonic.fit_clustered(made_up, FORMULA, "model", absorb="model", tracked=TERM_NAMES)
    y, X = patsy.dmatrices(FORMULA + " + C(model)", made_up, return_type="dataframe")
    y = y.iloc[:, 0].to_numpy()
    beta = np.linalg.lstsq(X.to_numpy(), y, rcond=None)[0]          # açık kuklalar (tekil tasarım, en küçük normlu)
    dummies = dict(zip(X.columns, beta))
    for term in TERM_NAMES:
        assert fit.params[term] == pytest.approx(dummies[term], abs=1e-8), term
    resid = y - X.to_numpy() @ beta
    assert r2 == pytest.approx(1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum(), abs=1e-10)
    assert "Intercept" not in fit.params and "C(brand)[T.bmw]" not in fit.params   # araç modeli içinde sabit


def test_intervals_are_clustered(made_up):
    fit, r2 = hedonic.fit_clustered(made_up, FORMULA, "series")
    n_groups = made_up["series"].nunique()
    half = (fit.conf_int()[1] - fit.conf_int()[0]) / 2
    assert np.allclose(half, stats.t.ppf(0.975, n_groups - 1) * fit.bse)
    assert r2 == pytest.approx(fit.rsquared)                        # emilen grup yoksa sıradan R²


def test_fit_clustered_stops(made_up):
    with pytest.raises(ValueError, match="tasarımda kalmayan"):     # marka araç modeli içinde sabit: düşer
        hedonic.fit_clustered(made_up, FORMULA, "model", absorb="model", tracked=["C(brand)[T.bmw]"])
    with pytest.raises(ValueError, match="tam ranklı"):             # aynı kolon iki kez
        hedonic.fit_clustered(made_up.assign(age_copy=made_up["age"]), "log_price ~ age + age_copy", "model")
    with pytest.raises(patsy.PatsyError):                           # boş değer satır atmaz, durdurur
        hedonic.fit_clustered(made_up.assign(age=made_up["age"].where(made_up.index != 5)), FORMULA, "model")


def test_coefficient_table(made_up):
    fit, _ = hedonic.fit_clustered(made_up, FORMULA, "model", absorb="model", tracked=TERM_NAMES)
    table = hedonic.coefficient_table(fit, TERM_NAMES)
    assert list(table.index) == TERM_NAMES
    assert (table["ci_lo"] <= table["coef"]).all() and (table["coef"] <= table["ci_hi"]).all()
    assert np.allclose(table["pct"], 100 * (np.exp(table["coef"]) - 1))
    assert np.allclose(table["pct_hi"], 100 * (np.exp(table["ci_hi"]) - 1))
    assert (table["contains_zero"] == ((table["ci_lo"] <= 0) & (table["ci_hi"] >= 0))).all()


def test_difference_test_uses_the_clustered_variance(made_up):
    """
    İkinci regresyonun bütün katsayıları ve artıkları sıfırsa fark birincinin katsayısıdır ve standart hatası,
    statsmodels'in kümeli standart hatasından yalnız küçük örneklem düzeltmesi (N - 1) / (N - K) kadar ayrılır.
    """
    fit_a, _ = hedonic.fit_clustered(made_up, FORMULA, "model")
    with np.errstate(invalid="ignore"):                             # sabit hedefte R² 0/0: kullanılmıyor
        fit_b, _ = hedonic.fit_clustered(made_up.assign(log_price=0.0), FORMULA, "model")
    test = hedonic.difference_test(fit_a, fit_b, made_up["model"], TERM_NAMES)
    n, k = fit_a.model.exog.shape
    assert np.allclose(test["diff"], fit_a.params[TERM_NAMES])
    assert np.allclose(test["se"], fit_a.bse[TERM_NAMES] * np.sqrt((n - k) / (n - 1)))
    critical = stats.t.ppf(0.975, made_up["model"].nunique() - 1)
    assert (test["significant"] == (test["t"].abs() > critical)).all()


def test_oof_predictions_never_see_the_scored_fold():
    """Bir katın hedefini değiştirmek o katın kat dışı tahminlerini olduğu gibi bırakır, ötekileri değiştirir."""
    rng = np.random.default_rng(0)
    n = 500
    X = np.column_stack([np.ones(n), rng.normal(size=(n, 3))])
    y = X @ np.array([1.0, 2.0, -1.0, 0.5]) + rng.normal(0, .1, n)
    fold = np.arange(n) % 5
    folds = [(np.flatnonzero(fold != k), np.flatnonzero(fold == k)) for k in range(5)]
    a = hedonic.oof_predictions(X, y, folds)
    y2 = y.copy()
    y2[fold == 0] += 10
    b = hedonic.oof_predictions(X, y2, folds)
    assert np.array_equal(a[fold == 0], b[fold == 0]) and not np.allclose(a[fold != 0], b[fold != 0])


def test_oof_predictions_with_an_unseen_dummy():
    """Eğitim katlarında hiç görülmeyen bir kukla sıfır katsayı alır: satır yalnız öteki kolonlardan tahmin edilir."""
    X = np.array([[1, 0], [1, 0], [1, 0], [1, 1]], dtype=float)     # ikinci kolon yalnız son satırda 1
    y = np.array([1.0, 2.0, 3.0, 10.0])
    folds = [(np.array([0, 1, 2]), np.array([3])), (np.array([1, 2, 3]), np.array([0]))]
    pred = hedonic.oof_predictions(X, y, folds)
    assert pred[3] == pytest.approx(2.0)                              # ilk üç satırın ortalaması
