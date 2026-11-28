"""
test_interval.py
notebooks/lib/interval.py küçük, uydurma ilanlarla: aralık log ölçekte kurulur ve kırpılır; sonlu örneklem yüzdeliği;
tek pay kolunda hiçbir katın q'su kendi hatalarını görmez; emsal ilanın kendi katı sayılmadan sayılır; ölçek
modelinin matrisi metin kolonsuz kat matrisi + log tahmin + emsal; s puanladığı katı görmez, iki eğitim aynı s'yi
verir, iş parçacığı sayısı s'yi değiştirmez, alt sınırda kırpılır ve emsal arttıkça artmaz; çapraz kalibreli kolda
puanlanan kat başına tek k ve o k puanlanan katın hatalarını görmez; isabet olasılığı s büyüdükçe artmaz; "10'da N" en
çok 9; aralık skoru, genişlik, kol özeti, güvenilirlik tablosu; servis edilen ölçek modeli bütün ilanların kodlarıyla;
işlenmiş örnekler.
Gerçek veri (data işareti): (araç modeli, yaş) hücreleri (araç modeli, model yılı) hücreleriyle aynı.
"""
import numpy as np
import pandas as pd
import pytest

from lib import data, folds, interval, model

NAMES = ["320i Sport", "320d Luxury", "520d Sport", "520i Base", "A4 35 TFSI", "A4 40 TDI", "A6 45 TDI", "A6 Base",
         "X5 30d", "Q5 40 TDI"]


def made_up(n=600, seed=0):
    """Uydurma ilanlar: data.FEATURES ve price; fiyat yaşa, kilometreye ve araç modeline bağlı, gürültü yaşla büyür."""
    rng = np.random.default_rng(seed)
    name = rng.choice(NAMES, n)
    table = pd.DataFrame({"model": name, "series": [s.split()[0][:2] for s in name]})
    for column in data.CAT:
        table[column] = rng.choice(["a", "b", "c"], n)
    for column in data.NUM:
        table[column] = rng.integers(0, 3, n).astype(float)
    table["vehicle_age"] = rng.integers(0, 15, n).astype(float)
    table["gb_mileage"] = rng.uniform(0, 300_000, n)
    level = pd.Series(name).map({s: i for i, s in enumerate(NAMES)}).to_numpy() / 10
    noise = rng.normal(0, 0.02 + 0.01 * table["vehicle_age"].to_numpy())
    table["price"] = np.round(np.exp(14 + level - 0.06 * table["vehicle_age"] - table["gb_mileage"] / 4e5 + noise))
    return table


@pytest.fixture(scope="module")
def toy():
    """Uydurma ilanlar, kat etiketleri, yaşla büyüyen mutlak log hatalar, log tahmin ve emsal sayısı."""
    table = made_up()
    rng = np.random.default_rng(1)
    fold = np.arange(len(table)) % 5 + 1
    abs_err = np.abs(rng.normal(0, 0.02 + 0.01 * table["vehicle_age"].to_numpy()))
    log_pred = np.log1p(table["price"].to_numpy()) + rng.normal(0, 0.05, len(table))
    comparables = rng.integers(0, 40, len(table))
    return table, fold, abs_err, log_pred, comparables


def test_interval_is_built_in_log_and_clipped():
    low, high = interval.interval(np.array([1e6, 2e7]), 0.1)
    assert low[0] == pytest.approx(np.expm1(np.log1p(1e6) - 0.1))
    assert high[0] == pytest.approx(np.expm1(np.log1p(1e6) + 0.1))
    assert high[1] == pytest.approx(interval.PRICE_CAP) and low[1] == pytest.approx(np.expm1(np.log1p(1.5e7) - 0.1))
    assert interval.interval(np.array([10.0]), 50.0)[0][0] == 0                       # alt uç 0'ın altına inmez
    assert list(interval.inside(np.r_[low[0], high[0], high[0] + 1], np.full(3, 1e6), 0.1)) == [True, True, False]


def test_conformal_quantile_rank():
    assert interval.conformal_quantile(np.arange(1, 20), 0.9) == 18                    # ceil(20 · 0,9) = 18
    assert interval.conformal_quantile(np.arange(1, 20)[::-1], 0.9) == 18              # sırası önemli değil
    with pytest.raises(ValueError, match="çok küçük"):
        interval.conformal_quantile(np.arange(3), 0.9)                                  # ceil(4 · 0,9) = 4 > 3


def test_single_q_never_sees_the_scored_fold():
    rng = np.random.default_rng(0)
    err, fold = rng.random(2000), np.arange(2000) % 5
    q = interval.cross_single_q(err, fold)
    blown = err.copy()
    blown[fold == 0] *= 100
    q2 = interval.cross_single_q(blown, fold)
    assert np.array_equal(q[fold == 0], q2[fold == 0]) and not np.array_equal(q[fold != 0], q2[fold != 0])
    assert q[fold == 1][0] == interval.conformal_quantile(err[fold != 1])


def test_comparables_leave_out_the_own_fold():
    got = interval.other_fold_counts(["a", "a", "a", "b"], np.array([1, 1, 2, 1]))
    assert got.tolist() == [1, 1, 2, 0]


def test_scale_matrix_layout(toy):
    table, fold, abs_err, log_pred, comparables = toy
    train, valid = np.flatnonzero(fold != 1), np.flatnonzero(fold == 1)
    X_train, X_valid, categorical = interval.scale_matrices(table, train, valid, log_pred, comparables)
    assert list(X_train.columns) == data.CAT + data.NUM + [interval.LOG_PRED, interval.COMPARABLES]
    assert categorical == data.CAT and list(X_valid.columns) == list(X_train.columns)
    assert np.array_equal(X_valid[interval.LOG_PRED], log_pred[valid])
    assert np.array_equal(X_valid[interval.COMPARABLES], comparables[valid])
    reference, _, _ = model.fold_matrices(table, train, valid, drop=data.TEXT)            # kodlar eğitim kısmından
    assert X_train[data.CAT + data.NUM].equals(reference)


def test_scale_never_sees_the_scored_fold(toy):
    table, fold, abs_err, log_pred, comparables = toy
    s = interval.nested_scale(table, abs_err, log_pred, comparables, fold)
    blown = abs_err.copy()
    blown[fold == 1] *= 100
    s2 = interval.nested_scale(table, blown, log_pred, comparables, fold)
    assert np.array_equal(s[fold == 1], s2[fold == 1]) and not np.array_equal(s[fold != 1], s2[fold != 1])


def test_scale_is_deterministic_floored_and_monotone(toy):
    table, fold, abs_err, log_pred, comparables = toy
    rows = np.arange(len(table))
    grow = abs_err * (1 + comparables / 10)          # emsalle büyüyen hatalar: kısıt yine artmaya izin vermez
    X, _, categorical = interval.scale_matrices(table, rows, rows, log_pred, comparables)
    a, b = (interval.predict_scale(interval.fit_scale(X, grow, categorical), X) for _ in range(2))
    assert np.array_equal(a, b) and a.min() >= interval.S_FLOOR
    fitted = interval.fit_scale(X, grow, categorical)
    steps = [fitted.predict(X.assign(**{interval.COMPARABLES: g})) for g in (0, 5, 20, 39)]
    assert all((later <= earlier + 1e-12).all() for earlier, later in zip(steps, steps[1:]))
    tiny = interval.predict_scale(interval.fit_scale(X, abs_err / 1000, categorical), X)
    assert np.all(tiny == interval.S_FLOOR)


def test_thread_count_does_not_change_the_scale_model(toy, monkeypatch):
    """Ölçek modeli her ağaçta ilanların bir kısmını çekiyor (subsample); sonuç iş parçacığı sayısına bağlı değil."""
    table, _, abs_err, log_pred, comparables = toy
    rows = np.arange(len(table))
    X, _, categorical = interval.scale_matrices(table, rows, rows, log_pred, comparables)
    many = interval.fit_scale(X, abs_err, categorical).predict(X)
    monkeypatch.setitem(interval.SCALE_PARAMS, "n_jobs", 1)
    assert np.array_equal(interval.fit_scale(X, abs_err, categorical).predict(X), many)


def test_cross_scale_has_one_k_per_scored_fold_from_other_folds(toy):
    table, fold, abs_err, log_pred, comparables = toy
    run = interval.cross_scale(table, abs_err, log_pred, comparables, fold)
    assert np.array_equal(run["s"], interval.nested_scale(table, abs_err, log_pred, comparables, fold))
    assert np.allclose(run["scores"], abs_err / run["s"]) and len(run["k_by_fold"]) == folds.N_FOLDS
    for k, k_value in zip(range(1, 6), run["k_by_fold"]):
        assert np.allclose(run["r"][fold == k] / run["s"][fold == k], k_value)
    assert np.all((run["p"] >= 0) & (run["p"] <= 1))
    blown = abs_err.copy()
    blown[fold == 1] *= 100
    again = interval.cross_scale(table, blown, log_pred, comparables, fold)
    assert again["k_by_fold"][0] == run["k_by_fold"][0] and np.array_equal(again["p"][fold == 1], run["p"][fold == 1])
    assert again["k_by_fold"][1] != run["k_by_fold"][1]


def test_fit_scale_all_uses_every_listing(toy):
    table, _, abs_err, log_pred, comparables = toy
    rows = np.arange(len(table))
    X, _, categorical = interval.scale_matrices(table, rows, rows, log_pred, comparables)
    served = interval.fit_scale_all(table, abs_err, log_pred, comparables)
    assert np.array_equal(served.predict(X), interval.fit_scale(X, abs_err, categorical).predict(X))
    assert served.feature_name_ == list(X.columns)


def test_hit_probability_falls_with_s_and_is_shown_rounded_down():
    scores = np.sort(np.random.default_rng(1).exponential(1.0, 5000))
    p = interval.p_within(scores, np.linspace(0.02, 0.4, 50))
    assert np.all(np.diff(p) <= 0) and p.max() <= 1 and p.min() >= 0
    assert interval.p_within(np.array([1.0, 2.0, 3.0, 4.0]), np.array([np.log1p(interval.WITHIN) / 2.0]))[0] == 0.5
    assert list(interval.shown([0.0, 0.29, 0.3, 0.899, 0.95, 1.0])) == [0, 2, 3, 8, 9, 9]


def test_interval_score_and_width():
    pred, r = np.array([1e6, 1e6]), 0.1
    low, high = interval.interval(pred, r)
    inside_only = interval.interval_score(np.array([1e6, 1e6]), pred, r)
    assert inside_only == pytest.approx(0.2)                                           # yalnız genişlik
    outside = interval.interval_score(np.array([1e6, np.expm1(np.log1p(high[1]) + 0.05)]), pred, r)
    assert outside == pytest.approx(0.2 + 0.5 * 20 * 0.05)                             # 2 / (1 - 0,9) = 20 kat ceza
    assert interval.width_pct(pred, r) == pytest.approx((high - low) / pred * 100)


def test_arm_summary_splits_the_misses():
    price = np.array([100.0, 120.0, 80.0, 101.0])
    out = interval.arm_summary(price, np.full(4, 100.0), 0.1, np.array([True, False, False, True]))
    assert out["coverage"] == 50 and out["missed_above"] == 25 and out["missed_below"] == 25
    assert out["coverage_no_comparable"] == 100


def test_reliability_tables():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.3, 0.99, 3000)
    hit = rng.uniform(size=3000) < p                                                     # kalibre bir olasılık
    out = interval.reliability(p, hit)
    assert len(out["deciles"]) == 10 and out["by_shown"]["n"].sum() == 3000
    assert out["mean_predicted"] == pytest.approx(100 * p.mean())
    assert out["mean_observed"] == pytest.approx(100 * hit.mean())
    assert out["brier"] < out["brier_constant"] and 0 < out["skill_pct"] < 100
    assert np.all(np.diff(out["deciles"]["predicted"]) > 0)
    assert out["quantiles"][50] == pytest.approx(100 * np.median(p))


def test_served_examples():
    table = made_up(400)
    rows = np.arange(len(table))
    final = model.fit_final(table, 30)
    log_pred = np.log1p(table["price"].to_numpy())
    comparables = np.full(len(table), 5)
    scale = interval.fit_scale_all(table, np.full(len(table), 0.05) + table["vehicle_age"].to_numpy() / 200,
                                   log_pred, comparables)
    car = table.iloc[[7]][data.FEATURES].reset_index(drop=True)
    cars = pd.concat([car, car.assign(model=car["model"].str[:-1]), car.assign(kb_fuel="never seen")],
                     ignore_index=True)
    counts = {(str(m), int(a)): 12 for m, a in zip(table["model"], table["vehicle_age"])}
    scores = np.sort(np.random.default_rng(3).exponential(1.0, 500))
    out = interval.served_examples(final, scale, 2.0, scores, table, cars, counts)
    X, _, _ = model.fold_matrices(table, rows, rows)
    assert out.loc[0, "price"] == pytest.approx(np.expm1(final.predict(X.iloc[[7]]))[0])   # ilanın kendisi
    assert list(out["comparables"]) == [12, 0, 12]                                       # bir harf eksik: emsal yok
    assert np.all(out["lower_pct"] < 0) and np.all(out["upper_pct"] > 0) and out["shown"].max() <= interval.SHOWN_MAX
    assert out.loc[0, "lower_pct"] == pytest.approx(100 * np.expm1(-2.0 * out.loc[0, "s"]))


@pytest.mark.data
def test_model_age_cells_are_model_year_cells():
    """
    Emsal servis edilirken (araç modeli, yaş) ile aranır, defterde (araç modeli, model yılı) ile sayılır: aynı hücreler.
    """
    listings = data.add_features(data.load_rows())
    by_age = listings.groupby(["model", "vehicle_age"]).ngroups
    by_year = listings.groupby(["model", "gb_year"]).ngroups
    assert by_age == by_year == len(listings[["model", "gb_year", "vehicle_age"]].drop_duplicates())
