"""
test_forward.py
notebooks/lib/forward.py küçük, uydurma ilanlarla: eğitim son taramadan önceki son satırlar, test son taramada ilk kez
görülen ilanlar; az test ilanında durur; fit_predict yalnız eğitim ilanlarından öğrenir; ileri koşumda ağaç sayısı,
skorlar ve k test ilanlarına bağlı değil, emsal eğitim kümesinden sayılır; küme bootstrap'i grupları çeker ve
dizileri eşli çeker. Gerçek veri (data işareti): test ilanları ilan tablosunda aynı fiyatla var.
"""
import numpy as np
import pandas as pd
import pytest

from lib import data, folds, forward, interval, model
from test_interval import made_up


def snapshot_rows():
    """
    Üç tarama: 1-3 numaralı ilanlar ilk iki taramada, 3 ve 4-6 son taramada; 2 ikinci taramada fiyat değiştirir.
    """
    rows = pd.DataFrame({"ad_id": [1, 2, 2, 3, 3, 3, 4, 5, 6],
                         "snap": ["s1", "s1", "s2", "s1", "s2", "s3", "s3", "s3", "s3"],
                         "price": [10, 20, 21, 30, 30, 31, 40, 50, 60]})
    rows["search_date"] = pd.to_datetime(rows["snap"].map({"s1": "2026-01-01", "s2": "2026-02-01", "s3": "2026-03-01"}))
    return rows


def test_split_trains_before_and_tests_on_new_listings():
    train, test, until, test_snap = forward.split(snapshot_rows(), min_test=3)
    assert train["ad_id"].tolist() == [1, 2, 3]
    assert train["price"].tolist() == [10, 21, 30]                       # son satır, son tarama hariç
    assert test["ad_id"].tolist() == [4, 5, 6] and (until, test_snap) == ("s2", "s3")   # 3 daha önce görüldü
    with pytest.raises(ValueError, match="çok az"):
        forward.split(snapshot_rows(), min_test=4)


def test_fit_predict_learns_from_the_training_part_only():
    table = made_up(400)
    rows = np.arange(len(table))
    copies = table.iloc[:20][data.FEATURES]
    got = forward.fit_predict(table, copies, 25)
    final = model.fit_final(table, 25)
    X, _, _ = model.fold_matrices(table, rows, rows)
    assert np.allclose(got, final.predict(X.iloc[:20]))                 # eğitim ilanının kopyası aynı tahmini alır
    other = forward.fit_predict(table, copies.assign(model="tamamen başka bir ad"), 25)
    assert not np.allclose(other, got)                                   # test satırının kendi tahmini değişir


@pytest.fixture(scope="module")
def forward_parts():
    table = made_up(900, seed=4)
    train, test = table.iloc[:700].reset_index(drop=True), table.iloc[700:].reset_index(drop=True)
    cell = lambda d: (d["model"] + "|" + d["vehicle_age"].astype(int).astype(str))  # noqa: E731
    return train, test, cell(train), cell(test)


def test_forward_run_calibrates_on_training_only(forward_parts):
    train, test, train_cells, test_cells = forward_parts
    run = forward.forward_run(train, test[data.FEATURES], train_cells, test_cells)
    cv = model.oof_lightgbm(train, folds.make_folds(len(train)))
    assert run["cv_trees"] == cv["trees"] and run["trees"] == int(np.median(cv["trees"]))
    assert np.array_equal(run["train_pred_log"], cv["pred_log"])
    changed = test.assign(gb_mileage=test["gb_mileage"] * 0.5, model=test["model"].str[:-1])
    again = forward.forward_run(train, changed[data.FEATURES], train_cells, test_cells)
    assert np.array_equal(again["scores"], run["scores"]) and again["trees"] == run["trees"]   # k test satırına bakmaz
    assert not np.array_equal(again["test_pred_log"], run["test_pred_log"])
    counts = train_cells.value_counts()
    assert np.array_equal(run["test_comparables"], test_cells.map(counts).fillna(0).to_numpy(dtype=int))
    fold = run["train_fold"]
    assert sorted(np.unique(fold)) == [1, 2, 3, 4, 5]
    train_err = np.abs(np.log1p(train["price"].to_numpy()) - np.log1p(np.expm1(run["train_pred_log"])))
    s = interval.nested_scale(train, train_err, run["train_pred_log"], interval.other_fold_counts(train_cells, fold),
                              fold)
    assert np.allclose(run["scores"], train_err / s) and run["test_s"].min() >= interval.S_FLOOR


def test_cluster_bootstrap_draws_whole_groups():
    groups = np.array(["a", "a", "b", "b", "c"])
    same = forward.cluster_bootstrap(groups, [np.array([1.0, 1.0, 1.0, 1.0, 1.0])], n_boot=50, seed=0)
    assert same.shape == (50, 1) and np.allclose(same, 1)
    by_group = forward.cluster_bootstrap(groups, [np.array([1.0, 1.0, 2.0, 2.0, 3.0])], n_boot=200, seed=0)[:, 0]
    assert by_group.min() >= 1 and by_group.max() <= 3                  # yalnız grup ortalamalarının karışımı
    values = np.random.default_rng(0).uniform(size=5)
    pair = forward.cluster_bootstrap(groups, [values, values + 1], n_boot=30, seed=1)
    assert np.allclose(pair[:, 1] - pair[:, 0], 1)                       # aynı çekilişler
    first, second = (forward.cluster_bootstrap(groups, [values], 30, 1) for _ in range(2))
    assert np.array_equal(first, second)


@pytest.mark.data
def test_forward_test_listings_are_listings_at_the_same_price():
    """
    4. defter 6.1: ileri testin test ilanları ilan tablosunda da var (son satırları son taramada), fiyatları ve
    öznitelikleri aynı; temel koşumun kat dışı tahminleri bu yüzden aynı ilanlar için okunabilir.
    """
    listings = data.add_features(data.load_rows())
    _, test, _, test_snap = forward.split(data.add_features(data.load_rows(latest_only=False)))
    same = listings.set_index("ad_id").loc[test["ad_id"]]
    assert (same["snap"] == test_snap).all()
    assert np.array_equal(same["price"].to_numpy(), test["price"].to_numpy())
    assert same[data.FEATURES].reset_index(drop=True).equals(test[data.FEATURES].reset_index(drop=True))
