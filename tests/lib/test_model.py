"""
test_model.py
notebooks/lib/model.py küçük, uydurma ilanlarla: kategori kodları ve metin kolonları yalnız eğitim kısmından kurulur;
kat matrisinin kolon sırası ve kategorik listesi; çıkarılan öznitelik matrise girmez, eklenen kolon en sona kategorik
olarak girer, bilinmeyen ad durdurur; bir katın tahmini ağaçları öteki katlarda büyüyen modelden ve tahmin modelinin
kullandığı bir kolonu çıkarmak hatayı büyütür (eski "tesisat" denetimi); son model istenen ağaç sayısıyla bütün
ilanlarda kurulur;
hata ölçüleri; medyan tabanının merdiveni; eşli bootstrap aynı çekilişi kullanır; base_oof temel koşumun kaydını okur.
Gerçek veri ve data/cache okunmaz; eğitimler birkaç yüz satırlık.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from lib import cache, data, folds, model

NAMES = ["320i Sport", "320d Luxury", "520d Sport", "520i Base", "A4 35 TFSI", "A4 40 TDI", "A6 45 TDI", "A6 Base",
         "X5 30d", "Q5 40 TDI"]


def made_up(n=400, seed=0):
    """Uydurma ilanlar: data.FEATURES ve price; fiyat yaşa, kilometreye ve araç modeline bağlı."""
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
    table["price"] = np.round(np.exp(14 + level - 0.06 * table["vehicle_age"] - table["gb_mileage"] / 4e5
                                     + rng.normal(0, 0.05, n)))
    return table


@pytest.fixture(scope="module")
def table():
    return made_up()


def test_category_codes_come_from_the_training_part():
    train_codes, other_codes = model.category_codes(pd.Series(["b", "a", "b", "c"]), pd.Series(["a", "d", "c"]))
    assert list(train_codes) == [0, 1, 0, 2]                         # ilk görülme sırası
    assert list(other_codes) == [1, -1, 2]                           # eğitimde görülmemiş değer -1
    with pytest.raises(ValueError, match="boş değer"):
        model.category_codes(pd.Series(["a", None]), pd.Series(["a"]))


def test_text_columns_are_fitted_on_the_training_part(table):
    train, other_a = model.text_columns(table["model"][:300], table["model"][300:], "model", 150)
    train_again, other_b = model.text_columns(table["model"][:300], pd.Series(["tamamen başka bir ad"] * 5),
                                              "model", 150)
    assert np.array_equal(train.to_numpy(), train_again.to_numpy())  # öteki kısım eğitim kolonlarını değiştirmez
    assert list(train.columns) == [f"model_{i}" for i in range(train.shape[1])]
    assert train.shape[1] < 150 and other_a.shape == (100, train.shape[1])   # az terimde boyut kırpılır


def test_fold_matrix_layout(table):
    extra = table.assign(kb_color=np.where(np.arange(len(table)) < 390, "Beyaz", "Siyah"))
    train, valid = np.arange(380), np.arange(380, 400)
    X_train, X_valid, categorical = model.fold_matrices(extra, train, valid, added=["kb_color"])
    text = [c for c in X_train.columns if c.startswith(("model_", "series_"))]
    assert list(X_train.columns) == data.CAT + data.NUM + text + ["kb_color"]
    assert categorical == data.CAT + ["kb_color"] and list(X_valid.columns) == list(X_train.columns)
    assert set(X_train["kb_color"]) == {0} and set(X_valid["kb_color"]) == {0, -1}   # Siyah eğitimde yok
    X_train, _, categorical = model.fold_matrices(table, train, valid, drop=["model", "brand", "gb_mileage"])
    assert not [c for c in X_train.columns if c.startswith("model_") or c in ("brand", "gb_mileage")]
    assert "brand" not in categorical and any(c.startswith("series_") for c in X_train.columns)


def test_fold_matrices_stop_on_unknown_names(table):
    rows = np.arange(len(table))
    with pytest.raises(ValueError, match="bilinmeyen"):
        model.fold_matrices(table, rows, rows, drop=["gb_milage"])            # yazım hatası sessizce geçmez
    with pytest.raises(ValueError, match="bilinmeyen"):
        model.fold_matrices(table, rows, rows, added=["kb_color"])            # tabloda yok
    with pytest.raises(ValueError, match="bilinmeyen"):
        model.fold_matrices(table, rows, rows, added=["brand"])               # zaten öznitelik


def test_fold_prediction_comes_from_the_other_folds(table):
    """
    Bir katın tahmini, ağaçları yalnız öteki katlarda büyüyen modelden: doğrulama kısmı hiç verilmeden, katın ağaç
    sayısıyla kurulan model aynı tahmini verir. Katın kendi hedefi yalnız durma noktasını (ağaç sayısını) seçer.
    """
    cv = folds.make_folds(len(table))
    run = model.oof_lightgbm(table, cv)
    y = np.log1p(table["price"].to_numpy())
    train, valid = cv[0]
    X_train, X_valid, categorical = model.fold_matrices(table, train, valid)
    alone = lgb.LGBMRegressor(**{**model.LGB_PARAMS, "n_estimators": run["trees"][0]})
    alone.fit(X_train, y[train], categorical_feature=categorical)
    assert np.allclose(alone.predict(X_valid), run["pred_log"][valid])
    assert len(run["trees"]) == folds.N_FOLDS and all(0 < t <= model.LGB_PARAMS["n_estimators"] for t in run["trees"])


def test_thread_count_does_not_change_lightgbm(table, monkeypatch):
    cv = folds.make_folds(len(table))
    many = model.oof_lightgbm(table, cv)
    monkeypatch.setitem(model.LGB_PARAMS, "n_jobs", 1)
    one = model.oof_lightgbm(table, cv)
    assert np.array_equal(one["pred_log"], many["pred_log"]) and one["trees"] == many["trees"]


def test_removing_a_used_column_reaches_the_model(table):
    """Kolonu çıkarmak tahmin modeline ulaşıyor: fiyatı belirleyen kilometre çıkınca tahminler değişir, hata büyür."""
    cv = folds.make_folds(len(table))
    price = table["price"].to_numpy()
    full = model.oof_lightgbm(table, cv)
    without = model.oof_lightgbm(table, cv, drop=["gb_mileage"])
    rmse = [model.price_metrics(price, np.expm1(r["pred_log"]))["rmse"] for r in (full, without)]
    assert rmse[1] > rmse[0]


def test_fit_final_uses_every_listing_and_the_given_trees(table):
    final = model.fit_final(table, 17)
    rows = np.arange(len(table))
    X, _, _ = model.fold_matrices(table, rows, rows)
    assert final.booster_.num_trees() == 17 and final.feature_name_ == list(X.columns)
    assert np.allclose(final.predict(X), final.booster_.predict(X.to_numpy()))


def test_price_metrics():
    actual, pred = np.array([100.0, 200.0, 400.0]), np.array([110.0, 180.0, 400.0])
    m = model.price_metrics(actual, pred)
    assert m["mae"] == pytest.approx(10) and m["medae"] == pytest.approx(10)
    assert m["rmse"] == pytest.approx(np.sqrt(500 / 3)) and m["mape"] == pytest.approx(20 / 3)
    assert model.price_metrics(actual, actual)["r2"] == 1.0


def test_median_baseline_steps_down_the_ladder():
    key = ["x", "x", "x", "y", "x", "z"]
    year = [2020, 2020, 2018, 2019, 2020, 2020]
    price = [100.0, 300.0, 50.0, 80.0, 999.0, 999.0]
    cv = [(np.array([0, 1, 2, 3]), np.array([4, 5])), (np.array([1, 2, 3, 4, 5]), np.array([0]))]
    pred, rung = model.median_baseline(key, year, price, cv)
    assert list(rung[[4, 5, 0]]) == ["key_year", "overall", "key_year"]
    assert pred[4] == 200 and pred[5] == 90 and pred[0] == 649.5  # hücre medyanı; genel medyan; öteki katın hücresi
    cv = [(np.array([0, 1, 2]), np.array([3]))]
    pred, rung = model.median_baseline(["x", "x", "x", "x"], [2020, 2020, 2018, 2019], [100.0, 300.0, 50.0, 1.0], cv)
    assert rung[3] == "key" and pred[3] == 100                      # yıl yok: anahtarın bütün yılları
    with pytest.raises(ValueError, match="model yılı"):
        model.median_baseline(["x"], [np.nan], [1.0], cv)


def test_paired_bootstrap_uses_one_draw_for_every_variant():
    rng = np.random.default_rng(1)
    actual = rng.uniform(100, 200, 50)
    base, worse = actual * 1.05, actual * 1.10
    out = model.paired_bootstrap(actual, base, {"same": base.copy(), "worse": worse}, n_resamples=30, seed=7)
    assert np.all(out["same"]["rmse"] == 0) and np.all(out["same"]["mape"] == 0)
    assert np.allclose(out["worse"]["mape"], 5)                      # her ilanda 5 puan: her örneklemde 5
    rows = np.random.default_rng(7).integers(0, 50, size=50)         # ilk örneklemin çekilişi
    expected = np.sqrt(((actual - worse)[rows] ** 2).mean()) - np.sqrt(((actual - base)[rows] ** 2).mean())
    assert out["worse"]["rmse"][0] == pytest.approx(expected) and len(out["worse"]["rmse"]) == 30


def test_base_oof_reads_the_base_record(table, tmp_path, monkeypatch):
    db = tmp_path / "cars.duckdb"
    db.write_bytes(b"made-up database")
    monkeypatch.setattr(cache, "DB_PATH", db)
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    listings = table.assign(description_text="", kb_color="Beyaz")   # öznitelik dışı kolonlar anahtara girmez
    cv = folds.make_folds(len(table))
    run = cache.load_or_run(model.oof_lightgbm, listings[data.FEATURES + ["price"]], cv)
    oof = model.base_oof(listings)
    assert len(list((tmp_path / "cache").glob("*.pkl"))) == 1        # aynı kayıt: yeniden koşmadı
    assert np.array_equal(oof["pred_log"], run["pred_log"])
    assert np.allclose(oof["pred_price"], np.expm1(run["pred_log"]))
    assert all((oof["fold"].to_numpy()[valid] == k).all() for k, (_, valid) in enumerate(cv, 1))
