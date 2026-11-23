"""
test_07_final_model.py
EN: Invariants of metrics/07_final_model.json (technical report §7, the served models): the final LightGBM uses
    the median of the CV fold tree counts.
TR: metrics/07_final_model.json'un değişmezleri (teknik rapor §7, servis modelleri): son LightGBM CV fold ağaç
    sayılarının medyanını kullanıyor.
"""
import statistics

import pytest


@pytest.fixture(scope="module")
def final(metrics):
    """EN: The script's metrics. / TR: Betiğin metrikleri."""
    return metrics("07_final_model")


def test_tree_count_is_cv_median(final, metrics):
    """EN: final trees = int(median of the 5 fold tree counts). / TR: son ağaç = int(5 fold ağaç sayısının medyanı)."""
    cv_trees = metrics("07_model_comparison")["meta"]["repro"]["cv_trees"]
    assert final["meta"]["repro"]["final_lgb_trees"] == int(statistics.median(cv_trees))


def test_served_interval_matches_the_analysis(final, metrics):
    """
    EN: The bundle serves what §8 measured: its k is 08_conformal_coverage's k, with the same floor and tolerance;
        one nested score per listing; the comparables table counts every listing once (its cells are the (model, age)
        cells — the script stops if they are not the analysis's (model, year) cells); the data date is the last
        snapshot. That the served s equals lib.conformal's on every listing is checked by the script itself at build
        time (it stops otherwise), not here.
    TR: Paket §8'in ölçtüğünü servis eder: k'sı 08_conformal_coverage'ın k'sı, aynı alt sınır ve toleransla; ilan
        başına bir iç içe skor; emsal tablosu her ilanı bir kez sayar (hücreleri (model, yaş) hücreleri — analizin
        (model, yıl) hücreleri değilse betik durur); veri tarihi son tarama. Servis edilen s'nin her ilanda
        lib.conformal'ınkine eşit olduğunu betik derleme anında kendisi sınar (değilse durur), burada değil.
    """
    si, meta = final["report"]["served_interval"], metrics("01_dedup_leakage")["meta"]
    conf = metrics("08_conformal_coverage")["report"]
    assert si["k"] == conf["scale"]["k"] and si["s_floor"] == conf["scale"]["s_floor"]
    assert si["within_pct"] == conf["confidence"]["within_pct"]
    assert si["n_scores"] == si["comparable_total"] == meta["n_dedup"] and 0 < si["comparable_cells"] <= meta["n_dedup"]
    assert si["data_until"] == meta["snapshots"][-1]


def test_served_examples(final):
    """
    EN: The worked examples: a car with many comparables, the same car where its model has no listing of that age,
        with a misspelt model name, and with a category value never seen. Without comparables the margin is wider and
        the hit probability lower; margins are ordered; the shown value is the probability rounded down, at most 9.
    TR: İşlenmiş örnekler: bol emsalli bir araç, aynı aracın modelinin o yaşta hiç ilanı olmayan hali, model adı yanlış
        yazılmış hali ve hiç görülmemiş bir kategori değeri taşıyan hali. Emsalsizde pay daha geniş, isabet olasılığı
        daha düşük; paylar sıralı; gösterilen değer olasılığın aşağı yuvarlanmışı, en çok 9.
    """
    ex = {e["id"]: e for e in final["report"]["served_interval"]["examples"]}
    assert list(ex) == ["many_comparables", "no_comparable", "model_typo", "unseen_category"]
    rich = ex["many_comparables"]
    assert rich["comparables"] > 0 and ex["unseen_category"]["comparables"] == rich["comparables"]
    for name in ("no_comparable", "model_typo"):
        assert ex[name]["comparables"] == 0 and ex[name]["upper_percent"] > rich["upper_percent"]
        assert ex[name]["hit_probability"] < rich["hit_probability"]
    for e in ex.values():
        assert e["lower_percent"] < 0 < e["upper_percent"] and e["price"] > 0
        assert e["shown"] == min(int(e["hit_probability"] // 10), 9)
