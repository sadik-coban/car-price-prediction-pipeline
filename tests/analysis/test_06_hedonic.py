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
"""
import math

import pytest

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
