"""
test_06_hedonic.py
EN: Invariants of metrics/06_hedonic.json (technical report §6 hedonic model): the OLS sample is the listings minus
    the dropped ones (hp or cc missing, both counted once); every bootstrap row is consistent (interval ordered, the
    "contains zero" flag matches it, the % effect is exp(β)−1); the headline effects are the bootstrap rows and the
    model-identity table's first column; VIF is ≥ 1 on the same terms centred and uncentred; adding C(model) never
    lowers the in-sample R².
TR: metrics/06_hedonic.json'un değişmezleri (teknik rapor §6 hedonik model): OLS örneklemi ilanlar eksi atılanlar
    (hp ya da cc eksik, ikisi bir kez sayılır); her bootstrap satırı tutarlı (aralık sıralı, "sıfırı içeriyor"
    bayrağı ona uyuyor, % etki exp(β)−1); manşet etkiler bootstrap satırları ve model kimliği tablosunun ilk
    sütunu; VIF ortalanmış ve ortalanmamış aynı terimlerde ≥ 1; C(model) eklemek örneklem içi R²'yi düşürmüyor.
"""
import math

import pytest


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
    assert hr["bootstrap_setup"]["n_boot"] == hr["bootstrap_setup"]["n_requested"]


def test_bootstrap_rows(hr):
    """
    EN: ci_lo ≤ ci_hi; contains_zero ⇔ ci_lo ≤ 0 ≤ ci_hi; % effect = 100·(e^point − 1) (point is rounded).
    TR: ci_lo ≤ ci_hi; contains_zero ⇔ ci_lo ≤ 0 ≤ ci_hi; % etki = 100·(e^point − 1) (point yuvarlanmış).
    """
    for b in hr["bootstrap"]:
        assert b["ci_lo"] <= b["ci_hi"], b["term"]
        assert b["contains_zero"] == (b["ci_lo"] <= 0 <= b["ci_hi"]), b["term"]
        assert b["pct_effect"] == pytest.approx(100 * (math.exp(b["point"]) - 1), abs=0.02), b["term"]


def test_headline_matches_bootstrap(doc, hr):
    """
    EN: The headline R² and effects are the bootstrap rows'; the engine effect is the headline rounded; the model
        identity table's first column is the headline.
    TR: Manşet R² ve etkiler bootstrap satırlarınınki; motor etkisi yuvarlanmış manşet; model kimliği tablosunun ilk
        sütunu manşet.
    """
    hd = doc["domain"]["hedonic"]
    effect = {b["term"]: b["pct_effect"] for b in hr["bootstrap"]}
    assert hd["r2"] == hr["model_r2"]
    heads = {"age_pct": "yaş", "damage_pct": "ağır hasar", "km100k_pct": "km(100K)", "hp100_pct": "+100 HP",
             "cc_litre_pct": "+1 litre"}
    for key, term in heads.items():
        assert hd[key] == effect[term], key
    assert hr["engine_effect"]["hp100_pct"] == round(hd["hp100_pct"], 1)
    assert hr["engine_effect"]["cc_litre_pct"] == round(hd["cc_litre_pct"], 1)
    coef = hr["with_model"]["coef_pct"]
    for key, head in (("age", "age_pct"), ("km100k", "km100k_pct"), ("hp100", "hp100_pct"), ("cc_litre", "cc_litre_pct"),
                      ("heavy_damage", "damage_pct")):
        assert coef[key][0] == hd[head], key


def test_vif(hr):
    """EN: Same terms centred and uncentred, all ≥ 1. / TR: Ortalanmış ve ortalanmamış aynı terimler, hepsi ≥ 1."""
    assert [t for t, _ in hr["vif"]] == [t for t, _ in hr["vif_raw"]]
    assert all(x >= 1 for _, x in hr["vif"] + hr["vif_raw"])


def test_model_identity(hr):
    """EN: Adding C(model) never lowers the in-sample R². / TR: C(model) eklemek örneklem içi R²'yi düşürmez."""
    assert hr["with_model"]["r2"] >= hr["model_r2"]
    assert hr["with_model"]["n_model"] > 0
