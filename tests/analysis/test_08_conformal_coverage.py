"""
test_08_conformal_coverage.py
EN: Invariants of metrics/08_conformal_coverage.json (technical report §8, pre-registered plan
    plans/08-mondrian-coverage). The calibration never sees the scored fold (checked on the script's own functions:
    changing one fold's errors cannot change that fold's q). The four predicted-price bands are equal-sized, so their
    coverages average to the overall one for each arm; the overall coverage of each arm sits at the target within
    3 binomial SE (cross-calibrated, so this is measured, not true by construction); the band bounds increase; the
    served q is positive. Per-band coverage of the single-q arm is NOT required to hit the target: the cheap band's
    under-coverage is a real finding.
TR: metrics/08_conformal_coverage.json'un değişmezleri (teknik rapor §8, ön kayıtlı plan plans/08-mondrian-coverage).
    Kalibrasyon puanlanan katı hiç görmez (betiğin kendi fonksiyonlarında sınanır: bir katın hatalarını değiştirmek o
    katın q'sunu değiştiremez). Tahmin fiyatının dört bandı eşit büyüklükte; her kolda kapsamalarının ortalaması
    geneli verir; her kolun genel kapsaması hedefte, 3 binom SH içinde (çapraz kalibreli, yani tanım gereği değil
    ölçülmüş); bant sınırları artar; servis edilen q pozitif. Tek q kolunun bant kapsamasının hedefi tutması
    BEKLENMEZ: ucuz banttaki düşük kapsama gerçek bir bulgu.
EN (scale model, 2026-11-23): on lib.conformal's own functions — comparables are counted without the listing's own
    fold; the s that scores a fold never sees that fold's errors (this proves the s level only: the errors themselves
    come from the main model's cross-fitting, and the leak-free replica's number is in docs/decisions.md); s cannot
    rise with the comparables; the floor is applied; fitting twice gives the same s (a determinism test, not a
    stability test: another seed moves a car's margin); the hit probability cannot rise with s; one k per scored fold.
    On the metrics — the scale arm sits at the target within 3 binomial SE; its band coverages average to its overall
    coverage; the comparables buckets and the shown values partition the listings; the published partial-dependence
    curve does not rise; the what-if table starts at the measured coverages.
TR (ölçek modeli, 2026-11-23): lib.conformal'ın kendi fonksiyonlarında — emsal, ilanın kendi katı sayılmadan sayılır;
    bir katı puanlayan s o katın hatalarını hiç görmez (bu yalnız s düzeyini kanıtlar: hataların kendisi ana modelin
    çapraz uydurmasından gelir, sızıntısız kopyanın sayısı docs/decisions.md'de); s emsal arttıkça artamaz; alt sınır
    uygulanır; iki kez eğitmek aynı s'yi verir (determinizm testi, kararlılık testi değil: başka bir tohum bir aracın
    payını oynatır); isabet olasılığı s arttıkça artamaz; puanlanan kat başına tek k.
    Metriklerde — ölçek kolu hedefte, 3 binom SH içinde; bant kapsamalarının ortalaması kendi genelini verir; emsal
    kovaları ve gösterilen değerler ilanları bölümler; yayımlanan kısmi bağımlılık eğrisi artmaz; varsayım tablosu
    ölçülen kapsamalarla başlar.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "analysis" / "08_conformal_coverage.py"


@pytest.fixture(scope="module")
def conf(metrics):
    """EN: The script's metrics. / TR: Betiğin metrikleri."""
    return metrics("08_conformal_coverage")


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
        head = SCRIPT.read_text(encoding="utf-8").split("# %% [4]")[0]
        ns = {"__name__": "conformal_functions"}
        exec(compile(head, str(SCRIPT), "exec"), ns)
    finally:
        sys.path.remove(str(ROOT / "analysis"))
        for k in [k for k in sys.modules if k == "lib" or k.startswith("lib.")]:
            del sys.modules[k]
        sys.modules.update(saved)
    return ns


def test_calibration_never_sees_the_scored_fold(functions):
    """
    EN: Blowing up fold 0's errors changes no q given to fold 0 (both arms), but does change the q of other folds.
    TR: 0. katın hatalarını büyütmek 0. kata verilen hiçbir q'yu değiştirmez (iki kol), öteki katlarınkini değiştirir.
    """
    rng = np.random.default_rng(0)
    n = 4000
    err, fold, band = rng.random(n), np.arange(n) % 5, rng.integers(0, 4, n)
    qg, qb = functions["cross_conformal"](err, fold, band)
    err2 = err.copy()
    err2[fold == 0] *= 100
    qg2, qb2 = functions["cross_conformal"](err2, fold, band)
    assert np.array_equal(qg[fold == 0], qg2[fold == 0]) and np.array_equal(qb[fold == 0], qb2[fold == 0])
    assert not np.array_equal(qg[fold != 0], qg2[fold != 0])


def test_conformal_quantile_rank(functions):
    """EN: The ceil((n+1)·0.9)-th smallest: 1..19 → 18. / TR: En küçükten ceil((n+1)·0,9)'uncu: 1..19 → 18."""
    assert functions["conformal_quantile"](np.arange(1, 20), 0.9) == 18


def test_bands_average_to_overall(conf):
    """
    EN: Four equal bands; for each arm the mean band coverage is the overall coverage.
    TR: Dört eşit bant; her kolda bant kapsamalarının ortalaması genel kapsama.
    """
    m = conf["report"]["mondrian"]
    rows = m["by_band"]
    assert [r[0] for r in rows] == ["Q1", "Q2", "Q3", "Q4"] and len({r[1] for r in rows}) == 1
    assert sum(r[2] for r in rows) / 4 == pytest.approx(conf["report"]["conformal_all"], abs=0.02)
    assert conf["domain"]["conformal"]["by_quantile"] == [[r[0], r[2]] for r in rows]
    assert conf["domain"]["conformal"]["mondrian_by_quantile"] == [[r[0], r[3]] for r in rows]


def test_overall_coverage_at_target(conf, metrics):
    """EN: Each arm: |coverage − target| ≤ 3 binomial SE. / TR: Her kol: |kapsama − hedef| ≤ 3 binom SH."""
    n = metrics("01_dedup_leakage")["meta"]["n_dedup"]
    target = conf["domain"]["conformal"]["coverage_target"]
    se = 100 * math.sqrt(target / 100 * (1 - target / 100) / n)
    mondrian_all = sum(r[3] for r in conf["report"]["mondrian"]["by_band"]) / 4      # four equal bands | dört eşit bant
    for cov in (conf["report"]["conformal_all"], mondrian_all):
        assert abs(cov - target) <= 3 * se


def test_accept_band_is_the_plan(conf):
    """
    EN: The published acceptance band is the pre-registered one (plans/08-mondrian-coverage, H1).
    TR: Yayımlanan kabul aralığı ön kayıttaki (plans/08-mondrian-coverage, H1).
    """
    plan = (ROOT / "plans" / "08-mondrian-coverage" / "analysis_plan.json").read_text(encoding="utf-8")
    lo, hi = conf["report"]["mondrian"]["accept_band"]
    assert f">= {lo} and <= {hi} (percent)" in plan


def test_band_bounds_and_q(conf):
    """
    EN: Three increasing cut points; the served q positive.
    TR: Artan üç kesim noktası; servis edilen q pozitif.
    """
    b = conf["report"]["q_bounds"]
    assert len(b) == 3 and b == sorted(b) and len(set(b)) == 3
    assert conf["report"]["conformal_q"] > 0


# ── Scale model | Ölçek modeli ─────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def toy(functions):
    """
    EN: A small synthetic market for the scale-model functions: 2,000 listings, two categorical and two numeric
        inputs of the main model, errors that grow with age. Returns: (CF, X, abs_err, log_pred, comparables, fold).
    TR: Ölçek modeli fonksiyonları için küçük bir yapay piyasa: 2.000 ilan, ana modelin iki kategorik ve iki sayısal
        girdisi, yaşla büyüyen hatalar. Döndürür: (CF, X, abs_err, log_pred, comparables, fold).
    """
    import pandas as pd
    rng = np.random.default_rng(0)
    n = 2000
    X = pd.DataFrame({"brand": rng.choice(["a", "b"], n), "kb_fuel": rng.choice(["x", "y", "z"], n),
                      "vehicle_age": rng.integers(0, 25, n), "gb_mileage": rng.uniform(0, 3e5, n)})
    abs_err = np.abs(rng.normal(0, 0.03 + 0.004 * X["vehicle_age"].values))
    return functions["CF"], X, abs_err, rng.uniform(13, 15, n), rng.integers(0, 60, n), np.arange(n) % 5


def test_comparables_leave_out_the_own_fold(functions):
    """
    EN: Three listings of cell a (two in fold 0, one in fold 1) and one of cell b: 1, 1, 2 and 0 comparables.
    TR: a hücresinden üç ilan (ikisi 0. katta, biri 1. katta) ve b hücresinden bir ilan: 1, 1, 2 ve 0 emsal.
    """
    got = functions["CF"].other_fold_counts(["a", "a", "a", "b"], np.array([0, 0, 1, 0]))
    assert got.tolist() == [1, 1, 2, 0]


def test_scale_never_sees_the_scored_fold(toy):
    """
    EN: Blowing up fold 0's errors leaves the s given to fold 0 as it was and changes the s of the other folds.
    TR: 0. katın hatalarını büyütmek 0. kata verilen s'yi olduğu gibi bırakır, öteki katlarınkini değiştirir.
    """
    CF, X, err, lp, comp, fold = toy
    _, s = CF.nested_scores(X, err, lp, comp, fold)
    err2 = err.copy()
    err2[fold == 0] *= 100
    _, s2 = CF.nested_scores(X, err2, lp, comp, fold)
    assert np.array_equal(s[fold == 0], s2[fold == 0]) and not np.array_equal(s[fold != 0], s2[fold != 0])


def test_scale_is_deterministic_floored_and_monotone(toy):
    """
    EN: Two fits give the same s; s is never below the floor (and sits on it when the errors are tiny); raising the
        comparables of every listing never raises s, even when the errors were made to grow with the comparables.
    TR: İki eğitim aynı s'yi verir; s alt sınırın altına inmez (hatalar çok küçükse sınırda durur); her ilanın
        emsalini artırmak s'yi hiç artırmaz, hatalar emsalle büyüyecek biçimde kurulmuş olsa bile.
    """
    CF, X, err, lp, comp, fold = toy
    idx = np.arange(len(err))
    grow = err * (1 + comp / 20)                       # errors that rise with the comparables | emsalle artan hatalar
    Xtr, Xva, catc = CF.scale_matrix(X, idx, idx, lp, comp)
    a, b = (CF.predict_scale(CF.fit_scale(Xtr, grow, catc), Xva) for _ in range(2))
    assert np.array_equal(a, b) and a.min() >= CF.S_FLOOR
    model = CF.fit_scale(Xtr, grow, catc)
    steps = [model.predict(Xva.assign(**{CF.COMPARABLES: g})) for g in (0, 5, 20, 59)]
    assert all((later <= earlier + 1e-12).all() for earlier, later in zip(steps, steps[1:]))
    tiny = CF.predict_scale(CF.fit_scale(Xtr, err / 1000, catc), Xva)
    assert np.all(tiny == CF.S_FLOOR)


def test_hit_probability_falls_with_s(functions):
    """
    EN: A wider expected error cannot raise the chance of landing within ±10%; the value is a share between 0 and 1.
    TR: Daha büyük beklenen hata ±%10 içinde kalma olasılığını artıramaz; değer 0 ile 1 arasında bir pay.
    """
    CF = functions["CF"]
    scores = np.sort(np.random.default_rng(1).exponential(1.0, 5000))
    p = CF.p_within(scores, np.linspace(0.02, 0.4, 50))
    assert np.all(np.diff(p) <= 0) and p.max() <= 1 and p.min() >= 0
    assert CF.p_within(np.array([1.0, 2.0, 3.0, 4.0]), np.array([np.log1p(CF.WITHIN) / 2.0]))[0] == 0.5


def test_one_k_per_scored_fold(functions, toy):
    """
    EN: In the cross-calibrated scale arm, half-width / s is one number per scored fold (its k), and changing a
        fold's errors does not change that fold's k.
    TR: Çapraz kalibreli ölçek kolunda yarı genişlik / s, puanlanan kat başına tek sayı (o katın k'sı); bir katın
        hatalarını değiştirmek o katın k'sını değiştirmez.
    """
    CF, X, err, lp, comp, fold = toy
    _, s = CF.nested_scores(X, err, lp, comp, fold)
    r, p = functions["cross_scale"](X, err, lp, comp, fold, s)
    ks = [np.unique(np.round(r[fold == k] / s[fold == k], 10)) for k in range(5)]
    assert all(len(k) == 1 for k in ks) and np.all((p >= 0) & (p <= 1))
    err2 = err.copy()
    err2[fold == 0] *= 100
    r2, _ = functions["cross_scale"](X, err2, lp, comp, fold, s)
    assert np.allclose(r2[fold == 0] / s[fold == 0], ks[0][0])


def test_scale_arm_at_target_and_consistent(conf, metrics):
    """
    EN: The scale arm: overall coverage within 3 binomial SE of the target; its four equal bands average to it; the
        global arm's row is the published single-q coverage; the figure rows are the band rows; k and the floor are
        positive and the floor-hit count is a count.
    TR: Ölçek kolu: genel kapsama hedefin 3 binom SH içinde; dört eşit bandının ortalaması onu verir; genel kolun
        satırı yayımlanan tek q kapsaması; figür satırları bant satırları; k ve alt sınır pozitif, alt sınıra değen
        ilan sayısı bir sayım.
    """
    n = metrics("01_dedup_leakage")["meta"]["n_dedup"]
    sc, target = conf["report"]["scale"], conf["domain"]["conformal"]["coverage_target"]
    arms = {r[0]: r for r in sc["arms"]}
    assert list(arms) == ["global", "band", "scale"] and len(sc["arm_columns"]) == len(sc["arms"][0])
    assert abs(arms["scale"][1] - target) <= 3 * 100 * math.sqrt(target / 100 * (1 - target / 100) / n)
    assert arms["global"][1] == conf["report"]["conformal_all"]
    assert sum(r[2] for r in sc["by_band"]) / 4 == pytest.approx(arms["scale"][1], abs=0.02)
    assert conf["domain"]["conformal"]["scale_by_quantile"] == [[r[0], r[2]] for r in sc["by_band"]]
    assert sc["k"] > 0 and sc["s_floor"] > 0 and 0 <= sc["floor_hits"] <= n


def test_scale_groups_partition_the_listings(conf, metrics):
    """
    EN: The comparables buckets add up to all listings and the first one is the no-comparable count the arms use; the
        "all" profile is every listing; margins are ordered; the latest snapshot's listings are a part of the data.
    TR: Emsal kovaları bütün ilanlara toplanır ve ilki kolların kullandığı emsalsiz sayısıdır; "all" profili her
        ilan; paylar sıralı; son taramanın ilanları verinin bir parçası.
    """
    meta = metrics("01_dedup_leakage")["meta"]
    sc = conf["report"]["scale"]
    assert sum(r[1] for r in sc["by_comparables"]) == meta["n_dedup"]
    assert sc["by_comparables"][0][1] == sc["no_comparable_n"]
    prof = {r[0]: r for r in sc["profiles"]}
    assert prof["all"][1] == meta["n_dedup"] and prof["no_comparable"][1] == sc["no_comparable_n"]
    assert all(len(r) == len(sc["profile_columns"]) and 0 < r[4] < r[5] and r[8] < r[9] for r in sc["profiles"])
    assert sc["latest_snapshot"]["snap"] == meta["snapshots"][-1] and 0 < sc["latest_snapshot"]["n"] < meta["n_dedup"]


def test_published_curve_and_what_if(conf):
    """
    EN: s against the comparables does not rise (the constraint, as published); width percentiles rise; the what-if
        table's first row is no shift and repeats the measured coverages and hit share.
    TR: Emsale göre s artmaz (kısıt, yayımlandığı haliyle); genişlik yüzdelikleri artar; varsayım tablosunun ilk
        satırı kaymasızdır ve ölçülen kapsamaları ve isabet payını yineler.
    """
    sc, cf = conf["report"]["scale"], conf["report"]["confidence"]
    curve = [v for _, v in sc["partial_dependence"]]
    assert all(b <= a for a, b in zip(curve, curve[1:]))
    widths = [r[1] for r in sc["width"]["percentiles"]]
    assert widths == sorted(widths) and sc["width"]["min"] <= widths[0]
    arms = {r[0]: r for r in sc["arms"]}
    first = sc["level_shift"][0]
    assert first[0] == 0 and first[1] == arms["global"][1] and first[2] == arms["scale"][1]
    assert first[5] == pytest.approx(cf["mean_observed"], abs=0.06)


def test_hit_probability_tables(conf, metrics):
    """
    EN: Ten reliability points with rising predictions; the Brier score beats the constant; the shown values run
        from 0 to shown_max at most and their counts add up to all listings.
    TR: Öngörüsü artan on güvenilirlik noktası; Brier skoru sabiti geçer; gösterilen değerler 0'dan en çok
        shown_max'e kadar ve sayıları bütün ilanlara toplanır.
    """
    cf, points = conf["report"]["confidence"], conf["domain"]["conformal"]["hit_reliability"]
    assert len(points) == 10 and [p for p, _ in points] == sorted(p for p, _ in points)
    assert all(0 <= v <= 100 for pair in points for v in pair)
    assert cf["brier"] < cf["brier_constant"] and cf["within_pct"] == 10
    assert all(0 <= r[0] <= cf["shown_max"] for r in cf["by_shown"])
    assert sum(r[1] for r in cf["by_shown"]) == metrics("01_dedup_leakage")["meta"]["n_dedup"]
