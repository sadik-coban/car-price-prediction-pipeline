"""
methodology_checks.py
EN: Heavy methodology checks on the real data, run in their own process by test_methodology_full.py (so
    analysis/lib and db/lib never meet in one pytest session). Each prints one JSON line:
      folds        the 5 folds cover every listing once and no ad_id is in two folds;
      shuffled     LightGBM OOF with the target shuffled: R² on log price must be ≈ 0 (the setup cannot find
                   signal where there is none — a leak would show up here);
      determinism  two identical LightGBM OOF runs give bit-identical predictions and tree counts.
    The setup is the one of analysis/07_model_comparison.py (load_clean, FEATURES, make_folds, lgb_oof).
TR: Gerçek veride ağır metodoloji denetimleri; test_methodology_full.py kendi sürecinde koşar (analysis/lib ile
    db/lib aynı pytest oturumunda karşılaşmasın). Her biri tek bir JSON satırı basar:
      folds        5 fold her ilanı bir kez kapsıyor ve hiçbir ad_id iki fold'da değil;
      shuffled     hedef karıştırılmış LightGBM OOF: log fiyatta R² ≈ 0 olmalı (kurulum olmayan sinyali bulamaz
                   — bir sızıntı burada görünür);
      determinism  iki aynı LightGBM OOF koşusu bit-birebir aynı tahmin ve ağaç sayısı verir.
    Kurulum analysis/07_model_comparison.py'ninkiyle aynı (load_clean, FEATURES, make_folds, lgb_oof).
Run / Koşum:
    python tests/analysis/methodology_checks.py folds|shuffled|determinism
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))

import numpy as np  # noqa: E402

from lib.common import FEATURES, load_clean  # noqa: E402
from lib.cv import lgb_oof, make_folds  # noqa: E402


def r2(y, pred):
    """EN: Coefficient of determination. / TR: Belirlilik katsayısı."""
    return float(1 - np.sum((y - pred) ** 2) / np.sum((y - np.mean(y)) ** 2))


def check_folds(listings):
    """
    EN: Fold coverage and ad_id overlap between folds. / TR: Fold kapsamı ve fold'lar arası ad_id örtüşmesi.
    """
    folds = make_folds(len(listings))
    ads = listings["ad_id"].values
    sets = [set(ads[va]) for _tr, va in folds]
    overlap = sum(len(a & b) for i, a in enumerate(sets) for b in sets[i + 1:])
    seen = np.concatenate([va for _tr, va in folds])
    return {"rows": len(listings), "unique_ad_ids": int(listings["ad_id"].nunique()),
            "rows_in_folds": int(len(seen)), "distinct_rows_in_folds": int(len(set(seen.tolist()))),
            "ad_id_overlap": int(overlap)}


def check_shuffled(listings):
    """EN: OOF R² with the target shuffled (seed 42). / TR: Hedef karıştırılmış OOF R² (tohum 42)."""
    yl = np.log1p(listings["price"].values.astype(float))
    fake = np.random.default_rng(42).permutation(yl)
    out = lgb_oof(listings[FEATURES].copy(), fake, make_folds(len(listings)))
    return {"r2_log_shuffled": r2(fake, out["pred_log"]), "trees": out["iters"]}


def check_determinism(listings):
    """EN: Two identical OOF runs, compared. / TR: İki aynı OOF koşusu, karşılaştırılır."""
    X, yl = listings[FEATURES].copy(), np.log1p(listings["price"].values.astype(float))
    folds = make_folds(len(listings))
    a, b = lgb_oof(X, yl, folds), lgb_oof(X, yl, folds)
    return {"max_abs_diff": float(np.max(np.abs(a["pred_log"] - b["pred_log"]))),
            "trees_equal": a["iters"] == b["iters"], "r2_log": r2(yl, a["pred_log"])}


CHECKS = {"folds": check_folds, "shuffled": check_shuffled, "determinism": check_determinism}

if __name__ == "__main__":
    print(json.dumps(CHECKS[sys.argv[1]](load_clean())))
