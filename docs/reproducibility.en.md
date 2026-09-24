# Reproducibility

> 🇹🇷 Türkçe (varsayılan): [reproducibility.md](reproducibility.md) · ← [README](../README.en.md)

The analysis chain is deterministic, and this was verified **by measurement**. On 2026-09-23 the computation
was split from one file (`build_site_data.py`, 1,600 lines) into one script per question; the new chain was
checked against the old one's output (reference: `archive/backups/referans-2026-09-23/`):
- **metrics:** every script's numbers match their counterpart in the old `site_data.json` /
  `error_drivers.json` / `oof_shap.json` leaf by leaf; the assembled `site_data.json` differs only in the
  bootstrap run time, the removed `noise_floor` and the new run stamp (`meta.generated_at/data_until/run_id`);
- **reports:** the six md files (technical, decision note, SHAP × TR + EN) and all 70 figures are
  **byte-identical** — last full run from scratch (`metrics/` and `data/analysis/` deleted) with `run_all.py`,
  16 min; only the `catboost_native` residue (below) can switch the native column of SHAP §4 to its other state
  on a given run;
- **serving:** the three served models give the same prediction as before on all 29,988 listings
  (difference ₺0); `encoders.pkl`, `lightgbm_tfidf_svd.txt` and the serving README are byte-identical.

Every metrics file comes out the same from the same data, apart from `_meta.generated_at`.

Two patches make that true:
- `ORDER BY ad_id` on the outer select of the dedup query — without it DuckDB does not guarantee row
  order, and because `KFold(shuffle=True, random_state=42)` shuffles *positions*, fold membership
  drifts between runs (`SEED` does not prevent this);
- LightGBM `deterministic=True, force_row_wise=True`.

Run parameters are recorded in `site_data.json` under `meta.repro` (seed · row order · CatBoost
device · n_jobs · CV tree counts · the final model's tree count). **CatBoost does not produce the same trees on GPU and CPU**, which is why the
device is recorded; local CPU is preferred for determinism.

**Known residue (measured 2026-09-23):** `domain.shap.catboost_native` moves slightly between runs in its
small items. The cause is not the SHAP computation (multi- and single-threaded runs are bit-identical) but the
native CatBoost model itself: the models from two runs give **the same prediction** on all 29,988 listings
(difference ₺0, same tree count), yet are stored with a different internal structure, and attribution shifts
between small items (body type 0.0079 ↔ 0.0189, brand 0.0039 ↔ 0.0075; age and mileage identical). Predictions
and metrics do not move; only the native column of SHAP report §4 does (the whole column, since the shares are
normalised; at most ~1.7 points). With the same model file the SHAP report is rebuilt byte for byte. Leaf
values and leaf weights are identical in both files; the difference is in the file's category/text metadata.
The attribution flips between two states, and it still does in the new chain (2026-09-23: two runs came out
in one state, the full run from scratch in the reference's). The technical report only uses LightGBM's SHAP table. SHAP lives
in a separate report (`reports/shap.{tr,en}.md`, 2026-09-21): it speaks **at the data scale** — global importance,
beeswarm, dependence, age cohorts, km × age; §6 keeps one listing's waterfall as an example (6 figures). The
older version with six case breakdowns and three waterfalls is under `archive/analysis-history/shap-v3-oof-vakalar-2026-09-21/`,
together with its script.

**OOF SHAP (2026-09-20; since 2026-09-23 `analysis/shap/02_oof_shap.py`).** The five folds are rebuilt with
`analysis/lib/cv.py`, exactly as in `07_model_comparison`. The rebuilt LightGBM OOF predictions match 07's stored
OOF **to the kuruş** (max difference ₺0.00). The match is checked as a gate on every run — if it fails, the
file is not written. (The CatBoost variants are not rebuilt here.)
