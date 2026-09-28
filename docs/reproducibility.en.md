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

Every generated JSON says when it was written, in the same format: a top-level `_meta.generated_at`, local time
with its UTC offset, to the second (e.g. `2026-09-25T04:03:29+03:00`). Covered: the metrics,
`data/analysis/oof_info.json`, `data/site_data.json`, `data/serving/column_labels.json` and the S3 manifest. In
builder outputs the stamp is the build's own time. `site_data.json` also keeps `meta.metrics_generated_at`: the
newest metrics stamp. The regeneration tests leave only this field out of the comparison.
`tests/metrics/test_generated_stamps.py` pins the format. `python tools/analysis_coverage.py` shows when each
script last ran.

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

**Verification gate and metrics baseline (2026-09-24).** Determinism is not measured once and left alone; it
is checked on every change by `tools/verify.py`. `tests/baselines/` holds two snapshots of the 24 metrics
files: `metrics_shape.json` (every key path and its JSON type) and `metrics_fingerprint.json` (the value of
each scalar; for lists, the length and the sha256 of their canonical JSON). The first baseline is the metrics
of commit `b2395e0` on the `restructure-2026-09` branch. Rules:
- the default is **exact equality**. The only keys allowed to move are listed in
  `tests/baselines/exemptions.json`, each with a tr/en reason: `_meta.generated_at` (the run stamp) and
  the `catboost_native*` keys of `shap/04_variants` (the residue above). The bootstrap run time of `06_hedonic`
  left with its key on 2026-09-28 (`--drop-exemption`). An exemption that no longer matches any key fails the test, so no stale exemption stays;
- the baseline changes only through `python tools/snapshot_metrics.py --accept "<reason>"`. The
  `metrics_view` consistency gate must pass first and the reason cannot be empty; every accept is logged in
  `tests/baselines/accept_log.jsonl` with the time, the reason, the number of changed keys and the first
  differences;
- reports are a pure function of the metrics. The test runs the builders into a temp folder through the
  `CARDATASYS_OUT` environment variable and compares the output with the repository: the md files ignoring
  line ends, the figures byte for byte, `site_data.json` as JSON. If the reports are not rebuilt after an
  accept, this test fails.

**English key names (2026-09-25).** Metric keys, the values code compares and the names in the builders were a
Turkish/English mix; they are all English snake_case now, and the Turkish text the reports show stayed as it was.
The old→new names are in one reviewed map: `docs/metric-key-renames.json` (path based, because the same word had two
meanings: `alt`/`ust` meant lower/upper for a candidate and low/up for a range). The work went in nine groups, and
each group was proven a **pure rename** against one origin taken once before the work (the P0 copy, outside git in
`archive/backups/renames-P0-2026-09-25/p0/`), with `tools/metric_renames.py`:
- `check-builders`: the four builders run on translated P0 metrics in a temp root; the six md files and every
  figure are byte-identical to P0;
- `check-metrics`: the rerun metrics equal translated P0 leaf by leaf (order and type included; the stamp, the
  source hashes and the exempt residue left out). **0 residual differences** in every group;
- `check-outputs`: the repository reports and figures equal P0, `site_data.json` equals translated P0.

Because these proof commands ran against the archived P0 copy, they were removed from the tool on 2026-09-27
(nothing from the archive reaches the live chain); `tools/metric_renames.py` keeps only the map reader the name
guard uses.

`run_id` did not change. The path and value map of `site_data.json` for the portfolio site is
`docs/site-data-renames.json`. The key `FINAL_LGB_AGAC` in `encoders.pkl` keeps its old name on purpose: it is the
serving/API contract, read outside the repository. `tests/metrics/test_metric_key_names.py` keeps the old names
from coming back.
