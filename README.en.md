# Used car price analysis — from data to report

> 🇹🇷 Türkçe (varsayılan): [README.md](README.md)

This repository carries the **entire** live chain: from collecting listings to the published report.
Before it existed the reports had no producer — their text had been hand-transcribed from Next.js
pages in a different repo, and their charts were screenshots of those pages.

**Golden rule:** no data figure is ever hand-written. Every number is read from JSON or derived in
code. The deliberate exception is flagged in [docs/decisions.md](docs/decisions.md) (hand-written example notes). The dropped-column table
(`feature_drop`) was hand-written until 2026-09-23 and did not match the data; every raw column is now
assigned to exactly one class in code and the counts are computed.

---

## The chain

```
scraper/ ──► data/raw/{audi,bmw}/<date>/details.jsonl
   └─ db/build_duckdb.py (← lib/process_for_db.py) ──► data/cars.duckdb (semi-raw)
        │
        ├─ analysis/NN_*.py        one script per question ──► metrics/NN_*.json
        │     └─ 07_model_comparison ──► data/analysis/oof.parquet (OOF) ──► 07_final_model · 08_* · shap/*
        ├─ analysis/shap/NN_*.py   the SHAP report's questions ──► metrics/shap/*.json + reports/figures/*-sh-*.png
        │
        └─ builders/ (no analysis; read metrics/*.json only)
              ├─ build_site_data.py ──► data/site_data.json
              ├─ build_technical_report.py ──► reports/technical ×{tr,en} · reports/figures
              ├─ build_business_report.py ──► reports/business ×{tr,en} · reports/figures
              └─ build_shap_report.py ──► reports/shap ×{tr,en}

   db/build_gold_db.py (← gold_rules.json): data/cars.duckdb ──► data/cars_gold.duckdb (API contract)
      └─ db/publish_data_to_s3.py ──► S3 "data/cars.duckdb"  (publishing arm; not part of the report chain;
                                       uploads only a file that keeps the gold contract — docs/database.md)
```

## How to run

Every script resolves its paths against `__file__` → **it runs from anywhere**, no `cd` required.

```bash
# 0) environment (at the repo root)
python -m venv .venv && .venv/Scripts/activate      # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements-pipeline.txt

# 1) data collection (optional — only if a new snapshot is needed)
python scraper/main.py                        # -> data/raw/<brand>/<date>/  (scraping code is local only)
python db/build_duckdb.py                     # -> data/cars.duckdb (+ duplicate_ad_ids.csv next to it)
python db/build_gold_db.py                    # -> data/cars_gold.duckdb (what the API gets; the only file publishing uploads)

# 2) everything: the analysis scripts in dependency order + the builders   (~16 min)
python analysis/run_all.py
python analysis/run_all.py --from 07          # from 07 on (+ builders)
python analysis/run_all.py --only 08          # only the 08_* scripts (no builders)

# a single question: writes only its own metrics/<script>.json
python analysis/08_conformal_coverage.py

# 3) reports only (metrics already there, ~1 min)
python builders/build_site_data.py
python builders/build_technical_report.py
python builders/build_business_report.py
python builders/build_shap_report.py

# 4) done? one verdict, must exit 0 (see "Verification gate" below)
python tools/verify.py                        # fast gate (~10 s)
python tools/verify.py --full                 # run_all.py first (~16 min), then the same gate
```

> **Windows note:** the scripts print Turkish characters. If output is redirected to a file,
> `PYTHONIOENCODING=utf-8 PYTHONUTF8=1` is required, otherwise cp1252 raises `UnicodeEncodeError`.

**If the data hasn't changed, regenerating the reports takes about a minute** — the heavy compute stays in
the analysis scripts (heaviest: `07_lofo` ~6 min, `07_model_comparison` ~4 min, the `06_hedonic` bootstrap
~1.5 min, `shap/02` and `shap/04` ~1 min).

### Verification gate

`python tools/verify.py` is the one command that decides whether a change is done: exit 0 means done. It only
selects which tests run and summarises them; every check is a pytest test under `tests/`:
- `tests/db/`, `tests/scraper/`: the raw → DB arm and the collection settings;
- `tests/metrics/`: does every `metrics/*.json` equal the baseline (`tests/baselines/`): key paths, types and
  values. Keys known to move between runs are listed with their reason in `tests/baselines/exemptions.json`;
- `tests/reports/`: the four builders are rerun into a temp folder; the six reports, every figure and
  `site_data.json` must equal what is in the repository. A hand-edited report, or one not rebuilt after a
  metrics change, fails here;
- `tests/repo/`: bilingual docstrings, relative links in the documents, no code that reads another source
  file as data.

If an analysis change moves the numbers **on purpose**, the gate fails and shows the difference. Inspect it
with `python tools/snapshot_metrics.py --diff`, accept it with
`python tools/snapshot_metrics.py --accept "<reason>"`, then rebuild the reports. An accept without a reason
is refused; every accept is logged in `tests/baselines/accept_log.jsonl`. Rules:
[Reproducibility](docs/reproducibility.en.md).

## Documents

- [How to read the analysis scripts](docs/analysis-scripts.en.md) — three layers, cell-by-cell runs, shared code, consistency gate
- [Decisions and limits](docs/decisions.en.md) — business / technical split, why the text analysis is archived, privacy, honest framing
- [Database notes](docs/database.en.md) — semi-raw rules, safe build, **the gold step (API): read before rebuilding the DB**
- [Reproducibility](docs/reproducibility.en.md) — determinism, the reference comparison, the known residue
- [Metric key map](docs/metric-key-renames.json) — old Turkish keys → English (2026-09-25); the `site_data.json`
  path and value map for the portfolio site: [docs/site-data-renames.json](docs/site-data-renames.json)
- Reports: [technical](reports/technical.en.md) · [decision note](reports/business.en.md) · [SHAP](reports/shap.en.md) (Turkish: `*.tr.md`)

---

## Contents

| path | what |
|---|---|
| `scraper/` | listing collection → `data/raw/`; what is collected (brands, price ranges, search query) lives in `collection_config.json`, read by `collection.py` — the analysis (`01_dedup_leakage`) reads the same file. The scraping code (`main.py`, `getlistofcars.py`, `getdetails.py`) is not in the repository, only local; the repository tracks these two config files |
| `db/` | `build_duckdb` (→ semi-raw `cars.duckdb`: every listing the site showed, blue plates included; safe build; stops without writing on a raw value the register has not seen) + `observed_values.json` + `lib/observed_values` (the register of observed values: what the data actually holds, the only cases code may expect — [docs/database.en.md](docs/database.en.md)) + `lib/process_for_db` (raw JSONL → row parsers) + `lib/damage_mappings.json` (the damage diagram's 13 panel / 5 status labels; `analysis/01_unspecified_panels` reads it too) · `build_gold_db` (→ `cars_gold.duckdb`, the API contract: unknown = no/0, blue-plate rows left out; rules and reasons in `gold_rules.json`) · `publish_data_to_s3` (S3 publishing; uploads only a file that keeps the gold contract; `--dry-run` only checks, no S3 connection) + `lib/s3_publish` (the S3 connection) |
| `tests/` | pytest tests — `tests/db/`: the raw → DB arm (every format seen in the real data, the semi-raw rules, an end-to-end build, the safe build) and the S3 publishing arm (validation, versioning, upload order, manifest). Records are fake but in the real formats; everything runs in temp folders; no network, no `.env`, no `data/` · `tests/metrics/`, `tests/reports/`, `tests/repo/`: the verification gate (above); `tests/repo/test_hooks.py` tests the local Claude Code hooks and skips when they are absent · `tests/analysis/`: analysis invariants, the coverage matrix, the legacy list (`legacy.json`), the scaffold; `-m full` heavy methodology tests · `tests/plans/`: pre-registered plans · `tests/metrics/test_provenance.py`: stale metrics · `tests/baselines/`: the metrics baseline, exemptions, accept log. Install `pip install -r requirements-dev.txt`, run `python tools/verify.py` (or `python -m pytest tests -v`), coverage `python -m pytest tests --cov=db --cov-branch --cov-report=term-missing` |
| `analysis/` | **all computation**: one script per question, number = technical report section (`01_dedup_leakage` · `01_engine_rule` · `01_unspecified_panels` · `01_gold_contract` (the gold data the API gets) · `02_missingness` · `03_association` · `03_segment_quality` · `03_brand_ablation` · `04_target` · `05_segmentation` · `06_hedonic` · `07_model_comparison` · `07_final_model` (serving files) · `07_lofo` · `07_text_flag` · `08_conformal_coverage` · `08_residuals` · `08_large_errors` · `09_drift` · `09_backtest` · `10_free_text`) + `shap/` (the SHAP report's sections: `02_oof_shap` · `03_what_sets_price` · `04_variants` · `06_one_listing`) + `lib/` (shared code: `common.py` · `cv.py` · `segment_rule.py` · `text_flags.py` · `labels.py`) + `run_all.py` + `frozen/text_ablation.json` (ablation frozen from the archived text analysis) + `cards/` (analysis cards: question, method, leakage risks, evidence; how to add an analysis: [docs/analysis-scripts.en.md](docs/analysis-scripts.en.md)) |
| `builders/` | builders that **do no analysis** and read only `metrics/*.json`: `build_site_data.py` (→ `site_data.json`, same schema) · `build_technical_report.py` · `build_business_report.py` · `build_shap_report.py`; shared code under `report_lib/`: `metrics_view.py` (the single reader + consistency gate) · `report_common.py` (shared numbers, figures, formatters) · `column_labels.py` |
| `tools/` | `observed_values.py` (writes the register of observed values; `--check` compares it with the data) · `verify.py` (definition of done: selects which tests run and summarises them; `--full` runs the chain first, `--json` prints a one-line summary) · `snapshot_metrics.py` (the metrics baseline: `--diff` shows the differences, `--accept "<reason>"` writes the new baseline) · `analysis_coverage.py` (the analysis coverage matrix: card · test · baseline · provenance) · `new_analysis.py` (scaffold of a new analysis: script + a test that fails on purpose + card) · `new_plan.py` (pre-registered plan: `new` / `lock`) |
| `internal_tool/` | **local-only internal tool (Streamlit, 127.0.0.1)**: the reports section by section with the scripts behind each section (code · card · metrics · tests · last run; read-only) · data explorer: conditions on any column joined with AND over the raw JSONL / silver / gold, a table with `ad_id` and `url`. Its own environment `/.venv-tool/` (outside git); setup and run: [internal_tool/README.md](internal_tool/README.md) |
| `plans/` · `backlog/` | locked pre-registered plans (`plans/<id>/analysis_plan.json`, `plans/README.md`) · ideas outside the plan (`backlog/ideas.md`) |
| `metrics/` | one JSON per script (`metrics/<script>.json`, `metrics/shap/<script>.json`); sections `meta` · `domain` · `methodology` (the site tree) · `error_drivers` · `oof_shap` · `shap*` · `report` (numbers only the reports use), each with `_meta` |
| `reports/` | `business.{tr,en}.md` · `technical.{tr,en}.md` · `shap.{tr,en}.md` — all generated, never hand-edited; which figure goes into which report is set by `BUSINESS_FIGS` / `TECHNICAL_FIGS` in `builders/report_lib/report_common.py` · `figures/` the report figures (`{tr,en}-NN-*.png`; the SHAP ones, `-sh-`, are drawn by `analysis/shap/`); the markdown files reference them as `figures/...` |
| `docs/` | the long documents, tr + en ("Documents" below) |
| **`data/`** | **all heavy data — `.gitignore`d**: `raw/` · `cars.duckdb` · `site_data.json` · `serving/` · `analysis/` (OOF and OOF SHAP artefacts) · `langextract/` |
| `archive/` | **archive, outside git** (index in `archive/README.md`): `obsolete/` · `analysis-history/` — the text analysis was taken out of the published work on 2026-09-19 ([docs/decisions.en.md](docs/decisions.en.md)); the SHAP report's versioned archive: `shap-v1-deneme-2026-09-20/` (every library variant, 46 figures) · `shap-v2-final-model-2026-09-20/` (pruned, drawn from the final model, 22 figures) · `shap-v3-oof-vakalar-2026-09-21/` (OOF, six case breakdowns + three waterfalls, with the generator script) · `robustness-2026-09-21/` (model robustness report: script + 2 md + 10 figures + measurement JSON) · `backups/` · `experiments/` · `published-report/` |

### `.gitignore` note

The pattern is **`/data/`** — the leading slash is deliberate. A bare `data/` would match
**at every depth**, and folders like `metrics` would silently go untracked too.
(That is why that folder is called `metrics`, not `data`.)

## What stayed outside the repository root, and why

**Only the live chain is tracked.** The folders below **stay on disk but are not
tracked by git** (`.gitignore`); none of them were deleted:

| root folder | what | git |
|---|---|---|
| `archive/` | the single archive root (five separate root folders until 2026-09-24): `obsolete/` (formerly `obselete/` — moved, not deleted, reasons in its own `README.md`; the method and decision log up to 2026-09-18 is in `archive/obsolete/docs/`, not to be confused with today's root `docs/`) · `analysis-history/` (formerly `_arsiv/`) · `backups/` (formerly `backup/`) · `experiments/` (experiments, large errors · domain analyses — if a finding sticks, its producer moves into the repository root) · `published-report/` (formerly `car-price-export/`: frozen copy of the published report — the reference for number audits). Index: `archive/README.md` | untracked |
| `.claude/` | local Claude Code setup: the verification hooks (`hooks/`, `settings.local.json`), the `methodology-reviewer` agent, skills | untracked |
