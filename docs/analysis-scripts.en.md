# How to read the analysis scripts

> 🇹🇷 Türkçe (varsayılan): [analysis-scripts.md](analysis-scripts.md) · ← [README](../README.en.md)

Each script answers one question and has **three layers**: (1) pure analysis functions — no file reading or
writing, they only return values; (2) `to_metrics` — naming and rounding only; (3) run cells —
`[4] Load · [5] Compute · [6] Save`. `save_metrics` is called **once** per script, in the last cell; if a
function fails no JSON is written and the old one stays as it was. File names are English; every file and
every function has a bilingual description (`EN:` / `TR:`).

**Cell by cell in VS Code:** the scripts use `# %%` cells; with the Python + Jupyter extensions
**Shift+Enter** runs the cell in the Interactive Window. You can run up to `[5]` and look at the result;
without `[6]` the JSON does not change.

**Shared code:** `analysis/lib/common.py` exposes two things — `load_clean()` (TR-plated, price > 0, latest row
per `ad_id`, `ORDER BY ad_id`, derived features) and `save_metrics()` (JSON + `_meta`: script name, run time,
last snapshot date in the data, DB fingerprint, `run_id`). `analysis/lib/cv.py` is the single copy of the 5-fold
OOF (same folds, TF-IDF+SVD inside the fold, same LightGBM settings) and the reader/writer of the OOF
artefact; `analysis/lib/segment_rule.py` is the segment rule's single source; `analysis/lib/text_flags.py` holds
four pattern flags read from the ad text (no LLM).

**Consistency gate:** the builders check that every metrics file carries the same DB fingerprint and
everything model-derived the same `run_id`; no report is written from mixed or stale output, and the message
says which script to rerun. A script that depends on 07's artefact stops if the artefact is missing or comes
from another DB.

## How to add a new analysis

The gate does not count a new analysis as done without a card and a test (`tests/analysis/test_coverage.py`).
The order:

1. **Plan (for a new question):** `python tools/new_plan.py new <id> "soru" "question"`. Fill it in, show it to the
   owner, and only after their explicit approval `python tools/new_plan.py lock <id> --by "<approver>"`. A locked
   plan only changes hypothesis status and evidence (`plans/README.md`).
2. **Scaffold:** `python tools/new_analysis.py NN_name "soru" "question"`. Writes the three-layer script, a stub
   test that fails on purpose, and a card with status `todo`.
3. **`ORDER`:** add the script to the list in `analysis/run_all.py`, in numeric order.
4. **Analysis:** write and run the script (`python analysis/NN_name.py`). `save_metrics` records the provenance
   (sha256 of the script and of the `analysis/lib` files it loaded) in the metrics.
5. **Test:** replace the stub with at least one real invariant. Examples: a range, a total, a count that must
   match another script, a report claim that must follow from the numbers next to it. It reads the metrics JSON
   only.
6. **Card:** `analysis/cards/NN_name.json`:
   - question, data, method, split;
   - the four leakage types (a note is required for `present` / `unknown`);
   - evidence: collected test ids + report keys in the metrics;
   - `status: complete`.
7. **Baseline:** `python tools/snapshot_metrics.py --diff`, then, with the owner's approval,
   `--accept "<reason>"`.
8. **Done:** `python tools/verify.py` → PASS. `python tools/analysis_coverage.py` shows what is left.

- **Stale metrics:** when a script or an `analysis/lib` file it imports changes, its metrics count as stale and
  the gate names the script to rerun (`tests/metrics/test_provenance.py`). Touching `lib/common.py` makes every
  script stale (full run ~16 min).
- **Legacy scripts:** the 18 scripts written before cards and tests are in `tests/analysis/legacy.json` with the
  sha256 of their source. A legacy script that changes needs a card and a test; the list only shrinks.
- **Heavy methodology tests** (not in the fast gate): `python -m pytest -m full tests/analysis`. The checks: no
  `ad_id` overlap between folds, R² ≈ 0 on a shuffled target, bit-identical reruns.
