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
