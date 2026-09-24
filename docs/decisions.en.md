# Decisions and limits

> 🇹🇷 Türkçe (varsayılan): [decisions.md](decisions.md) · ← [README](../README.en.md)

**Deliberate departure from the published report — flat LOFO.** The raw `methodology.lofo` carries
both single-feature and group removals; plotting both on one axis double-counts (`DAMAGE_COLS`
competes with its own 13 members). The report reduces it to 5 **non-overlapping** groups.
Separately, 6 categorical features (`brand`, `kb_body_type`, `kb_drivetrain`, `segment`,
`kb_transmission`, `kb_fuel`) are **never measured** by LOFO — the producer removes the numeric
features, `model` and `series`, the groups and the roof/hood/trunk states one by one, but not these six. The report states that
limit.

**The technical reports' section order lives in one place.** Both generators
(`builders/build_technical_report.py`, and the archived text report generator) carry a `SECTIONS` list;
each section is its own `sec_*` function. Section numbers come from `enumerate` and the `§` cross-references
in the prose from `secno("key")` — to reorder, move the lines in `SECTIONS`; numbers and references stay
correct by themselves.

- **Price report:** data → missingness → redundancy/brand → target and preprocessing → market structure →
  hedonic → model → calibration → time → free text.
The archived text report's generator follows the same pattern; that report is not part of the
published work.

**Deliberate exceptions — hand-written content.**

1. ~~`HP_SEARCH_TL = 5000`~~ — removed on 2026-09-20 together with the noise-floor paragraph it
   belonged to; it appears nowhere now. The only hand-written content left is the example notes below
   (`feature_drop` has been computed since 2026-09-23).
2. `HANDWRITTEN_EXAMPLE_NOTES` (2026-09-17, user decision) — the explanations under the examples in the
   technical report's "Where the large errors come from", written by reading the ads. The table above them
   and its **automatic reasons** column come from `analysis/08_large_errors.py`. Each note is tied to an example
   by `(model, year, price)`; if the data changes and the example is not found, the generator **stops** —
   a note cannot silently go stale.

**Derived values.** Every number that needs the data is computed in an analysis script (the ones only the
reports use sit in the `report` section of the metrics JSON: error bands, conformal coverage, the Holm
correction …). The builders only do formatting-level arithmetic on published numbers (e.g. the model's
percentage improvement over the baseline, (base−model)/base). The noise-floor pair `1.42×` and `₺33K` left the reports on 2026-09-20 and is gone here too.

**Hand-written notes that contradicted the data were fixed (2026-09-16).** Three hand-written
sentences in the producers contradicted the measurements; the reports now read numbers, not these
notes:

- `kmeans_selection.not` said "silhouette is highest at k=3"; in the measurement k=3 is not where
  silhouette peaks. k=3 is fixed for interpretability, not chosen by silhouette. The note is now built
  from the data.
- `numeric_correlation.not` said "VIF all <3". Since 2026-09-23 VIF is computed from the fitted hedonic
  model's own design matrix, and the technical report prints the centred and uncentred versions together.
- The flag label in `controlled_effects` said "(hidden damage)", while the chain's own framing is
  "not hidden damage". Label now: *Contradictory 'clean' claim (seller's own form shows damage)*.

**The winning variant comes from the data.** The producer's rule looks at MAPE only; on the CPU run
CatBoost (TF-IDF+SVD) edges LightGBM on MAPE, whereas the published GPU run had the order reversed. The
gap is noise-level and changes direction from metric to metric; current values are in the technical
report's model section. The report places the ★ from the
data; wherever it says "the model" it means LightGBM — the one that is deterministic on CPU.

**The served model uses the measured setting (2026-09-23).** The report's metrics come from fold models
with early stopping; the final LightGBM used to be trained with a fixed 900 trees. Its tree count is now
the median of the tree counts early stopping picked in CV (`meta.repro.final_lgb_agac`, per-fold values in
`cv_agac`). `data/serving/serve/README.md` describes how to build the input exactly as in training;
`encoders.pkl` carries the segment rule (including `PERF_RE`) and `CONFORMAL_Q` for the 90% interval. In
LOFO the base and the drop models use the same tree limit; where each stopped is in
`methodology.lofo_agac`.

## Business / technical split

The **business note** = what to do, how much money. No method names (MAPE, R², conformal, OOF all
live in the technical report); everything in ₺ and plain percentages. The **technical report** =
protocol, controls, limits. Both are fed by **the same computation**: shared values are derived
once in `builders/report_lib/report_common.py::derive` and handed to both templates → a figure cannot differ between
the two. Some
charts (quartile error, lira quartile, coverage, backtest) appear in both: in the business note for the decision,
in the technical report as evidence — a deliberate repeat.

## Why the text analysis is archived

`archive/analysis-history/text_analysis/` — code, metrics, figures and reports are on disk but **not part of
the published work** (listed in `.gitignore`). Two reasons: its headline result was negative
(adding text features to the structured model produced no measurable gain, ΔR² ≈ 0.0015 — stated
in §10 of the car price report), and the detectors behind its price claims carried defects found by
audit (`archive/experiments/regex_audit/`). Nothing imports from the archive any more (2026-09-23): the four
pattern detectors behind the example reasons and the text flag were moved unchanged to
`analysis/lib/text_flags.py` (identical results to the archive on all 29,988 listings), and the ablation measure
and its setup were frozen in `analysis/frozen/text_ablation.json`.

## Privacy

1. `ad_id` is written into no report and no site data; it stays in the database and in
   `data/duplicate_ad_ids.csv` next to it (both local; `data/` is outside git). Ad text never reaches any
   markdown either.
2. **Listing-level rows are published deliberately** (decision 2026-09-16): best and worst
   predictions — model · price · age · km. A rare model plus an exact price can be found by
   searching; that is an accepted risk.

## Honest framing (must be preserved)

- Structural data solves price; **text adds ~0 to price accuracy** (measured, not asserted).
- **Controlled ≠ raw.** Every price claim is given with vehicle features held constant.
- **Derived ≠ fed.** `segment` is derived from the series and model name (the raw `gb_segment` is
  corrupt); the rule lives in `analysis/lib/segment_rule.py` (single source; `07_final_model` copies it into
  `encoders.pkl`), and the run stops if a series or model cannot be resolved. The report says so.
- **Scope comes from the collection filters.** A price cap (price is right-truncated), an earliest model
  year, the `/otomobil/` category only (the SUV category was not collected; only a few SUVs are in the
  data) and four fuel types (no electric cars). Technical report
  §1 reads these filters from the scraper's config file (`scraper/collection_config.json`) and counts
  their footprint in the data.
- **Engine size = the bucket's upper bound, power = the mean of the lower and upper bounds** (the user's
  rule). On some listings the site gives both as a bucket; technical report §1 prints the table and figure
  that compare each candidate with the same model's exact-value listings. If the chosen candidate stops
  having the smallest gap, `analysis/01_engine_rule.py` stops.
- **"Unspecified" panels count as original** (a deliberate decision). Technical report §1 prints its
  size and the price evidence; the cost is that damage effects are pulled slightly toward zero. The
  translation happens in the analysis (`analysis/lib/common.py`); the DB keeps the information semi-raw
  (`NULL` from the next build on).
- **The hedonic model has no period effect** (user decision, 2026-09-23): periods are pooled; the shift in
  market level is read in technical report §9 from live listings of the same model and year.
- Scope is BMW + Audi; how far it generalises to other brands was not measured.
