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
2. `EXAMPLE_NOTES` (2026-09-17, user decision; formerly `HANDWRITTEN_EXAMPLE_NOTES`) — the explanations under
   the examples in the technical report's "Where the large errors come from", written by reading the ads. The table
   above them comes from `analysis/08_large_errors.py`. **Since 2026-09-27 the notes hold no hand-typed number:**
   the comparison each note rests on (series count, the name's other listings, the comparison group's n and median)
   is computed by 08 on every run (`examples[].compare`) and enters the note from there; the note's claim (e.g.
   "the only R8 in the data", "in line with same-year 730ds") is gated on the data. Each note is tied to an example
   by `(model, year, price)`; if the data changes and the example is not found or the claim no longer holds, the
   generator **stops** —
   a note cannot silently go stale.

**Derived values.** Every number that needs the data is computed in an analysis script (the ones only the
reports use sit in the `report` section of the metrics JSON: error bands, conformal coverage, the Mondrian
bands …). The builders only do formatting-level arithmetic on published numbers (e.g. the model's
percentage improvement over the baseline, (base−model)/base). The noise-floor pair `1.42×` and `₺33K` left the reports on 2026-09-20 and is gone here too.

**Hand-written notes that contradicted the data were fixed (2026-09-16).** Three hand-written
sentences in the producers contradicted the measurements; the reports now read numbers, not these
notes:

- `kmeans_selection.not` said "silhouette is highest at k=3"; in the measurement k=3 is not where
  silhouette peaks. k=3 is fixed for interpretability, not chosen by silhouette. The note is now built
  from the data.
- `numeric_correlation.not` said "VIF all <3". Since 2026-09-23 VIF is computed from the fitted hedonic
  model's own design matrix; since 2026-09-28 the technical report gives its highest value (centred and
  uncentred) in one sentence.
- The flag label in `controlled_effects` said "(hidden damage)", while the chain's own framing is
  "not hidden damage". Label now: *Contradictory 'clean' claim (seller's own form shows damage)*.

**The winning variant comes from the data.** The producer's rule looks at MAPE only; on the CPU run
CatBoost (TF-IDF+SVD) edges LightGBM on MAPE, whereas the published GPU run had the order reversed. The
gap is noise-level and changes direction from metric to metric; current values are in the technical
report's model section. The report places the ★ from the
data; wherever it says "the model" it means LightGBM — the one that is deterministic on CPU.

**The served model uses the measured setting (2026-09-23).** The report's metrics come from fold models
with early stopping; the final LightGBM used to be trained with a fixed 900 trees. Its tree count is now
the median of the tree counts early stopping picked in CV (`meta.repro.final_lgb_trees`, per-fold values in
`cv_trees`). `data/serving/serve/README.md` describes how to build the input exactly as in training;
`encoders.pkl` carries the segment rule (including `PERF_RE`) and `CONFORMAL_Q` for the 90% interval. In
LOFO the base and the drop models use the same tree limit; where each stopped is in
`methodology.lofo_trees`.

**The report describes the preprocessing, not the data layers (2026-09-26, owner's decision).** The technical
report does not talk about the semi-raw database, gold or the data the API gets; §1 lists the steps applied to
the collected data before the model, in order and plainly (plate, deduplication, "unspecified" = none, engine
range, damage flags, missing values, outliers, target). Gold is described only on the `db/` side and in
`docs/database.en.md`; `analysis/01_gold_contract.py`, which fed §1's former gold subsection, is archived.

## Simplification (2026-09-28, owner approved)

An externally prepared simplification list was assessed item by item. The list looked at the site's old copy (ΔR²
0.0015 and the hedonic period effect +5.3% came from there); the single source is the generator, and the site copies
the report's markdown.

- **Left the report:**
  - Cramér's V, the permutation floors and the Theil's U matrix (the asymmetry table stays);
  - the Pearson map and the |r|>0.5 table (the Spearman pairs are one sentence);
  - the VIF table, the cc–hp table by fuel, the assumption-test paragraph (one sentence each);
  - the best-5 and the two worst-6 tables (their findings stay);
  - the lira panel in the technical report (it stays in the decision note);
  - the drift p-values, the disjoint table and Holm (the snapshots share listings; a test assumes independence);
  - the LOFO coverage table (one sentence).
- **Method changed:**
  - Hedonic confidence intervals come from standard errors clustered by model. The row bootstrap treated listings
    as independent; 22 series are too few and too uneven to cluster on.
  - The hedonic model has two columns: segment control and model control. The decision note prints the model
    column (owner's decision).
  - Every grouping by price level is cut on the predicted price; grouping by the actual price produces regression
    to the mean.
  - Coverage is cross-calibrated. The per-band margin (Mondrian) was measured under a pre-registration
    (`plans/08-mondrian-coverage`, H1 confirmed); the single served q is unchanged.
  - The backtest runs the headline setup (the last insample block equals the headline OOF bit for bit), with 95%
    model-cluster bootstrap intervals and a paired single/cumulative comparison. The "pure time effect" and "more
    data, less error" claims were cut to what the intervals support.
- **Not taken:**
  - a hedonic period dummy (below);
  - EMD on the log scale and a log mean of the cell change;
  - LOFO for the six categorical features (decision of 2026-09-25);
  - a ±1% retraining threshold that does not come from the data.
- `tools/snapshot_metrics.py --drop-exemption` removes the exemption of a key removed on purpose (the bootstrap run
  time went this way).
- **Independent audit (same day) and its fixes:**
  - The model column was solved by a pseudo-inverse on a singular design; its clustered SEs blew up on small
    perturbations. It now uses the within transform (fixed effects): same estimates, full-rank design, tested.
  - The column-difference sentence now rests on a per-term cluster-robust difference test.
  - Sensitivities were added: clustering by series, and +100 hp without the listings with an inconsistent engine value.
  - "In lira the largest errors are under-predictions" is not a finding: a symmetric log error also gives about 68/100
    under-predictions; the report says so.
  - The horizon is also measured on one fixed test set (the same listings; the error grows as the training snapshot
    ages).
  - **Deviation from a pre-registration:** the criterion of `plans/08-mondrian-coverage` held almost by construction
    under random folds; it tested the code, not the method. The real question was asked under a new pre-registration
    (`plans/09-forward-coverage`): applied forward, the per-band margin fixes the cheapest band in every setup (H1
    confirmed), but does not hold 88% in every band (H2 refuted) and lowers coverage on listings without a comparable.
    The decision note's advice was written from that result.

## Business / technical split

The **business note** = what to do, how much money. No method names (MAPE, R², conformal, OOF all
live in the technical report); everything in ₺ and plain percentages. The **technical report** =
protocol, controls, limits. Both are fed by **the same computation**: shared values are derived
once in `builders/report_lib/report_common.py::derive` and handed to both templates → a figure cannot differ between
the two. Some
charts (quartile error, coverage, backtest) appear in both (the lira quartile only in the business note since
2026-09-28): in the business note for the decision,
in the technical report as evidence — a deliberate repeat.

## Why the text analysis is archived

`archive/analysis-history/text_analysis/` — code, metrics, figures and reports are on disk but **not part of
the published work** (listed in `.gitignore`). Two reasons: its headline result was negative
(adding text features to the structured model produced no measurable gain), and the detectors behind its
price claims carried defects found by audit (`archive/experiments/regex_audit/`). Since 2026-09-27 that result is
not taken from the archive: §10 of the technical report measures the text's contribution live on every run with
`analysis/10_free_text.py` (pre-registered as `plans/10-text-contribution`; the model's OOF against the same folds
plus description TF-IDF/SVD). Nothing imports from the archive, and since 2026-09-27 no data or frozen number is
read from it either (the "Nothing from the archive" decision below). The text flag's two pattern detectors
(conversion, modification) live in `analysis/lib/text_flags.py`; the patterns came from the archived chain and the
modification word list was distilled from the vocabulary of an archived LLM extraction trial. They stay because
they are applied to the current text on every run and their output is used. The hp and M/RS model detectors,
which fed nothing, were removed; `analysis/frozen/` and the archived LLM extraction file are gone.

## Nothing from the archive reaches the live chain (2026-09-27, owner's decision)

"Take nothing from the old archive … don't use what was copied from it and is not fully live or has no effect."
Every input of the live chain is produced by the current scripts in the current run; data, tables, frozen numbers
or hand-typed numbers copied from `archive/`, an older run or an older DB never enter it. Code and rules that came
from the archive (the segment map, model settings, the text flag's word list) stay only if they are applied to the
current data on every run and their output is used, and their origin is written down; nothing ineffective is
kept. Guard: `tests/repo/test_no_archive_inputs.py`. Removed under this decision:
- the frozen text ablation (`analysis/frozen/text_ablation.json`) and the archived LLM extraction file → §10 is a
  live measurement;
- `dashboard_cache` / `options_cache` carried over from older DBs → dropped from silver and gold; the gold contract
  refuses any extra table;
- the hand-typed numbers in the §8 notes → computed on every run;
- ineffective items: `08_large_errors`'s old-D series group (`old_d_group`), the unused text reasons,
  `tools/metric_renames.py`'s proof commands against the archived P0 copy, two dead data files
  (`archive/obsolete/archive-inputs-2026-09-27/`).

## Privacy

1. `ad_id` is written into no report and no site data; it stays in the database and in
   `data/duplicate_ad_ids.csv` next to it (both local; `data/` is outside git). Ad text never reaches any
   markdown either.
2. **Listing-level rows are published deliberately** (decision 2026-09-16): the §7 sample
   predictions and the §8 examples — model · price · age · km (the best-5 and the two worst-6 tables left
   on 2026-09-28). A rare model plus an exact price can be found by
   searching; that is an accepted risk.

## Honest framing (must be preserved)

- Structural data solves price; **text adds ~0 to price accuracy** (measured, not asserted).
- **Controlled ≠ raw.** Every price claim is given with vehicle features held constant.
- **Derived ≠ fed.** `segment` is derived from the series and model name (the raw `gb_segment` is
  corrupt); the rule lives in `analysis/lib/segment_rule.py` (single source; `07_final_model` copies it into
  `encoders.pkl`), and the run stops if a series or model cannot be resolved. The report says so.
- **Scope comes from the collection filters.** A price cap (price is right-truncated), an earliest model
  year, the "otomobil" (car) category only (the SUV category was not collected; only a few SUVs are in
  the data) and four fuel types (no electric cars). Technical report
  §1 reads these filters from the scraper's config file (`scraper/collection_config.json`) and counts
  their footprint in the data.
- **Scope is Turkish-plated cars only** (2026-09-26). Blue-plate listings (a different tax regime) and
  listings with an empty plate field (regime unknown) do not enter the model. Blue plates stay in the
  database but do not go to gold (`db/gold_rules.json`). The counts are in technical report §1.
- **A listing with no body style is not filled in** (2026-09-26). The model name does not fix the body: the
  same name is sold in several body styles, and the site labels even the same name inconsistently. The
  model sees these listings as a category of their own; the median-price-by-body-style chart leaves them
  out and counts them in its title.
- **Engine size = the bucket's upper bound, power = the mean of the lower and upper bounds** (the user's
  rule). On some listings the site gives both as a bucket; technical report §1 prints the table and figure
  that compare each candidate with the same model's exact-value listings. If the chosen candidate stops
  having the smallest gap, `analysis/01_engine_rule.py` stops.
- **"Unspecified" panels count as original** (a deliberate decision); the heavy-damage record follows the
  same rule. Technical report §1 prints its size. The translation happens in the analysis
  (`analysis/lib/common.py`); the database keeps the information semi-raw (`NULL`).
- **The hedonic model has no period effect** (user decision, 2026-09-23): periods are pooled; the shift in
  market level is read in technical report §9 from live listings of the same model and year. Proposed again on
  2026-09-28 (+5.3%) and not taken: the number came from the site's old July data, and a "period" in this data
  is the snapshot a listing was last seen in, so it is mixed with survival.
- **Retraining is not tied to a fixed PSI threshold** (2026-09-27). Drift is watched and the model is
  retrained on new snapshots. The reason is in the data: with PSI far below the threshold, the error of a
  model trained on one snapshot still grows as the test horizon lengthens (technical report §9). The price
  arrives with every snapshot, so the model's error on new listings is watched directly; there is no fixed
  threshold (2026-09-28).
- Scope is BMW + Audi; how far it generalises to other brands was not measured.
