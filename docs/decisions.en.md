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
  hedonic → model → calibration → free text (the time section left on 2026-11-22).
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
  silhouette peaks. k=3 is fixed for interpretability, not chosen by silhouette. Since 2026-09-29 no clusters
  are built at all; §5 gives only the silhouette scan in one paragraph. The note is now built
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
- **Second list (2026-09-29, owner approved):**
  - §5 is one paragraph: the data has no clear clusters (silhouette always below 0.25); the fixed-k clusters, PCA
    and three figures left, and the numbering did not shift.
  - The Theil's U table in §3 became two sentences.
  - The best-case sample prediction table in §7 left; the served model's tree rule replaces it.
  - In both reports retraining looks at the error first (the fixed-test-set gap); drift is a diagnostic.
  - The §1 engine subsection stays as it is (the 2026-09-26 decision, repeated); LOFO's six features stay unmeasured.
- **Site comparison (2026-10-12 / 10-19, owner approved):**
  - The engine-consistency count in §6 gives both of §8's numbers: §8 flags 15 listings over all listings, 14 of them
    are in the hedonic sample (one has no valid engine size).
  - The §9 lead now gives the oldest-training gap on the fixed test set and ties retraining to the measured error
    only; drift is a diagnostic. It used to headline the 6.29% → 7.34% from different test sets.
  - The process wording in §8 (the pre-registration file path, the "tested the code" sentence) became reader wording.
  - Figure 15 left the decision note: it read as the evidence for "do not discard old snapshots" but draws the
    non-temporal plain OOF; that advice rests on the paired comparison. It stays in technical §9 with its "not
    temporal" note.

## Business / technical split

The **business note** = what to do, how much money. No method names (MAPE, R², conformal, OOF all
live in the technical report); everything in ₺ and plain percentages. The **technical report** =
protocol, controls, limits. Both are fed by **the same computation**: shared values are derived
once in `builders/report_lib/report_common.py::derive` and handed to both templates → a figure cannot differ between
the two. Some
charts (quartile error, coverage) appear in both (the lira quartile only in the business note since
2026-09-28): in the business note for the decision,
in the technical report as evidence — a deliberate repeat.

## Why the text analysis is archived

`archive/analysis-history/text_analysis/` — code, metrics, figures and reports are on disk but **not part of
the published work** (listed in `.gitignore`). Two reasons: its headline result was negative
(adding text features to the structured model produced no measurable gain), and the detectors behind its
price claims carried defects found by audit (`archive/experiments/regex_audit/`). Since 2026-09-27 that result is
not taken from the archive: the technical report's free-text section measures the text's contribution live on every run with
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
- the frozen text ablation (`analysis/frozen/text_ablation.json`) and the archived LLM extraction file → a live
  measurement in the free-text section;
- `dashboard_cache` / `options_cache` carried over from older DBs → dropped from silver and gold; the gold contract
  refuses any extra table;
- the hand-typed numbers in the §8 notes → computed on every run;
- ineffective items: `08_large_errors`'s old-D series group (`old_d_group`), the unused text reasons,
  `tools/metric_renames.py`'s proof commands against the archived P0 copy, two dead data files
  (`archive/obsolete/archive-inputs-2026-09-27/`).

## Drift removed, one forward test kept (2026-11-22, owner's decision)

The owner wanted time and drift out of the analysis ("it gets too tangled"). The decision was made after looking at
how the data is collected.

**Findings:**
- **Snapshots are stock dumps.** Each one takes every BMW/Audi listing live on the site that day: 10,901 · 11,254 ·
  11,478 · 11,526 listings.
- **The gaps are uneven.** 9, 53 and 98 days lie between snapshots. The share of listings unseen in earlier snapshots
  is 26%, 63% and 77%.
- **`listing_date` is not the posting date.** It changed for 6,057 of the 10,831 listings seen in more than one
  snapshot (56%); it is the day of the last update. The median "listing age" in a snapshot is therefore 10–13 days.
  The only reliable time axis is the four snapshots.
- **Listing age matters little.** The OOF residual is almost flat by listing age: medians between −0.46% and +0.12%.

**Why drift went.** Stock dumps do not see the flow. A listing that does not sell enters several snapshots; a quick
one opened and closed between two snapshots enters none. The difference between two snapshots' price distributions is
therefore a mix of three things: the market really moving, which listings stay in stock, and composition. PSI, KS, EMD
and the density curves cannot separate them, and the "residual by last-seen snapshot" is mixed with survival.

**Why one forward test stayed.** "Learn from the past, price the listings that arrive later" is a well-defined
question and the measurement closest to how the model is used. It is also the only evidence that the interval fails
on cars without a comparable.
- **Setup:** train on every listing seen before the last snapshot (latest price per listing), measure on the
  listings first seen in the last snapshot.
- **Measured:** MAPE with a model-resampled 95% interval; forward coverage of the served recipe (overall, per band,
  without a comparable).
- **Where:** `analysis/08_forward_test.py`; one paragraph in §8 of the report.

**Removed:**
- `analysis/09_drift.py` and the single, cumulative, paired, horizon, in-sample and per-snapshot arms of `09_backtest`;
- figures 13, 14, 15; the technical report's "Time" section (free text §10 → §9);
- `08_large_errors`' residual by last-seen snapshot and live/gone comparison; `08_residuals`' period sentence;
- the business note's retraining, PSI and regime items.

That the model does not see time stays in the report as one limitation sentence. The hedonic model had no period
effect already (2026-09-23).

**Withdrawn plan.** `plans/09-forward-coverage` (locked, 2026-09-28) pre-registered nine forward setups:
- H1 confirmed: the per-band margin fixes the cheapest band in every setup.
- H2 refuted: it does not hold 88% in every band.
- The hypotheses say "in every setup", so a single setup cannot carry them. The plan was withdrawn; its folder is kept
  locally in `archive/plans-withdrawn/`. The nine setups' numbers and evidence are in commit `fd0376a`
  (`metrics/09_backtest.json`).
- No new plan was opened: the single test's result had already been seen as one of the nine setups, so
  "pre-registering" it now would come after the fact.

## Served interval: scale model and hit probability (2026-11-23, owner approved)

**Decision.** The served 90% interval is now per car. A small second model (the scale model, s) predicts the size of
the error to expect for this kind of car; the interval is `expm1(log prediction ± k·s)`. A hit probability comes with
it: the chance that the listing price is within ±10% of the prediction. The price prediction did not change: the main
model file is byte-identical. Recipe and code are in `analysis/lib/conformal.py`; the plan is
`served_interval_plan.md` (repo root).

**Before.** One margin for everybody (q = 0.1412, i.e. −13.2% / +15.2%). Overall coverage was 90%, but not by group:
97% at age 0–7, 75% at age 19+, 71% for cars without a comparable, 66% in segment F.

**Arms tried** (OOF, cross-calibrated, 29,988 listings; forward test 8,889 new listings). IS is the interval score in
log space, smaller is better:

| arm | OOF coverage | OOF IS | OOF no comparable | forward coverage | forward IS | forward no comparable | outcome |
|---|---|---|---|---|---|---|---|
| one q | 89.99 | 0.4211 | 70.7 | 87.40 | 0.4438 | 68.35 | stays in the report (old serving) |
| q per band (plan 08) | 90.01 | 0.4059 | 66.8 | 87.99 | 0.4296 | 65.32 | stays in the report (pre-registered) |
| CQR | 90.00 | – | 73.7 | 88.68 | – | 70.03 | not in the code |
| band × comparables | 90.06 | 0.3929 | 90.8 | 87.68 | 0.4178 | 90.91 | not in the code |
| scale model | 90.09 | 0.3657 | 90.3 | 86.83 | 0.3977 | 88.55 | **served** |
| three-input scale (prediction, age, comparables) | – | 0.3780 | – | – | difference is noise | – | not in the code |

- **The middle way** (one k per comparables bucket) changed nothing in the forward test; not in the code.
- The numbers of CQR, band × comparables and the three-input arm come from read-only audit scripts; they are not in
  the live chain.

**Why the scale model.** Band × comparables fixed only the two failures we had seen (the cheap band, cars without a
comparable). The scale model fixes the other axes too: over the 42 groups on the axes it takes as inputs, 18 are more
than 5 points off target with one q and none with the scale model. It has the best interval score out of fold. The
reason for 30 inputs over three: the damage information a user enters should move the margin (coverage on heavily
damaged cars 80.8 against 89.7). In the forward test the difference between the two scale models is within noise.

**What was given up.** Band × comparables had a separate calibration in each of its 16 cells. With the scale model the
good coverage by group is empirical; that per-cell calibration was traded for a sharper interval and the other axes.

**Honesty notes.**
- **Chosen after the results.** The method, the settings of s (300 trees, 15 leaves, at least 100 listings per leaf,
  floor 0.02), the monotone constraint and the ±10% were fixed after results were seen. None was searched. No
  independent test is left.
- **90% holds approximately, for the same period.** The served centre and s are built on all data, the scores on fold
  models; this construction has no finite-sample guarantee. It does not hold over time: 86.8 on listings that arrive
  later.
- **The leak was measured.** The tests prove only that s never sees the scored fold; the errors come from the main
  model's cross-fitting. Rebuilding everything (main model, inner OOF errors, s, k) from the other four folds for each
  outer fold gave coverage 90.11, no comparable 90.03, IS 0.3669. The difference is negligible.
- **Stability.** Over 10 seeds and 7 settings: OOF coverage 90.00–90.25, k 2.075–2.105, forward coverage 86.69–87.06.
  But one car's margin moves by a median 2.5% (95th percentile 9.5%) from seed to seed. No seed averaging: serving is
  deterministic with one model, and serving five models is not worth this gain.
- **Monotonicity is a design rule.** "More comparables never widen the margin." It has no measurable effect on
  coverage (constrained and unconstrained are the same in the forward test).
- **Narrow intervals go stale first.** If the price level moves, the narrowest intervals lose coverage first; the report
  shows this with a what-if (`report.scale.level_shift`).
- **What s does not see.** Seller, city, colour and a flaw mentioned in the description are not in s; on those axes the
  scale model is neither better nor worse than one q.
- **The target is the listing price**, not the sale price. **The model name is free text:** comparables are looked up
  by the exact training string; a typo makes the count 0 and widens the interval (the cautious direction).

**Pre-registration.** The locked `plans/08-mondrian-coverage` covers the one-q and per-band arms; the scale model is
outside it and that plan stays as it is. No new pre-registration was opened: the results had been seen, and
"pre-registering" afterwards would mislead (the path taken when `plans/09-forward-coverage` was withdrawn).

**On the site.** The hit is shown as "N in 10", rounded down and at most 9 (with rounding 11% of cars would see "10 in
10", against 94–96.5% observed). The sentences carry the period: "by January–June 2026 listing prices". No percentage,
s, k or "high / medium / low" label is shown.

**Time audit (same day).** The owner asked whether the forward test's +2.2 log points is a price shift and whether the
project is fit for a drift analysis; three independent audits were run.
- **The +2.2 is not a price rise.** The forward signed error is +2.46 on average, +2.02 at the median. +1.16 of it
  comes from how the training set is built: 46% of the training prices are January-dated and belong to ads that left
  before March; ads that leave are ~2 log points cheaper on the same day than ads that stay. The price level rose by
  about +1 to +1.5 in 98 days (training on the March stock alone: +1.30 [1.0–1.6]). Nominal listing prices, not
  deflated.
- **PSI, KS and EMD do not come back.** Between consecutive snapshots the price PSI is below a noise floor that keeps
  shared ads fixed; the 0.10 threshold is exceeded only on `model` and location, where the value equals the noise. The
  "drift removed" decision was confirmed.
- **No date in the model, no constant offset (owner's decision).** A snapshot input learns the cheap leaving ads as a
  date effect (+1.28 for 9 days; the stock moved 0.42). Shifting the target and retraining gives the same result as a
  constant offset. The offset (+1.73) removes the bias on June listings and lifts their coverage from 88.7 to 89.3, but
  it cannot be validated because no test at a later date will be run; it was not applied. The model sits about 1.7%
  below the last snapshot's listings on average; this is written as a limitation.

**Outside the repo.** The API and the site (`sadik-portfolio`) are not changed from this repo; the specification is in
steps 7–9 of the plan and the parity examples in `data/serving/serve/parity_examples.json`.

## The hedonic section was rebuilt (2026-11-23, owner's request)

**Why.** The two-column coefficient table ("segment control", "model control") asked the question backwards. What the
owner wanted to see: how much of the price is explained without knowing what car it is, how much by the identity alone,
and how each effect changes once the identity is held fixed. The hedonic regression was not removed: no other section
gives the effects of mileage, heavy damage and changed parts this cleanly (in SHAP the split between age and mileage
depends on the price model).

**Added** (`analysis/06_hedonic.py`, `report.hedonic`):
- **The ladder:** in-sample and 5-fold out-of-fold R² of six regressions. Out of fold: segment, brand, fuel and
  transmission only 0.38; condition only (age, mileage, damage, paint, changed parts) 0.73; the car model name only
  0.82; condition + engine 0.87; segment control 0.93; car-model control 0.96.
- **A third column without identity controls.** Mileage is −10% without the identity and −15% with it; +100 hp is +50%
  without the identity and +20% once the segment enters. So uncontrolled mileage misleads, and most of the "effect" of
  power is the car's class.
- **Age as yearly steps** (age dummies and a mileage spline within the car model): −4.7% to −6.8% per year, depending on
  age. The quadratic's single number at the median car (−5.8%) depended on the functional form; in the audit flexible
  forms gave −4.1…−4.4%. The data is one period: what is measured is the difference of one model year, not the ageing
  of one car.
- **Parts by count:** a painted part is not linear (1: −2.3%; 2: −3.6%; 5 and more: −7.0%; the linear coefficient said
  −1.0% per part). Changed parts are close to linear (1: −3.7%; 2: −6.4%; 3: −9.3%).

**Removed from the business note.** The "+100 hp" row: within one car model power hardly varies (it varies in 354 of
735 car models, within-model spread 9 hp) and the number is sensitive to a single car model. It stays in the technical
report with its caveats. The heading's "how much each item moves the price" was causal and was reworded.

**No pre-registration.** Everything was added after the results were seen; pre-registering afterwards would mislead.
The numbers of the two published columns did not change.

**Wording rule (same day).** In the reports "model" meant two things. The data column is the "car model", LightGBM is
the "price model", wherever the two could be confused.

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
  market level is not measured since 2026-11-22 (below, "Drift removed, one forward test kept"). Proposed again on
  2026-09-28 (+5.3%) and not taken: the number came from the site's old July data, and a "period" in this data
  is the snapshot a listing was last seen in, so it is mixed with survival.
- **The model does not see time** (2026-11-22). The snapshots' prices are learned together; drift, the period
  effect and the retraining advice left the analysis. The reports say so in one limitation sentence; the error on
  the last snapshot's new listings is in technical report §8 (`analysis/08_forward_test.py`). The earlier rule
  ("retraining is not tied to a fixed PSI threshold", 2026-09-27/28) went with it.
- Scope is BMW + Audi; how far it generalises to other brands was not measured.
