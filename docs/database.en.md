# Database (`cars.duckdb`) notes

> 🇹🇷 Türkçe (varsayılan): [database.md](database.md) · ← [README](../README.en.md)

This file is the analysis' input. It does not go to the API directly: publishing uploads the gold file
derived from it (`data/cars_gold.duckdb`, below). Until 2026-09-24 this file itself was published.

- **Semi-raw (2026-09-24, the owner's principle: "everything in the DB should be semi-raw").** A value the
  page does not give is not guessed, it stays `NULL`; the analysis decides how to read "Belirtilmemiş"
  (unspecified). The real file was rebuilt with this rule on 2026-09-24 (the previous, old-contract file is
  in `archive/backups/cars-duckdb-eski-sozlesme-2026-09-24/`):
  - `is_heavy_damaged` / `kb_is_heavy_damaged` from "KısaBilgi - Ağır Hasarlı": Evet / Hayır / `NULL`
    (unspecified or missing; 29,641 rows `False` → `NULL`). The scraper's `Agir_Hasar` flag was dropped
    because it is `False` also when the page says nothing;
  - the three flags of a panel whose status is "Belirtilmemiş" are `NULL` (97,996 panels, 293,988 cells;
    1/0 for a known status);
  - `gb_is_first_owner` is `NULL` when the field is missing (120 rows); the damage counts are `NULL` when
    empty (0 rows today);
  - description: only the page's section heading "Açıklama" (and the one space after it) is removed from
    `description_text`; the seller's "-" / ":" stays (differs from the old `description_clean` in 778 rows).
    The `description_clean` column is gone;
  - new raw column `kb_paint_change_summary`: the "Boya-değişen" line of the short-info box (like "2
    değişen, 3 boyalı"), as written. A coarse summary of the damage diagram.

  Done in the analysis with the rebuild (2026-09-24):
  - `analysis/lib/text_flags.descriptions` reads `description_text`; none of the text flags changed;
  - `02_missingness`: `description_clean` left `IDENTITY`, `kb_paint_change_summary` is in the "derived
    duplicate" (F) class. The 45 columns whose `NULL` means "Belirtilmemiş" (the columns the gold rules fill,
    read from `db/gold_rules.json`) are kept out of the missing list and the blocks and reported on their own:
    the heavy-damage record is unspecified on 68.2% of listings, the panel flags on 12.2%–19.1% (technical
    report §1–§2);
  - `load_clean`: `is_heavy_damaged` now holds `NULL`, so pandas reads it as nullable `boolean` and the old
    `fillna(0)` raised; an unknown now counts as `False` (not heavily damaged), the same result as before.
  No value that reaches the model and none of the model's numbers changed (checked against the metrics
  baseline: `tests/baselines/accept_log.jsonl`).
- **Safe build.** The new DB is built in `<out>.tmp` and moved over the old one when complete; a failure
  leaves the old DB untouched. It stops before touching anything when `<out>.wal` exists, when the DB is open
  in another program, or when the raw data is incomplete (a brand folder, a snapshot without
  `details.jsonl`, an unreadable line). `duplicate_ad_ids.csv` is written next to the DB.
- **No guessing: the register of observed values (2026-09-25, the owner's rule: "don't assume cases that never
  happened").** `db/observed_values.json` records what the data actually holds:
  - each raw field's values (up to 60 distinct) or formats (`"# TL"`, `"# - # cm3"`);
  - the damage labels;
  - the silver columns with few values;
  - series → models.

  It is written by `python tools/observed_values.py` (`--check` compares it with the data). The build ties to it in
  three places:
  - `build_duckdb.py` compares the raw data with the register before reading any rows. A new field, an unseen
    value or format, or an unknown damage label stops the build **with nothing written**; the message names the
    field, the value, how many records and the first file:line.
  - The parsers (`db/lib/process_for_db.py`) raise `UnknownValue` on any form the data never showed instead of
    guessing. A single production year used to become "start = end", an unknown first-owner text became `False`,
    and an unknown damage label was skipped.
  - Tests check both ways: every value the code names was seen in the data, and every value in the data is handled
    on purpose (`tests/db/test_observed_values.py`, `tests/analysis/test_observed_values_analysis.py`).

  At the same time, three assumptions the data never showed were removed: the plate `"Yabancı plakalı"`, 14 series
  in the segment rule (`8 Serisi`, `X1`–`X7`, `Z4`, `Q2`–`Q8`), and `02_missingness`'s missing tokens such as
  `"nan"`/`"None"`. The real raw data rebuilt in a temp folder equals `data/cars.duckdb` row for row (a data test).

  When a new value shows up, the code is adapted to it first, then the register is updated.
- **`engine_cc_val` means two things in two places.** In the DB it is the bucket's midpoint (the known
  bound for an open-ended bucket); the model's feature of the same name is `engine_cc_up` (the upper bound,
  reasoning in technical report §1). The DB column was not renamed, so an API reading it does not break.
  The API's drift screen reads this column, so it shows the midpoint while the model uses the upper bound (7,827
  bucketed listings differ, median 99.5 cc; measured 2026-09-25). Drift compares two snapshots on the same
  definition, so the result holds; left as is by the owner's decision.
- **`dashboard_cache` / `options_cache`** belong to no generator in this repo (the archived
  `build_aggregates.py`); their content is stale (2026-07-14, no TR plate filter). On a rebuild
  `build_duckdb.py` carries them over from the old DB unchanged: they are neither dropped nor refreshed.
- **`gb_body_type`** holds body type merged with the seat count on part of the rows; the model uses
  `kb_body_type`.
- **`duplicate_ad_ids`** lists `ad_id`s with more than one row, i.e. listings seen again in later
  snapshots (there are no repeats within a single snapshot).
- **Parser fixes of 2026-09-23:** in the trade field "Takasa Uygun Değil" (not open to trade) is now
  `False` (it used to be `True` because it contains "uygun"), and a missing field is `NULL`; first owner was
  always `False` because of the Turkish capital İ; "up to … HP" / "… and above" power buckets are now
  open-ended buckets; for an open-ended engine size bucket the DB midpoint used to be `NULL`;
  `kb_fuel_cons_avg` also fills from the "Yakıt Tüketimi" tab; a description that only says "Açıklama" is
  `NULL` in `description_clean` (`description_text` keeps the raw text). No value that reaches the model changed (old and new DB compared column by column). The DB was
  rebuilt locally; **uploading it to S3 is a separate step** (first `python db/publish_data_to_s3.py --dry-run`,
  then without `--dry-run`).

## The gold step (API) — written 2026-09-24

The analysis reads the semi-raw DB; the live API (outside this repo) expects the old contract: unknown heavy
damage / first owner `false`, "Belirtilmemiş" panel `0`. One file cannot serve both, so the API gets a separate
gold file.

```
data/raw ──build_duckdb──► data/cars.duckdb (semi-raw, analysis) ──build_gold_db──► data/cars_gold.duckdb (API)
                                                                                   └─publish_data_to_s3─► S3 "data/cars.duckdb"
```

- **`db/build_gold_db.py`** only reads the semi-raw DB and writes `data/cars_gold.duckdb` safely (`.tmp` + move,
  `.wal` check). It first checks that the input really is the semi-raw DB (columns `id` + `DB_COLUMNS`); an
  old-contract or a gold file is refused. The rules, with their reasons, are in **`db/gold_rules.json`** (the
  technical report's gold section will read the same file):
  - `is_heavy_damaged`, `kb_is_heavy_damaged`, `gb_is_first_owner`: `NULL` → `false`;
  - the 39 panel flags and the 3 damage counters: `NULL` → `0` (a "Belirtilmemiş" panel counts as original, as in
    the analysis);
  - `kb_paint_change_summary` is left out;
  - the description as in the semi-raw DB: only the heading-free `description_text`, **no `description_clean`**
    (owner's decision 2026-09-24: keep the description as `description_text`; if it touches the API, the owner
    fixes the API);
  - result: 116 columns + `id`; rows, order, ids, types and every other cell unchanged; the four other tables as
    they are.
- **`publish_data_to_s3.py`**: default input `data/cars_gold.duckdb`, S3 object name unchanged
  (`data/cars.duckdb`). Before uploading it checks the gold contract (columns = `id` + the gold columns, no `NULL`
  in a rule column); a semi-raw or old-contract file cannot reach the API.
- **Run order:** `build_duckdb.py` → `build_gold_db.py` → (owner) `publish_data_to_s3.py`.
- **Tests** (`tests/db/test_build_gold_db.py` + the publishing tests): the rule file fits the DB columns; `NULL` →
  false/0 in the rule columns and known values untouched; non-rule columns and the four tables equal cell by
  cell; schema and types; the input check; the safe write; the publisher refusing a non-gold file and a `NULL` in
  a rule column.
- **Proof (2026-09-24, real raw data, in a temp folder):** a semi-raw DB built by the new code → gold, compared
  cell by cell with today's `data/cars.duckdb` (built by the old code, old contract):
  - columns: old minus `description_clean` = gold (117; same names, order and types); 45,277 rows, same ids;
  - the only differing column is `description_text` (the page heading is gone in every row). Against the old
    `description_clean` only 778 rows differ (the seller's leading "-" / ":"); the 114 heading-only rows are
    `NULL`;
  - `duplicate_ad_ids`, `price_history` and both caches identical;
  - cells gold fills: 59,402 (heavy damage ×2 + first owner), 293,988 (panel flags), 0 (counters);
  - `publish --dry-run`: gold passed; the semi-raw DB and today's old DB were refused.
- **What the API will see: nothing different** (checked 2026-09-25 in the API code and with the API's own
  functions, `sadik-portfolio/api`). The API reads 55 columns of `car_listings` only; `description_clean`,
  `description_text`, `kb_paint_change_summary`, `kb_is_heavy_damaged`, `gb_is_first_owner` and the four other
  tables are never read, so no API fix is needed. The API also gives identical output on the semi-raw (silver)
  file (dashboard rows, snapshot list, drift), since its code implicitly counts NULL as 0. Gold is still what gets
  published: an explicit contract instead of relying on that implicit behaviour (owner's decision).
- **The real files (2026-09-24):** `data/cars.duckdb` was rebuilt semi-raw and `data/cars_gold.duckdb` derived
  from it. The proof was repeated on the real files: gold equals the backed-up old DB cell for cell except the
  description (778 / 114 rows, the other tables identical). Nothing was uploaded to S3; publishing is the
  owner's call.
- **In the report (2026-09-24):** technical report §1 → "The data the API gets (gold)". Its numbers are counted
  by `analysis/01_gold_contract.py` (over every snapshot row, rules from `db/gold_rules.json`): 353,390 cells are
  filled (59,402 heavy damage + first owner, 293,988 panel flags), `kb_paint_change_summary` is not sent, 114
  descriptions are empty.
