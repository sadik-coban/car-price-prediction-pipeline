# Database (`cars.duckdb`) notes

> 🇹🇷 Türkçe (varsayılan): [database.md](database.md) · ← [README](../README.en.md)

This file is the analysis' input; until now it also went to S3 via `publish_data_to_s3.py` (the API reads it).

- **Semi-raw (2026-09-24, the owner's principle: "everything in the DB should be semi-raw").** A value the
  page does not give is not guessed, it stays `NULL`; the analysis decides how to read "Belirtilmemiş"
  (unspecified). Applies from the next build (today's file was built on 2026-09-23 with the old code):
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

  To do in the analysis after the rebuild: `analysis/lib/text_flags.descriptions` reads `description_text`;
  `description_clean` leaves `02_missingness`'s `IDENTITY` list and `kb_paint_change_summary` gets a class
  (the §2 missingness table changes). No value that reaches the model changes: `load_clean` already counts
  these `NULL`s as 0.
- **Safe build.** The new DB is built in `<out>.tmp` and moved over the old one when complete; a failure
  leaves the old DB untouched. It stops before touching anything when `<out>.wal` exists, when the DB is open
  in another program, or when the raw data is incomplete (a brand folder, a snapshot without
  `details.jsonl`, an unreadable line). `duplicate_ad_ids.csv` is written next to the DB.
- **`engine_cc_val` means two things in two places.** In the DB it is the bucket's midpoint (the known
  bound for an open-ended bucket); the model's feature of the same name is `engine_cc_up` (the upper bound,
  reasoning in technical report §1). The DB column was not renamed, so an API reading it does not break.
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

## Next: the gold step (API) — not written yet

Owner's decision (2026-09-24): noted, not done now. The analysis reads the semi-raw DB; the live API (outside
this repo) expects the old contract: unknown heavy damage / first owner `false`, "Belirtilmemiş" panel `0`,
118 columns including `description_clean`. One file cannot serve both.

> ⚠️ **Order:** the gold step must be written before the semi-raw DB is actually rebuilt. Running
> `publish_data_to_s3.py` after such a rebuild without gold would send `NULL`s to the API and drop the
> `description_clean` column. Today's `data/cars.duckdb` has the old contract; S3 is safe today.

```
data/raw ──build_duckdb──► data/cars.duckdb (semi-raw, analysis) ──build_gold_db──► data/cars_gold.duckdb (API)
                                                                                      └─publish_data_to_s3─► S3 "data/cars.duckdb"
```

- **`db/build_gold_db.py`** only reads the semi-raw DB and writes `data/cars_gold.duckdb` safely (`.tmp`
  + move, `.wal` check). The rules sit in one list (`GOLD_RULES`) with their reasons:
  - `is_heavy_damaged`, `kb_is_heavy_damaged`, `gb_is_first_owner` → `COALESCE(x, false)`;
  - the 39 panel flags and the 3 damage counts → `COALESCE(x, 0)`;
  - `kb_paint_change_summary` is left out; `description_clean` = `description_text`, right after
    `description_text`. Result: today's 118 columns with the same names, types and order;
  - the other tables (`duplicate_ad_ids`, `price_history`, `dashboard_cache`, `options_cache`) as they are.
- **`publish_data_to_s3.py`**: default input `data/cars_gold.duckdb`; the S3 object name stays; the "file
  not found" hint names both steps.
- **Run order:** `build_duckdb.py` → `build_gold_db.py` → (owner) `publish_data_to_s3.py`.
- **Tests** (`tests/db/test_build_gold_db.py`): `NULL` → false/0 in the rule columns and non-`NULL`
  values untouched; non-rule columns and the four tables equal cell by cell; the schema equals today's; the
  safe write; the publisher's new default.
- **Proof:** gold is compared cell by cell with the DB the old code builds from the raw data. The old code is
  in `archive/obsolete/pipeline-yedek-2026-09-24/`; it was checked to reproduce today's DB exactly (the folder's
  README explains how to run it). Expected differences only in the description: `description_text` in every
  row (no heading; the 114 heading-only rows `NULL`), `description_clean` in exactly 778 rows. No other cell
  may change.
