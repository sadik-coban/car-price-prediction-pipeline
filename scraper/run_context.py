# utils/run_context.py
from datetime import datetime, timezone
from pathlib import Path
import os

ALLOWED_BRANDS = {"bmw", "audi", "mercedes", "all"}


def create_run_context(brand: str, run_ts: str | None = None):
    brand = brand.lower()

    if brand not in ALLOWED_BRANDS:
        raise ValueError(f"Invalid brand: {brand}")

    if run_ts is None:
        run_ts = os.getenv("RUN_TS") or datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H-%M-%SZ"
        )

    run_dir = Path("data/raw/cars") / brand / run_ts
    run_dir.mkdir(parents=True, exist_ok=True)

    return {
        "brand": brand,
        "run_ts": run_ts,
        "run_dir": run_dir,
        "links_file": run_dir / "links.csv",
        "details_file": run_dir / "details.jsonl",
        "progress_file": run_dir / "detail_progress.json",
    }
