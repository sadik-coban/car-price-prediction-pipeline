import pandas as pd
import jsonlines
import time
import random
import json
from datetime import datetime, timezone

from getdetails import get_car_details
from getlistofcars import scrape_links_price_blocks
from run_context import create_run_context

# --------------------
# RUN CONTEXT (🔥 ENFORCED)
# --------------------
ctx = create_run_context(brand="audi")
RUN_TS = ctx["run_ts"]

# --------------------
# SHORT ALIASES
# --------------------
BASE_DIR = ctx["run_dir"]
LINKS_FILE = ctx["links_file"]
OUTPUT_FILE = ctx["details_file"]
PROGRESS_FILE = ctx["progress_file"]

SLEEP_MIN = 0.01
SLEEP_MAX = 0.03


# --------------------
# PROGRESS UTILS
# --------------------
def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f).get("last_index", 0)
    return 0


def save_progress(i):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(
            {
                "last_index": i,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            f,
        )


# --------------------
# SCRAPE LINKS (aynı run içine)
# --------------------
scrape_links_price_blocks(
    min_price=200000,
    max_price=700000,
    step=200000,
    output_dir=str(BASE_DIR),
    brand=ctx["brand"],
)

df = pd.read_csv(LINKS_FILE)
links = df["url"].drop_duplicates().tolist()

start_i = load_progress()
print(f"▶ Brand: {ctx['brand']}")
print(f"▶ Run: {RUN_TS}")
print(f"▶ Starting from index: {start_i}")

# --------------------
# SCRAPE DETAILS
# --------------------
with jsonlines.open(OUTPUT_FILE, mode="a") as writer:
    for i, link in enumerate(links[start_i:], start=start_i):
        print(f"[{i+1}/{len(links)}] → {link}")

        try:
            data = get_car_details(link)
            data["error"] = None
        except Exception as e:
            data = {"error": str(e)}

        # data.update(
        #     {
        #         "brand": ctx["brand"],
        #         "link": link,
        #         "run_ts": RUN_TS,
        #         "scraped_at": datetime.now(timezone.utc).isoformat(),
        #     }
        # )

        writer.write(data)
        save_progress(i + 1)

        time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))
