"""Re-run the cleaning pipeline on the existing processed dataset.

Use this when cleaning rules were improved AFTER a scrape run, so you do not
have to re-scrape (which takes ~20 minutes). It reads
data/processed/psa_parallel_week1.csv, applies SRC.cleaning.clean again,
re-assigns IDs, rewrites the CSV + build_stats.json, and regenerates the
Week 1 report.

Usage (from the project root):
    python scripts/reclean.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make the project root importable when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from SRC import config
from SRC.cleaning import clean
from SRC.report import generate_report
from SRC.schema import assign_ids, validate_schema


def main():
    csv_path = config.DATASET_CSV
    if not csv_path.exists():
        print(f"[reclean] no dataset found at {csv_path}; run scripts/run_week1.py first")
        return 1

    df = pd.read_csv(csv_path, dtype=str).fillna("")
    # Rows already have IDs; drop them so clean -> re-assign is deterministic.
    df = df.drop(columns=["PSA_ID"], errors="ignore")

    cleaned, stats = clean(df)
    cleaned = assign_ids(cleaned)

    problems = validate_schema(cleaned)
    if problems:
        for p in problems:
            print(f"[reclean] SCHEMA PROBLEM: {p}")
        return 1

    cleaned.to_csv(csv_path, index=False, encoding="utf-8")

    payload = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "reclean (no rescrape)",
        "cleaning": stats,
        "rows_total": int(len(cleaned)),
        "rows_per_source": cleaned["Source"].value_counts().to_dict(),
    }
    config.STATS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[reclean] wrote {len(cleaned)} rows -> {csv_path}")
    print(f"[reclean] cleaning: {stats}")

    report = generate_report(csv_path)
    print(f"[reclean] report -> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
