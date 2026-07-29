"""Incrementally add scraped PSAs from specific sources to the EXISTING dataset.

Unlike run_week1.py (which rebuilds the dataset from scratch), this script:
  1. scrapes only the sites you name,
  2. appends the new records to data/processed/psa_parallel_week1.csv,
  3. re-runs cleaning over the combined data (dedup removes any overlap),
  4. re-assigns IDs, refreshes build_stats.json and the report.

Usage (from the project root):
    python scripts/scrape_more.py --sites dci_units,covaw_news
    python scripts/scrape_more.py --sites nacada_drug_prev --max-pages 10
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from SRC import config
from SRC.cleaning import clean
from SRC.collectors import collect_all
from SRC.report import generate_report
from SRC.schema import assign_ids, validate_schema


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sites", default="",
                    help="comma-separated site names (optional; omit to only "
                         "ingest data/manual/*.csv)")
    ap.add_argument("--max-pages", type=int, default=None,
                    help="override per-site page cap (default: config values)")
    args = ap.parse_args()

    names = [s.strip() for s in args.sites.split(",") if s.strip()]
    csv_path = config.DATASET_CSV
    if not csv_path.exists():
        print(f"[scrape_more] no dataset at {csv_path}; run scripts/run_week1.py first")
        return 1

    existing = pd.read_csv(csv_path, dtype=str).fillna("")
    rows_before = len(existing)
    print(f"[scrape_more] existing dataset: {rows_before} rows")
    print(f"[scrape_more] scraping: {', '.join(names)}")

    new_records = collect_all(names, max_pages=args.max_pages) if names else []

    # Also ingest any team-written PSAs placed in data/manual/*.csv
    from SRC.corpora.manual import import_manual
    manual_records = import_manual()
    if manual_records:
        print(f"[scrape_more] + {len(manual_records)} team-written records from data/manual/")
    new_records = new_records + manual_records

    if not new_records:
        print("[scrape_more] no new records collected; dataset unchanged")
        return 0

    new_df = pd.DataFrame(new_records)
    combined = pd.concat(
        [existing.drop(columns=["PSA_ID"], errors="ignore"), new_df],
        ignore_index=True,
    )
    cleaned, stats = clean(combined)
    cleaned = assign_ids(cleaned)

    problems = validate_schema(cleaned)
    if problems:
        for p in problems:
            print(f"[scrape_more] SCHEMA PROBLEM: {p}")
        return 1

    cleaned.to_csv(csv_path, index=False, encoding="utf-8")
    payload = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": f"scrape_more ({', '.join(names)})",
        "rows_before": rows_before,
        "scraped_raw": len(new_records),
        "rows_after": int(len(cleaned)),
        "net_added": int(len(cleaned) - rows_before),
        "cleaning": stats,
        "rows_per_source": cleaned["Source"].value_counts().to_dict(),
    }
    config.STATS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[scrape_more] scraped {len(new_records)} raw records; "
          f"net +{len(cleaned) - rows_before} rows after cleaning")
    print(f"[scrape_more] wrote {len(cleaned)} rows -> {csv_path}")
    report = generate_report(csv_path)
    print(f"[scrape_more] report -> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
