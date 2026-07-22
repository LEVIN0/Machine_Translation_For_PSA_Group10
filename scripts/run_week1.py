#!/usr/bin/env python3
"""CLI entry point for the Week 1 data collection pipeline.

Run from the project root:
    python scripts/run_week1.py --no-scrape          # fast offline run
    python scripts/run_week1.py                      # full run with scraping
"""

import argparse
import sys
from pathlib import Path

# Make the project root importable when run as a script from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from SRC.build_dataset import build  # noqa: E402
from SRC.report import generate_report  # noqa: E402


def parse_args(argv=None):
    """Parse command-line arguments for the Week 1 pipeline."""
    parser = argparse.ArgumentParser(
        description="Build the Week 1 PSA parallel dataset (EN-SW).")
    parser.add_argument("--no-scrape", action="store_true",
                        help="skip all website scraping (offline run)")
    parser.add_argument("--sites", type=str, default=None,
                        help="comma-separated site names to scrape (default: all)")
    parser.add_argument("--max-pages", type=int, default=None,
                        help="cap article pages per site")
    parser.add_argument("--no-tico", action="store_true",
                        help="skip the TICO-19 corpus import")
    parser.add_argument("--tico-max", type=int, default=None,
                        help="cap the number of TICO-19 pairs imported")
    parser.add_argument("--no-tatoeba", action="store_true",
                        help="skip the Tatoeba import")
    parser.add_argument("--report", dest="report", action="store_true",
                        default=True, help="generate the Week 1 report (default)")
    parser.add_argument("--no-report", dest="report", action="store_false",
                        help="skip report generation")
    return parser.parse_args(argv)


def main(argv=None):
    """Build the dataset, then (unless --no-report) generate the report."""
    args = parse_args(argv)
    site_names = ([s.strip() for s in args.sites.split(",") if s.strip()]
                  if args.sites else None)

    csv_path = build(
        scrape=not args.no_scrape,
        site_names=site_names,
        max_pages=args.max_pages,
        use_tico=not args.no_tico,
        use_tatoeba=not args.no_tatoeba,
        tico_max=args.tico_max,
    )
    if args.report:
        generate_report(csv_path=csv_path)
    print("[run_week1] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
