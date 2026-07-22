"""Build pipeline: collect -> concat -> clean -> assign IDs -> validate -> write."""

import json
from datetime import datetime, timezone

import pandas as pd

from .cleaning import clean
from .collectors import collect_all
from .config import DATASET_CSV, STATS_JSON, ensure_dirs
from .corpora import import_tatoeba, import_tico19
from .schema import COLUMNS, assign_ids, validate_schema


def build(scrape=True, site_names=None, max_pages=None, use_tico=True,
          use_tatoeba=True, tico_max=None, verbose=True):
    """Build the Week 1 dataset and write DATASET_CSV + STATS_JSON.

    All source failures are non-fatal: a source that yields nothing simply
    contributes zero rows. Returns the path to the written CSV. Raises
    ValueError if the final frame fails schema validation.
    """
    ensure_dirs()
    frames = []
    source_counts = {}

    if scrape:
        try:
            scraped = collect_all(names=site_names, max_pages=max_pages,
                                  verbose=verbose)
        except Exception as exc:
            print(f"[build] WARNING: scraping failed entirely ({exc}); continuing")
            scraped = []
        if scraped:
            frames.append(pd.DataFrame(scraped))
        source_counts["scraped"] = len(scraped)

    if use_tico:
        try:
            tico = import_tico19(max_pairs=tico_max, verbose=verbose)
        except Exception as exc:
            print(f"[build] WARNING: TICO-19 import failed ({exc}); continuing. "
                  f"See data/external/tico19/README.txt for manual download.")
            tico = []
        if tico:
            frames.append(pd.DataFrame(tico))
        source_counts["tico19"] = len(tico)

    if use_tatoeba:
        try:
            tatoeba = import_tatoeba(verbose=verbose)
        except Exception as exc:
            print(f"[build] WARNING: Tatoeba import failed ({exc}); continuing")
            tatoeba = []
        if tatoeba:
            frames.append(pd.DataFrame(tatoeba))
        source_counts["tatoeba"] = len(tatoeba)

    if frames:
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.DataFrame(columns=COLUMNS)
        if verbose:
            print("[build] No records collected from any source; "
                  "writing an empty (schema-valid) dataset.")

    df, stats = clean(df)
    df = assign_ids(df)

    problems = validate_schema(df)
    if problems:
        raise ValueError(f"Schema validation failed: {problems}")

    df.to_csv(DATASET_CSV, index=False, encoding="utf-8")

    stats_payload = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source_counts": source_counts,
        "cleaning": stats,
        "output_rows": len(df),
    }
    STATS_JSON.write_text(json.dumps(stats_payload, indent=2), encoding="utf-8")

    if verbose:
        print(f"[build] wrote {len(df)} rows -> {DATASET_CSV}")
        print(f"[build] stats -> {STATS_JSON}")
        print(f"[build] cleaning: {stats}")
    return DATASET_CSV
