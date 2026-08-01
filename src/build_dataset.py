"""Build pipeline: collect -> audit -> merge gold -> clean -> IDs -> write.

Pipeline order (one command: ``python scripts/run_week1.py``):
  1. collect frames: scraped sites, TICO-19, Tatoeba (optional), team-written
  2. framework audit (src/audit.py): classify every row, stamp psa_class in
     Metadata, delete scraped rows that fail the PSA framework
  3. merge the lecturer gold dataset (verbatim, exempt from the audit)
  4. clean (dedupe, language checks, length/domain filters), assign IDs,
     validate against the schema
  5. write the dataset CSV, build_stats.json and reports/framework_audit.md
"""

import json
from datetime import datetime, timezone

import pandas as pd

from .audit import apply_framework_audit, write_audit_report
from .cleaning import clean
from .collectors import collect_all
from .config import DATASET_CSV, EXTERNAL_DIR, REPORTS_DIR, STATS_JSON, ensure_dirs
from .corpora import import_manual, import_tatoeba, import_tico19, load_lecturer
from .schema import COLUMNS, assign_ids, validate_schema

DEFAULT_LECTURER_PATH = EXTERNAL_DIR / "PSA_KE_Final.csv"


def _rows_per_type(df):
    """Final-row counts per Metadata.type (gold/scraped/corpus/manual)."""
    from .audit import _parse_metadata
    counts = {}
    for raw in df["Metadata"]:
        t = _parse_metadata(raw).get("type", "") or "(no type)"
        counts[t] = counts.get(t, 0) + 1
    return dict(sorted(counts.items()))


def build(scrape=True, site_names=None, max_pages=None, use_tico=True,
          use_tatoeba=True, tico_max=None, use_manual=True,
          use_lecturer=True, lecturer_path=None, run_audit=True,
          verbose=True):
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

    if use_manual:
        try:
            manual = import_manual(verbose=verbose)
        except Exception as exc:
            print(f"[build] WARNING: manual import failed ({exc}); continuing")
            manual = []
        if manual:
            frames.append(pd.DataFrame(manual))
        source_counts["manual"] = len(manual)

    if frames:
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.DataFrame(columns=COLUMNS)
        if verbose:
            print("[build] No records collected from any source; "
                  "writing an empty (schema-valid) dataset.")

    # -- framework audit: classify, stamp psa_class, drop scraped non-PSA ----
    audit_info = None
    if run_audit and len(df):
        df, audit_info = apply_framework_audit(df, verbose=verbose)

    # -- lecturer gold merge (verbatim, exempt from the audit) ----------------
    lecturer_records = []
    if use_lecturer:
        path = lecturer_path or DEFAULT_LECTURER_PATH
        try:
            lecturer_records = load_lecturer(path, verbose=verbose)
        except Exception as exc:
            print(f"[build] WARNING: lecturer gold import failed ({exc}); "
                  f"continuing")
            lecturer_records = []
        if lecturer_records:
            df = pd.concat([df[COLUMNS], pd.DataFrame(lecturer_records)],
                           ignore_index=True)
        source_counts["lecturer_gold"] = len(lecturer_records)

    df, stats = clean(df)
    df = assign_ids(df)

    problems = validate_schema(df)
    if problems:
        raise ValueError(f"Schema validation failed: {problems}")

    df.to_csv(DATASET_CSV, index=False, encoding="utf-8")

    built_at = datetime.now(timezone.utc).isoformat()
    stats_payload = {
        "built_at": built_at,
        "source_counts": source_counts,
        "framework_audit": (
            {k: v for k, v in audit_info.items()
             if k not in ("deleted", "type_before", "type_kept")}
            if audit_info else None),
        "cleaning": stats,
        "output_rows": len(df),
        "output_rows_per_type": _rows_per_type(df),
    }
    STATS_JSON.write_text(json.dumps(stats_payload, indent=2), encoding="utf-8")

    if audit_info is not None:
        write_audit_report(audit_info, df, len(lecturer_records),
                           REPORTS_DIR, built_at=built_at, verbose=verbose)

    if verbose:
        print(f"[build] wrote {len(df)} rows -> {DATASET_CSV}")
        print(f"[build] stats -> {STATS_JSON}")
        print(f"[build] cleaning: {stats}")
    return DATASET_CSV
