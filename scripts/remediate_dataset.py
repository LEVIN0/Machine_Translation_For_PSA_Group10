"""One-command remediation pipeline (SPEC_REMEDIATION.md §3).

Merges the lecturer gold dataset, audits every row against the frozen PSA
framework classifier, DELETES scraped rows that fail the audit (corpus /
manual / gold rows are exempt and kept whole), re-cleans, re-assigns IDs,
validates, and writes the dataset CSV plus reports/framework_audit.md and
build_stats.json.

Usage (from the project root):
    python scripts/remediate_dataset.py --lecturer data/external/PSA_KE_Final.csv
        [--dataset data/processed/psa_parallel_week1.csv] [--dry-run]

Dry-run: everything except writing the dataset CSV.
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from SRC import config
from SRC.cleaning import clean
from SRC.corpora.lecturer import load_lecturer
from SRC.psa_classify import classify_frame
from SRC.schema import COLUMNS, assign_ids, validate_schema


def _parse_metadata(raw):
    """Parse a Metadata cell (JSON string, dict, or missing) into a dict."""
    if isinstance(raw, dict):
        return dict(raw)
    if raw is None:
        return {}
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return {}
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _dump_metadata(meta):
    """Serialize a Metadata dict the same way schema.new_record does."""
    return json.dumps(meta, ensure_ascii=False)


def _truncate(text, width=100):
    text = str(text)
    return text if len(text) <= width else text[: width - 1].rstrip() + "…"


def remediate(dataset_path, lecturer_path, dry_run=False, verbose=True):
    """Run the remediation pipeline; return a stats dict (or None on error).

    Writes the dataset CSV (unless dry_run), reports/framework_audit.md and
    build_stats.json (next to the dataset CSV).
    """
    dataset_path = Path(dataset_path)
    lecturer_path = Path(lecturer_path)
    if not dataset_path.exists():
        print(f"[remediate] no dataset at {dataset_path}; nothing to remediate")
        return None

    df = pd.read_csv(dataset_path, dtype=str, encoding="utf-8").fillna("")
    rows_before = len(df)
    if verbose:
        print(f"[remediate] dataset: {rows_before} rows from {dataset_path}")

    # -- 1. lecturer gold rows (optional) ------------------------------------
    lecturer_records = load_lecturer(lecturer_path, verbose=verbose)

    # -- 2. classify ALL rows; only Metadata.type=="scraped" can be deleted --
    classified = classify_frame(df)
    meta = classified["Metadata"].map(_parse_metadata)
    row_type = meta.map(lambda m: m.get("type", ""))
    is_scraped = row_type == "scraped"

    # Stamp psa_class into Metadata for EVERY row (exempt rows too, for
    # analysis); serialize back to a JSON string per the established pattern.
    stamped = [
        _dump_metadata({**m, "psa_class": cls})
        for m, cls in zip(meta, classified["psa_class"])
    ]
    classified["Metadata"] = stamped

    delete_mask = is_scraped & (classified["psa_class"] != "PSA")
    deleted = classified[delete_mask]
    kept = classified[~delete_mask].drop(columns=["psa_class", "psa_score"])
    if verbose:
        print(f"[remediate] audit: {int(is_scraped.sum())} scraped rows, "
              f"{len(deleted)} deleted (non-PSA), {len(kept)} kept")

    # -- 3. merge lecturer gold -> clean -> assign IDs -> validate -----------
    frames = [kept[COLUMNS]]
    if lecturer_records:
        frames.append(pd.DataFrame(lecturer_records))
    combined = pd.concat(frames, ignore_index=True)
    cleaned, clean_stats = clean(combined)
    cleaned = assign_ids(cleaned)

    problems = validate_schema(cleaned)
    if problems:
        for p in problems:
            print(f"[remediate] SCHEMA PROBLEM: {p}")
        return None

    if dry_run:
        print(f"[remediate] DRY-RUN: dataset CSV NOT written "
              f"(would write {len(cleaned)} rows -> {dataset_path})")
    else:
        cleaned.to_csv(dataset_path, index=False, encoding="utf-8")
        if verbose:
            print(f"[remediate] wrote {len(cleaned)} rows -> {dataset_path}")

    # -- 4. framework audit report --------------------------------------------
    stats = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "remediate",
        "dry_run": dry_run,
        "dataset": str(dataset_path),
        "lecturer_file": str(lecturer_path),
        "rows_before": rows_before,
        "rows_after": int(len(cleaned)),
        "deleted_total": int(len(deleted)),
        "deleted_per_source": dict(Counter(
            t or "(no type)" for t in row_type[delete_mask])),
        "class_distribution": dict(Counter(classified["psa_class"])),
        "lecturer_rows": len(lecturer_records),
        "cleaning": clean_stats,
    }

    report_path = Path(config.REPORTS_DIR) / "framework_audit.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _render_audit(classified, deleted, kept, cleaned,
                      lecturer_records, stats),
        encoding="utf-8")
    if verbose:
        print(f"[remediate] audit report -> {report_path}")

    stats_path = dataset_path.parent / "build_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False),
                          encoding="utf-8")
    if verbose:
        print(f"[remediate] stats -> {stats_path}")
    return stats


def _render_audit(classified, deleted, kept, cleaned,
                  lecturer_records, stats):
    """Render reports/framework_audit.md (spec §3 step 5)."""
    meta_before = classified["Metadata"].map(_parse_metadata)
    type_before = meta_before.map(lambda m: m.get("type", "") or "(no type)")
    meta_kept = kept["Metadata"].map(_parse_metadata)
    type_kept = meta_kept.map(lambda m: m.get("type", "") or "(no type)")

    lines = []
    lines.append("# Framework audit — PSA classification remediation\n")
    lines.append(f"_Generated {stats['built_at']} by "
                 "`scripts/remediate_dataset.py`"
                 + (" (DRY-RUN: dataset CSV not written)_" if stats["dry_run"]
                    else "_"))
    lines.append("")
    lines.append("## Methodology\n")
    lines.append(
        "Every dataset row was audited against the lecturer's 4-step PSA "
        "framework (PSA FRAMEWORK.pdf) using the frozen, lecturer-calibrated "
        "scoring classifier in `SRC/psa_classify.py`. The classifier is a "
        "documented heuristic: directive/imperative signals, audience markers "
        "and call-to-action phrases add score; press-release, legal-notice, "
        "encyclopedic and connective-continuation signals subtract. Rows "
        "scoring >= 2 are PSA. Per the locked team decision, only rows with "
        "Metadata.type == \"scraped\" are eligible for deletion; TICO-19 "
        "(corpus), team-written (manual) and lecturer gold rows are exempt "
        "and kept whole, with their psa_class recorded in Metadata for "
        "analysis. Deleted scraped rows are removed entirely from the "
        "dataset.")
    lines.append("")

    # -- per-source kept/dropped table (source = Metadata.type; audit phase,
    #    i.e. before the lecturer merge — merge stats are reported below) ----
    lines.append("## Kept / dropped per source (Metadata.type, audit phase)\n")
    lines.append("| source type | before | dropped | kept |")
    lines.append("|---|---:|---:|---:|")
    all_types = sorted(set(type_before) | set(type_kept))
    for t in all_types:
        n_before = int((type_before == t).sum())
        n_kept = int((type_kept == t).sum())
        lines.append(f"| {t} | {n_before} | {n_before - n_kept} | {n_kept} |")
    lines.append("")

    # -- per-domain kept table (final dataset, after merge + clean) ----------

    # -- per-domain kept table -------------------------------------------------
    lines.append("## Kept rows per domain\n")
    lines.append("| domain | rows |")
    lines.append("|---|---:|")
    for dom, n in cleaned["Domain"].value_counts().sort_index().items():
        lines.append(f"| {dom} | {n} |")
    lines.append("")

    # -- sample of deleted rows -------------------------------------------------
    lines.append("## Sample of DELETED rows (up to 10)\n")
    if len(deleted):
        lines.append("| Source | psa_class | score | English (truncated) |")
        lines.append("|---|---|---:|---|")
        for _, r in deleted.head(10).iterrows():
            en = _truncate(str(r["English"]).replace("|", "\\|")
                           .replace("\n", " "))
            lines.append(f"| {r['Source']} | {r['psa_class']} | "
                         f"{r['psa_score']} | {en} |")
    else:
        lines.append("(no rows were deleted)")
    lines.append("")

    # -- totals ------------------------------------------------------------------
    lines.append("## Totals\n")
    lines.append(f"- Rows before: **{stats['rows_before']}**")
    lines.append(f"- Rows after: **{stats['rows_after']}**")
    lines.append(f"- Deleted (scraped non-PSA): **{stats['deleted_total']}**")
    lines.append(f"- Class distribution (all rows, pre-deletion): "
                 f"`{json.dumps(stats['class_distribution'], ensure_ascii=False)}`")
    lines.append("")

    # -- lecturer merge stats -----------------------------------------------------
    lines.append("## Lecturer gold merge\n")
    n_lec = len(lecturer_records)
    lines.append(f"- Gold rows imported: **{n_lec}** "
                 f"(from `{Path(stats['lecturer_file']).name}`)")
    if n_lec:
        lec_df = pd.DataFrame(lecturer_records)
        domains = lec_df["Domain"].value_counts().sort_index()
        lines.append("- Domains: " + ", ".join(
            f"{d} ({n})" for d, n in domains.items()))
        n_sw = int((lec_df["Kiswahili"].str.strip() != "").sum())
        n_guz = int((lec_df["Ekegusii"].str.strip() != "").sum())
        lec_meta = lec_df["Metadata"].map(_parse_metadata)
        n_dholuo = sum(1 for m in lec_meta if m.get("dholuo"))
        n_somali = sum(1 for m in lec_meta if m.get("somali"))
        lines.append(f"- Translation coverage: Kiswahili {n_sw}/{n_lec}, "
                     f"Ekegusii {n_guz}/{n_lec}, Dholuo {n_dholuo}/{n_lec}, "
                     f"Somali {n_somali}/{n_lec}")
    else:
        lines.append("- (lecturer file missing or empty — merge skipped)")
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lecturer", default=str(config.EXTERNAL_DIR
                                              / "PSA_KE_Final.csv"),
                    help="path to the lecturer gold CSV (PSA_KE_Final.csv)")
    ap.add_argument("--dataset", default=str(config.DATASET_CSV),
                    help="path to the dataset CSV to remediate")
    ap.add_argument("--dry-run", action="store_true",
                    help="run everything except writing the dataset CSV")
    args = ap.parse_args(argv)

    stats = remediate(args.dataset, args.lecturer, dry_run=args.dry_run)
    if stats is None:
        return 1
    print(f"[remediate] done: {stats['rows_before']} -> {stats['rows_after']} "
          f"rows ({stats['deleted_total']} scraped non-PSA deleted, "
          f"+{stats['lecturer_rows']} lecturer gold)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
