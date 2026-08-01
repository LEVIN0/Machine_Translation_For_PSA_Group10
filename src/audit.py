"""PSA framework audit — a step of the Week 1 build pipeline.

Every collected row is classified against the lecturer's 4-step PSA
framework (PSA FRAMEWORK.pdf) by the frozen, lecturer-calibrated scoring
classifier in ``src/psa_classify.py``. Each row's ``psa_class`` is stamped
into its Metadata so the decision is auditable per row. Scraped rows that
fail the audit are deleted; corpus (TICO-19), manual (team-written) and
gold (lecturer) rows are exempt and kept whole.

The audit runs inside ``src/build_dataset.py`` after the scrape/corpus/
manual sources are concatenated and before the lecturer gold merge and
cleaning. ``write_audit_report`` renders the audit to
``reports/framework_audit.md``.
"""

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .psa_classify import classify_frame


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


def apply_framework_audit(df, verbose=True):
    """Classify all rows, stamp psa_class, delete scraped non-PSA rows.

    Returns (kept_df, info) where info carries the deleted frame, per-type
    kept/before counts and the class distribution for the audit report.
    Only rows with Metadata.type == "scraped" are eligible for deletion.
    """
    classified = classify_frame(df)
    meta = classified["Metadata"].map(_parse_metadata)
    row_type = meta.map(lambda m: m.get("type", ""))
    is_scraped = row_type == "scraped"

    # Stamp psa_class into Metadata for EVERY row (exempt rows too, for
    # analysis); serialize back to a JSON string per the established pattern.
    classified["Metadata"] = [
        _dump_metadata({**m, "psa_class": cls})
        for m, cls in zip(meta, classified["psa_class"])
    ]

    delete_mask = is_scraped & (classified["psa_class"] != "PSA")
    deleted = classified[delete_mask]
    kept = classified[~delete_mask].drop(columns=["psa_class", "psa_score"])

    info = {
        "deleted": deleted,
        "rows_before": int(len(classified)),
        "rows_after": int(len(kept)),
        "deleted_total": int(len(deleted)),
        "n_scraped": int(is_scraped.sum()),
        "class_distribution": dict(Counter(classified["psa_class"])),
        "type_before": dict(Counter(t or "(no type)" for t in row_type)),
        "type_kept": dict(Counter(
            t or "(no type)"
            for t in row_type[~delete_mask])),
    }
    if verbose:
        print(f"[audit] {info['n_scraped']} scraped rows, "
              f"{info['deleted_total']} deleted (non-PSA), "
              f"{info['rows_after']} kept")
    return kept, info


def render_audit_report(info, cleaned, lecturer_rows, built_at=None):
    """Render reports/framework_audit.md as markdown text.

    ``info`` is the dict from apply_framework_audit; ``cleaned`` is the
    final dataset frame (after gold merge + cleaning); ``lecturer_rows``
    is the number of gold records merged (0 when the gold source is off).
    """
    built_at = built_at or datetime.now(timezone.utc).isoformat(
        timespec="seconds")
    deleted = info["deleted"]
    all_types = sorted(set(info["type_before"]) | set(info["type_kept"]))

    lines = []
    lines.append("# Framework audit — PSA classification\n")
    lines.append(f"_Generated {built_at} by the framework-audit step of the "
                 "Week 1 build (`src/audit.py`, run via "
                 "`scripts/run_week1.py`)_")
    lines.append("")
    lines.append("## Methodology\n")
    lines.append(
        "Every collected row is audited against the lecturer's 4-step PSA "
        "framework (PSA FRAMEWORK.pdf) using the frozen, lecturer-calibrated "
        "scoring classifier in `src/psa_classify.py`. The classifier is a "
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

    lines.append("## Kept / dropped per source (Metadata.type, audit phase)\n")
    lines.append("| source type | before | dropped | kept |")
    lines.append("|---|---:|---:|---:|")
    for t in all_types:
        n_before = info["type_before"].get(t, 0)
        n_kept = info["type_kept"].get(t, 0)
        lines.append(f"| {t} | {n_before} | {n_before - n_kept} | {n_kept} |")
    lines.append("")

    lines.append("## Kept rows per domain\n")
    lines.append("| domain | rows |")
    lines.append("|---|---:|")
    for dom, n in cleaned["Domain"].value_counts().sort_index().items():
        lines.append(f"| {dom} | {n} |")
    lines.append("")

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

    lines.append("## Totals\n")
    lines.append(f"- Rows before the audit: **{info['rows_before']}**")
    lines.append(f"- Rows after the audit (pre-cleaning): "
                 f"**{info['rows_after']}**")
    lines.append(f"- Deleted (scraped non-PSA): **{info['deleted_total']}**")
    lines.append(f"- Class distribution (all rows, pre-deletion): "
                 f"`{json.dumps(info['class_distribution'], ensure_ascii=False)}`")
    lines.append("")

    lines.append("## Lecturer gold merge\n")
    lines.append(f"- Gold rows merged: **{lecturer_rows}** "
                 "(from `PSA_KE_Final.csv`)")
    lines.append("")
    return "\n".join(lines)


def write_audit_report(info, cleaned, lecturer_rows, reports_dir,
                       built_at=None, verbose=True):
    """Write reports/framework_audit.md; return the path."""
    path = Path(reports_dir) / "framework_audit.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_audit_report(info, cleaned, lecturer_rows,
                                        built_at=built_at),
                    encoding="utf-8")
    if verbose:
        print(f"[audit] report -> {path}")
    return path
