#!/usr/bin/env python3
"""CLI entry point for the Week 2 pipeline: preprocessing, EDA, splits, and
the native-speaker validation subset.

Run from the project root:
    python scripts/run_week2.py                       # full Week 2 pipeline
    python scripts/run_week2.py --no-figures          # skip matplotlib figures
    python scripts/run_week2.py --val-size 200 --seed 7
"""

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

# Make the project root importable when run as a script from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from SRC import config  # noqa: E402
from SRC.eda import compute_eda, make_figures, write_eda_report  # noqa: E402
from SRC.preprocessing import preprocess_dataframe  # noqa: E402
from SRC.splits import make_splits, save_splits  # noqa: E402

VALIDATION_EXTRA_COLUMNS = ["Reviewer", "Fluency_1to5", "Adequacy_1to5",
                            "Issues", "Notes"]


def parse_args(argv=None):
    """Parse command-line arguments for the Week 2 pipeline."""
    parser = argparse.ArgumentParser(
        description="Week 2: preprocessing, EDA, splits, validation subset.")
    parser.add_argument("--input", type=Path, default=None,
                        help="input dataset CSV (default: config.DATASET_CSV)")
    parser.add_argument("--no-figures", action="store_true",
                        help="skip matplotlib figure generation")
    parser.add_argument("--val-size", type=int, default=500,
                        help="size of the native-speaker validation subset "
                             "(default: 500)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for splits and sampling (default: 42)")
    return parser.parse_args(argv)


def make_validation_subset(df, val_size=500, seed=42):
    """Stratified (by Domain) random sample of up to `val_size` rows.

    Proportional allocation with largest-remainder rounding; every domain
    present in the data is represented when val_size >= #domains. Adds the
    empty reviewer columns (Reviewer, Fluency_1to5, Adequacy_1to5, Issues,
    Notes) for the native-speaker validation pass.
    """
    val_size = min(val_size, len(df))
    if val_size <= 0:
        subset = df.head(0).copy()
    else:
        domains = df["Domain"].value_counts()
        exact = {d: n * val_size / len(df) for d, n in domains.items()}
        quota = {d: int(math.floor(v)) for d, v in exact.items()}
        remainder = val_size - sum(quota.values())
        order = sorted(exact, key=lambda d: (exact[d] - quota[d]), reverse=True)
        for d in order[:remainder]:
            quota[d] += 1

        parts = []
        for domain, n in quota.items():
            if n <= 0:
                continue
            pool = df[df["Domain"] == domain]
            parts.append(pool.sample(n=min(n, len(pool)), random_state=seed))
        subset = (pd.concat(parts).sample(frac=1.0, random_state=seed)
                  .reset_index(drop=True))
    for col in VALIDATION_EXTRA_COLUMNS:
        subset[col] = ""
    return subset


def write_validation_guide(out_path=None):
    """Write docs/validation_guide.md for the native-speaker reviewers."""
    out_path = Path(out_path or (config.BASE_DIR / "docs" / "validation_guide.md"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Native-Speaker Validation Guide (Week 2)",
        "",
        "File to review: `data/validation/validation_subset.csv` — a stratified",
        "(by Domain) random sample of the dataset. Each reviewer fills the empty",
        "columns: `Reviewer`, `Fluency_1to5`, `Adequacy_1to5`, `Issues`, `Notes`.",
        "",
        "## What to rate",
        "",
        "For each row, compare the **English** text with its **Kiswahili**",
        "translation and score two dimensions on a 1–5 scale:",
        "",
        "- **Fluency_1to5** — is the Kiswahili grammatical and natural, ignoring",
        "  the English? 5 = flawless native Kiswahili; 1 = ungrammatical/",
        "  incomprehensible.",
        "- **Adequacy_1to5** — does the Kiswahili convey the same meaning as the",
        "  English? 5 = full meaning preserved; 1 = meaning lost or wrong.",
        "",
        "Rows with an **empty Kiswahili cell are not ratable**: leave the score",
        "columns blank and write `missing translation` under `Issues`.",
        "",
        "## Issues to flag (comma-separated in `Issues`)",
        "",
        "- `mistranslation` — the Kiswahili says something different.",
        "- `omission` — part of the English meaning is missing.",
        "- `addition` — the Kiswahili adds information not in the English.",
        "- `grammar` — spelling, agreement, or word-order errors.",
        "- `cultural term` — a glossary term (see `data/glossary.json`, e.g.",
        "  harambee, matatu, NHIF) is translated inconsistently or wrongly.",
        "",
        "Use `Notes` for anything else (e.g. awkward register, dialect variant,",
        "orthographic variation worth normalizing). Write your name or initials",
        "in `Reviewer` on every row you score.",
        "",
        "## Rules",
        "",
        "1. Rate independently — do not discuss scores while reviewing.",
        "2. Do not edit the English, Kiswahili, or Ekegusii columns; only fill",
        "   the reviewer columns.",
        "3. When unsure between two scores, choose the lower one and explain in",
        "   `Notes`.",
        "4. Save the completed file as `validation_subset_reviewed_<name>.csv`",
        "   and share it with the integration lead.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[week2] wrote validation guide -> {out_path}")
    return out_path


def main(argv=None):
    """Run the Week 2 pipeline end-to-end; returns 0 on success."""
    args = parse_args(argv)
    input_csv = args.input or config.DATASET_CSV
    seed = args.seed

    print(f"[week2] loading {input_csv}")
    df = pd.read_csv(input_csv, dtype=str).fillna("")

    # 1-2. Preprocess and save.
    pre = preprocess_dataframe(df)
    preprocessed_path = config.PROCESSED_DIR / "psa_preprocessed.csv"
    preprocessed_path.parent.mkdir(parents=True, exist_ok=True)
    pre.to_csv(preprocessed_path, index=False, encoding="utf-8")
    print(f"[week2] preprocessed {len(pre)} rows -> {preprocessed_path}")

    # 3. EDA statistics + figures + report.
    stats = compute_eda(pre)
    figures = [] if args.no_figures else make_figures(pre)
    report_path = write_eda_report(stats, figures)

    # 4. Train/dev/test splits.
    train, dev, test = make_splits(df, seed=seed)
    splits_stats = save_splits(train, dev, test, seed=seed)

    # 5. Native-speaker validation subset + reviewer guide.
    subset = make_validation_subset(df, val_size=args.val_size, seed=seed)
    val_path = config.BASE_DIR / "data" / "validation" / "validation_subset.csv"
    val_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(val_path, index=False, encoding="utf-8")
    print(f"[week2] validation subset ({len(subset)} rows) -> {val_path}")
    guide_path = write_validation_guide()

    print("[week2] ---- summary ----")
    print(f"[week2] rows: {len(df)} | paired EN-SW: {stats['paired']} "
          f"({stats['paired_share'] * 100:.1f}%)")
    print(f"[week2] splits: train={len(train)} dev={len(dev)} test={len(test)} "
          f"(stats: {splits_stats})")
    print(f"[week2] EDA report: {report_path} "
          f"({len(figures)} figures{' [skipped]' if args.no_figures else ''})")
    print(f"[week2] validation: {val_path} ({len(subset)} rows), "
          f"guide: {guide_path}")
    print("[week2] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
