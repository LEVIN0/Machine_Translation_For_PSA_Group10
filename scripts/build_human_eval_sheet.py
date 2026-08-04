#!/usr/bin/env python3
"""Week 4 human evaluation sheet builder (docs/week4_human_eval_guide.md).

Unlike the Week 2 `data/validation/validation_subset.csv` (native-speaker
review of the *dataset's* EN-SW translations), this samples the *model's*
Ekegusii output from `scripts/run_week4_eval.py` predictions, for a native
speaker to rate fluency/adequacy of the MT system itself.

Reads reports/week4_eval/predictions/en-guz.csv (English -> model Ekegusii,
with the held-out gold Ekegusii as reference) and writes a stratified
(by Domain), seeded sample with empty reviewer columns.

Run from the project root, after scripts/run_week4_eval.py:

    python scripts/build_human_eval_sheet.py
    python scripts/build_human_eval_sheet.py --n-per-domain 8 --seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_PREDICTIONS = "reports/week4_eval/predictions/en-guz.csv"
DEFAULT_OUT = "data/validation/week4_model_output_review.csv"
REVIEWER_COLS = ["Reviewer", "Fluency_1to5", "Adequacy_1to5", "Issues", "Notes"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the Week 4 model-output human-eval sheet.")
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    parser.add_argument("--n-per-domain", type=int, default=6,
                        help="rows sampled per PSA domain (default: 6)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    pred_path = Path(args.predictions)
    if not pred_path.is_file():
        parser.error(f"{pred_path} not found — run scripts/run_week4_eval.py "
                     "first (it writes the en-guz predictions this reads)")

    df = pd.read_csv(pred_path, encoding="utf-8").fillna("")

    parts = []
    for domain, group in df.groupby("Domain"):
        n = min(args.n_per_domain, len(group))
        parts.append(group.sample(n=n, random_state=args.seed))
    sampled = pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]

    sheet = sampled[["PSA_ID", "Domain", "English", "Ekegusii", "hypothesis"]].rename(
        columns={"Ekegusii": "Reference_Ekegusii", "hypothesis": "Model_Ekegusii"})
    for col in REVIEWER_COLS:
        sheet[col] = ""

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[build_human_eval_sheet] {len(sheet)} rows across "
          f"{sheet['Domain'].nunique()} domains -> {out_path}")
    print("[build_human_eval_sheet] see docs/week4_human_eval_guide.md "
          "for reviewer instructions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
