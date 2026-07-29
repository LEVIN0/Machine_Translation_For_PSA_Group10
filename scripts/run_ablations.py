#!/usr/bin/env python3
"""Run the Week 3 ablation matrix (SPEC_WEEK3.md §2.8/§2.9).

Run from the project root:
    python scripts/run_ablations.py                          # full matrix
    python scripts/run_ablations.py --matrix quick           # smoke matrix
    python scripts/run_ablations.py --only zs_nllb,ft_nllb_base
    python scripts/run_ablations.py --table-only             # just the table

Each run trains (unless it is a zero-shot eval-only entry), evaluates on the
dev/FLORES specs and writes runs/<run>/evals/*.json. At the end the markdown
results table is written to reports/week3_results.md.
"""

import argparse
import sys
import time
from pathlib import Path

# Make the project root importable when run as a script from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.ablate import (DEFAULT_AUGMENTED_CSV, run_matrix,  # noqa: E402
                             standard_matrix, write_results_table)

RESULTS_MD = Path("reports/week3_results.md")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Week 3 ablation matrix and write the results "
                    "table.")
    parser.add_argument("--matrix", choices=["standard", "quick"],
                        default="standard",
                        help="'quick' caps max_samples=2000 and epochs=2")
    parser.add_argument("--only", default=None,
                        help="comma-separated run names to run "
                             "(e.g. zs_nllb,ft_nllb_base)")
    parser.add_argument("--eval-n", type=int, default=200,
                        help="max examples per eval spec (default: 200)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--augmented-csv", default=str(DEFAULT_AUGMENTED_CSV),
                        help="back-translation csv for ft_nllb_aug")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="re-run even when evals/*.json already exist")
    parser.add_argument("--table-only", action="store_true",
                        help="skip training/eval, just rebuild "
                             "reports/week3_results.md from runs/")
    args = parser.parse_args(argv)

    runs_root = Path(args.runs_root)

    if not args.table_only:
        matrix = standard_matrix(quick=args.matrix == "quick")
        if args.only:
            wanted = {s.strip() for s in args.only.split(",") if s.strip()}
            known = {c.run_name for c in matrix}
            unknown = wanted - known
            if unknown:
                parser.error(f"unknown run names {sorted(unknown)}; "
                             f"have: {sorted(known)}")
            matrix = [c for c in matrix if c.run_name in wanted]

        print(f"[ablations] {len(matrix)} run(s): "
              + ", ".join(c.run_name for c in matrix))
        t0 = time.perf_counter()
        for cfg in matrix:
            start = time.perf_counter()
            run_matrix([cfg], augmented_csv=Path(args.augmented_csv),
                       eval_n=args.eval_n, batch_size=args.batch_size,
                       skip_existing=not args.no_skip_existing)
            mins = (time.perf_counter() - start) / 60
            print(f"[ablations] {cfg.run_name} done in {mins:.1f} min",
                  flush=True)
        print(f"[ablations] matrix finished in "
              f"{(time.perf_counter() - t0) / 60:.1f} min")

    out = write_results_table(runs_root, RESULTS_MD)
    print(f"[ablations] results table -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
