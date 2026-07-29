#!/usr/bin/env python3
"""Evaluate a checkpoint on registered eval specs (SPEC_WEEK3.md §2.9).

Run from the project root:
    python scripts/run_eval.py --checkpoint runs/ft_nllb_base/checkpoint-best
    python scripts/run_eval.py --checkpoint runs/ft_nllb_base/checkpoint-best \
        --specs psa_dev_en-sw,psa_test_en-guz --n 200

Results are printed as a table and written to
runs/<run>/evals/<spec>.json (override with --out-dir).
"""

import argparse
import sys
from pathlib import Path

# Make the project root importable when run as a script from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.evaluate import EVAL_SPECS, evaluate_checkpoint  # noqa: E402

DEFAULT_SPECS = "psa_dev_en-sw,psa_dev_sw-en,psa_test_en-guz,psa_test_guz-en"


def _default_out_dir(checkpoint: Path) -> Path | None:
    """runs/<run>/checkpoint-best -> runs/<run>/evals; else None."""
    if checkpoint.name.startswith("checkpoint"):
        return checkpoint.parent / "evals"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint with sacreBLEU + chrF.")
    parser.add_argument("--checkpoint", required=True,
                        help="checkpoint dir or hub id "
                             "(e.g. runs/ft_nllb_base/checkpoint-best)")
    parser.add_argument("--specs", default=DEFAULT_SPECS,
                        help=f"comma-separated eval specs "
                             f"(default: {DEFAULT_SPECS}; "
                             f"available: {','.join(EVAL_SPECS)})")
    parser.add_argument("--n", type=int, default=200,
                        help="max examples per spec (default: 200)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="generation batch size (default: 16)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=None,
                        help="where to write <spec>.json results "
                             "(default: runs/<run>/evals when the checkpoint "
                             "lives in a run dir, else nowhere)")
    args = parser.parse_args(argv)

    specs = [s.strip() for s in args.specs.split(",") if s.strip()]
    unknown = [s for s in specs if s not in EVAL_SPECS]
    if unknown:
        parser.error(f"unknown specs {unknown}; "
                     f"available: {sorted(EVAL_SPECS)}")

    checkpoint = Path(args.checkpoint)
    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir(checkpoint)

    results = []
    for spec in specs:
        print(f"[run_eval] {spec} ...", flush=True)
        res = evaluate_checkpoint(checkpoint, spec, n=args.n,
                                  batch_size=args.batch_size, seed=args.seed,
                                  out_dir=out_dir)
        results.append(res)
        print(f"[run_eval] {spec}: BLEU {res['bleu']:.2f} | "
              f"chrF {res['chrf']:.2f} | n={res['n']} | "
              f"{res['seconds']:.1f}s", flush=True)

    print()
    print(f"{'eval_spec':<18} {'n':>5} {'BLEU':>8} {'chrF':>8} {'seconds':>9}")
    print("-" * 52)
    for r in results:
        print(f"{r['eval_spec']:<18} {r['n']:>5} {r['bleu']:>8.2f} "
              f"{r['chrf']:>8.2f} {r['seconds']:>9.1f}")
    if out_dir:
        print(f"\nresults written to {out_dir}/<spec>.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
