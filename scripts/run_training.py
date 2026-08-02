#!/usr/bin/env python
"""Run one Week 3 training run (SPEC_WEEK3.md §2.9).

Examples:
    python scripts/run_training.py --model mt5_small --run-name ft_mt5_base \
        --direction both --epochs 3
    python scripts/run_training.py --model nllb_600m --run-name smoke \
        --max-samples 64 --epochs 1 --report-to none --quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from training.config import MODEL_ZOO, TrainConfig  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune one PSA MT model (EN/SW <-> Ekegusii).")
    p.add_argument("--model", required=True, choices=sorted(MODEL_ZOO),
                   help="model key from MODEL_ZOO")
    p.add_argument("--run-name", required=True, help="run name -> runs/<run-name>")
    p.add_argument("--direction", default="both",
                   choices=["en-sw", "sw-en", "en-guz", "sw-guz", "both", "all"],
                   help="translation direction(s) to train on")
    p.add_argument("--fewshot-guz", type=int, default=0,
                   help="cap on PSA-sourced Ekegusii train pairs "
                        "(0 = none, -1 = all)")
    p.add_argument("--freeze-encoder", action="store_true",
                   help="freeze the encoder stack (low-resource technique)")
    p.add_argument("--freeze-embed", action="store_true",
                   help="freeze the shared embeddings")
    p.add_argument("--use-augmentation", action="store_true",
                   help="include back-translated pairs (needs --augmented-csv)")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=None,
                   help="learning rate (default: MODEL_ZOO value)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="per-device batch size (default: MODEL_ZOO value)")
    p.add_argument("--max-samples", type=int, default=None,
                   help="cap training pairs (smoke/quick runs)")
    p.add_argument("--precision", default=None,
                   choices=["fp16", "bf16", "fp32"],
                   help="GPU mixed precision (default: MODEL_ZOO value — "
                        "bf16 for mT5, fp16 for NLLB)")
    p.add_argument("--report-to", default="wandb", choices=["wandb", "none"],
                   help="W&B if usable, else JSON-only logs")
    p.add_argument("--quick", action="store_true",
                   help="quick run: caps max_samples=2000, epochs=2")
    p.add_argument("--splits-dir", default="data/processed/splits",
                   help="directory with {train,dev,test}.csv")
    p.add_argument("--augmented-csv", default=None,
                   help="CSV of back-translated pairs (default: "
                        "data/processed/augmented.csv when --use-augmentation)")
    p.add_argument("--output-root", default="runs", help="root for run dirs")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    epochs = 2.0 if args.quick else args.epochs
    max_samples = args.max_samples
    if args.quick:
        max_samples = min(max_samples, 2000) if max_samples is not None else 2000

    augmented_csv = None
    if args.use_augmentation:
        augmented_csv = Path(args.augmented_csv) if args.augmented_csv else Path(
            "data/processed/augmented.csv")
        if not augmented_csv.exists():
            print(f"[run_training] WARNING: augmented csv {augmented_csv} not found; "
                  "training without augmentation rows")
            augmented_csv = None

    cfg = TrainConfig(
        run_name=args.run_name,
        model_key=args.model,
        direction=args.direction,
        fewshot_guz=args.fewshot_guz,
        use_augmentation=args.use_augmentation,
        freeze_encoder=args.freeze_encoder,
        freeze_embed=args.freeze_embed,
        epochs=epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        max_samples=max_samples,
        precision=args.precision,
        report_to=args.report_to,
        output_root=args.output_root,
        seed=args.seed,
    )

    from training.trainer import train  # lazy: pulls torch/transformers

    best = train(cfg, splits_dir=Path(args.splits_dir),
                 augmented_csv=augmented_csv)
    print(f"[run_training] checkpoint-best: {best}")
    return best


if __name__ == "__main__":
    main()
