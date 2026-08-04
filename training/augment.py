"""Back-translation augmentation (SPEC_WEEK3.md §2.3).

Takes rows of the PSA train split that LACK a translation in the target
language, translates English -> target with a fine-tuned checkpoint via
``training.inference.MTTranslator``, and writes the synthetic pairs as a
dataset-schema CSV that ``training.data.build_train_dataset`` can consume
(provenance="backtranslation").

Two uses:
  - tgt="swa" (Week 3 original): English-only rows -> synthetic EN-SW pairs.
  - tgt="guz" (Week 4): rows lacking Ekegusii (English-only AND EN-SW rows)
    -> synthetic en-guz pairs; existing Kiswahili text is carried over
    unchanged, so rows that had it also yield sw-guz pairs. This is the
    Ekegusii scaling-curve experiment of reports/week3_report.md taken one
    step further (2.5k real -> ~6k real+synthetic guz pairs).

Guardrails for tgt="guz" (the model's known failure modes, see
reports/week4_report.md §2/§3): decoding uses no_repeat_ngram_size=3 to
suppress repetition loops in the synthetic data itself, and copy-through
outputs (hypothesis identical to the English source) are dropped.

This module must import cleanly WITHOUT torch/transformers installed: the
translator (and therefore torch) is imported lazily INSIDE ``backtranslate``.

CLI:
    python training/augment.py --checkpoint runs/ft_nllb_guz_all/checkpoint-best \
        --tgt guz --out data/processed/augmented_guz.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# Week 1 schema (frozen): exact column order of the PSA dataset CSVs.
SCHEMA_COLUMNS = [
    "PSA_ID", "Domain", "English", "Kiswahili", "Ekegusii",
    "Source", "Date", "URL", "Metadata", "Status",
]

# target code -> (dataset column it fills, PSA_ID suffix)
TARGETS = {"swa": ("Kiswahili", "-BT"), "guz": ("Ekegusii", "-BTG")}

# decoding guardrail for synthetic generation (see module docstring)
NO_REPEAT_NGRAM = 3


def select_backtranslation_rows(df: pd.DataFrame, tgt: str,
                                seed: int = 42,
                                max_rows: int = 3000) -> pd.DataFrame:
    """Train-split rows with English text but no ``tgt`` translation yet.

    Pure function (no model) so it is unit-testable without torch.
    ``max_rows <= 0`` means no cap.
    """
    tgt_col, _ = TARGETS[tgt]
    df = df.fillna("")
    mask = (df["English"].str.strip() != "") & (df[tgt_col].str.strip() == "")
    selected = df[mask].sample(frac=1.0, random_state=seed)
    if max_rows and max_rows > 0:
        selected = selected.head(max_rows)
    return selected


def _usable(hyp: str, source: str) -> bool:
    """Reject empty and copy-through generations (Week 4 failure modes)."""
    norm = lambda s: str(s).strip().lower().rstrip(".!?")  # noqa: E731
    hyp = str(hyp).strip()
    if not hyp:
        return False
    return norm(hyp) != norm(source)


def backtranslate(model_ckpt: str, splits_dir: Path, out_csv: Path,
                  tgt: str = "swa", max_rows: int = 3000, batch_size: int = 32,
                  seed: int = 42) -> Path:
    """Back-translate train rows lacking ``tgt`` into synthetic pairs.

    - Reads ``<splits_dir>/train.csv``; keeps rows with English != "" and the
      target column == "" (seeded order, capped at ``max_rows``).
    - Translates eng -> ``tgt`` in batches via MTTranslator(model_ckpt)
      with no_repeat_ngram_size=3; empty/copy-through generations dropped.
    - Writes ``out_csv`` in the dataset schema: synthetic text in the target
      column, Source="Back-translation", Status="Synthetic", Metadata=
      {"type": "backtranslation", "model": <ckpt>, "tgt": <tgt>}, PSA_ID
      suffixed per TARGETS. Any pre-existing text in the OTHER language
      columns is carried over unchanged (so an EN-SW row back-translated
      to guz also yields a real-SW/synthetic-guz sw-guz pair).
    Returns out_csv.
    """
    # Lazy import: keeps torch/transformers out of module import time.
    from training.inference import MTTranslator

    if tgt not in TARGETS:
        raise ValueError(f"tgt must be one of {sorted(TARGETS)}, got {tgt!r}")
    tgt_col, suffix = TARGETS[tgt]

    splits_dir = Path(splits_dir)
    out_csv = Path(out_csv)
    train_csv = splits_dir / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"train split not found: {train_csv}")

    df = pd.read_csv(train_csv, dtype=str, encoding="utf-8")
    selected = select_backtranslation_rows(df, tgt, seed=seed, max_rows=max_rows)
    print(f"backtranslate: {len(selected)} train rows lack {tgt_col}; "
          f"generating eng->{tgt} with {model_ckpt}")

    translator = MTTranslator(model_ckpt)
    metadata = json.dumps({"type": "backtranslation", "model": str(model_ckpt),
                           "tgt": tgt}, ensure_ascii=False)

    rows: list[dict] = []
    records = selected.to_dict("records")
    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        texts = [r["English"] for r in chunk]
        try:
            hyps = translator.translate(texts, src="eng", tgt=tgt,
                                        no_repeat_ngram_size=NO_REPEAT_NGRAM)
        except Exception as exc:  # rows failing generation are skipped
            print(f"backtranslate: skipping batch at row {start}: {exc}")
            continue
        for rec, hyp in zip(chunk, hyps):
            if not _usable(hyp, rec["English"]):
                continue
            rows.append({
                "PSA_ID": f"{rec['PSA_ID']}{suffix}",
                "Domain": rec["Domain"],
                "English": rec["English"],
                "Kiswahili": rec.get("Kiswahili", "") or "",
                "Ekegusii": rec.get("Ekegusii", "") or "",
                tgt_col: str(hyp).strip(),
                "Source": "Back-translation",
                "Date": "",
                "URL": "",
                "Metadata": metadata,
                "Status": "Synthetic",
            })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=SCHEMA_COLUMNS).to_csv(
        out_csv, index=False, encoding="utf-8")
    print(f"backtranslate: wrote {len(rows)} synthetic pairs -> {out_csv}")
    return out_csv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Back-translation augmentation.")
    parser.add_argument("--checkpoint", required=True,
                        help="fine-tuned checkpoint used to generate the "
                             "synthetic translations")
    parser.add_argument("--splits-dir", default="data/processed/splits")
    parser.add_argument("--out", default=None,
                        help="output CSV (default: data/processed/augmented_<tgt>.csv)")
    parser.add_argument("--tgt", default="swa", choices=sorted(TARGETS),
                        help="target language to synthesize (default: swa)")
    parser.add_argument("--max-rows", type=int, default=0,
                        help="cap on rows to back-translate (default: 0 = all)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    out = Path(args.out) if args.out else Path(
        f"data/processed/augmented_{args.tgt}.csv")
    backtranslate(args.checkpoint, Path(args.splits_dir), out, tgt=args.tgt,
                  max_rows=args.max_rows, batch_size=args.batch_size,
                  seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
