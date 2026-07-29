"""Back-translation augmentation (SPEC_WEEK3.md §2.3).

Takes the ENGLISH-ONLY rows of the PSA train split, translates them
eng -> swa with a fine-tuned checkpoint via ``training.inference.MTTranslator``,
and writes the synthetic pairs as a dataset-schema CSV that
``training.data.build_train_dataset`` can consume (provenance="backtranslation").

This module must import cleanly WITHOUT torch/transformers installed: the
translator (and therefore torch) is imported lazily INSIDE ``backtranslate``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Week 1 schema (frozen): exact column order of the PSA dataset CSVs.
SCHEMA_COLUMNS = [
    "PSA_ID", "Domain", "English", "Kiswahili", "Ekegusii",
    "Source", "Date", "URL", "Metadata", "Status",
]


def backtranslate(model_ckpt: str, splits_dir: Path, out_csv: Path,
                  max_rows: int = 3000, batch_size: int = 32,
                  seed: int = 42) -> Path:
    """Back-translate English-only train rows into synthetic EN-SW pairs.

    - Reads ``<splits_dir>/train.csv``; keeps rows with English != "" and
      Kiswahili == "" (English-only), in a seeded row order, capped at
      ``max_rows``.
    - Translates eng -> swa in batches via MTTranslator(model_ckpt);
      rows whose generation fails are skipped.
    - Writes ``out_csv`` with the dataset schema columns; Source="Back-
      translation", Status="Synthetic", Metadata=
      {"type": "backtranslation", "model": <ckpt>} (JSON string), and
      nothing else (Ekegusii/Date/URL left "", PSA_ID = "<orig>-BT").
    Returns out_csv.
    """
    # Lazy import: training/inference.py is built by another agent in
    # parallel (interface frozen in spec §2.7); importing it here keeps
    # torch/transformers out of module import time.
    from training.inference import MTTranslator

    splits_dir = Path(splits_dir)
    out_csv = Path(out_csv)
    train_csv = splits_dir / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"train split not found: {train_csv}")

    df = pd.read_csv(train_csv, dtype=str, encoding="utf-8").fillna("")
    en_only = df[(df["English"].str.strip() != "")
                 & (df["Kiswahili"].str.strip() == "")]
    en_only = en_only.sample(frac=1.0, random_state=seed).head(max_rows)

    translator = MTTranslator(model_ckpt)
    metadata = json.dumps({"type": "backtranslation", "model": str(model_ckpt)},
                          ensure_ascii=False)

    rows: list[dict] = []
    records = en_only.to_dict("records")
    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        texts = [r["English"] for r in chunk]
        try:
            hyps = translator.translate(texts, src="eng", tgt="swa")
        except Exception as exc:  # rows failing generation are skipped
            print(f"backtranslate: skipping batch at row {start}: {exc}")
            continue
        for rec, hyp in zip(chunk, hyps):
            if not str(hyp).strip():
                continue
            rows.append({
                "PSA_ID": f"{rec['PSA_ID']}-BT",
                "Domain": rec["Domain"],
                "English": rec["English"],
                "Kiswahili": str(hyp).strip(),
                "Ekegusii": "",
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
