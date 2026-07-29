"""Week 3 data builders — PSA splits + FLORES-200 seed/eval (SPEC_WEEK3.md §2.2).

Canonical example columns (HF ``datasets.Dataset``):
    src_text, tgt_text, src_lang, tgt_lang, domain, provenance

``src_lang``/``tgt_lang`` in {"eng", "swa", "guz"}; ``provenance`` in
{"psa", "flores_dev_seed", "backtranslation"}.

Non-negotiables (spec §4):
- FLORES **devtest never in training** — dev seed only; build_train_dataset
  raises if asked for few-shot guz pairs while only devtest is on disk AND
  no PSA-sourced guz pairs are available.
- Ekegusii now comes primarily from real PSA pairs (lecturer gold data,
  remediated dataset) — see SPEC_WEEK3.md addendum. FLORES dev remains an
  OPTIONAL extra seed source; FLORES devtest stays a pure benchmark.
- Seeded everything (seed=42 default).
- No torch/transformers import at module import time; ``datasets`` itself is
  imported lazily inside functions so this module imports cleanly on a bare
  CPU box.
- pathlib everywhere, UTF-8 everywhere, Windows-compatible.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from training.config import TrainConfig

CANONICAL_COLUMNS = [
    "src_text", "tgt_text", "src_lang", "tgt_lang", "domain", "provenance",
]

# PSA split CSV column per language code (Week 1 schema is the source of truth).
PSA_TEXT_COLUMNS = {"eng": "English", "swa": "Kiswahili", "guz": "Ekegusii"}

PSA_DIRECTIONS = ("en-sw", "sw-en")        # available from the PSA splits
GUZ_DIRECTIONS = ("en-guz", "sw-guz")      # PSA rows with Ekegusii + FLORES seed

PROVENANCE_PSA = "psa"
PROVENANCE_FLORES_SEED = "flores_dev_seed"
PROVENANCE_BACKTRANSLATION = "backtranslation"

_FLORES_DOMAIN = "general"

DEFAULT_SEED_DIRS: list[str] = ["en-guz", "sw-guz"]


# ---------------------------------------------------------------------------
# Direction expansion (single place — spec §2.1 footnote)
# ---------------------------------------------------------------------------

def expand_directions(direction: str) -> list[str]:
    """Expand direction aliases to explicit direction codes.

    "both" -> ["en-sw", "sw-en"]; "all" -> ["en-sw", "sw-en", "en-guz",
    "sw-guz"]; single codes stay as-is. Unknown codes raise ValueError.
    """
    if direction == "both":
        return ["en-sw", "sw-en"]
    if direction == "all":
        return ["en-sw", "sw-en", "en-guz", "sw-guz"]
    if direction in PSA_DIRECTIONS + GUZ_DIRECTIONS:
        return [direction]
    raise ValueError(f"unknown direction '{direction}'")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_datasets():
    """Import HF datasets lazily; raise a helpful error if it is missing."""
    try:
        from datasets import Dataset, concatenate_datasets
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "training.data needs the 'datasets' package: pip install datasets"
        ) from exc
    return Dataset, concatenate_datasets


def _dataset_from_rows(rows: list[dict]):
    """Build a Dataset from canonical rows (empty -> empty-typed Dataset)."""
    Dataset, _ = _require_datasets()
    return Dataset.from_dict({c: [r[c] for r in rows] for c in CANONICAL_COLUMNS})


def _read_tsv(path: Path) -> pd.DataFrame:
    """Read a canonical FLORES TSV (header ``eng\\tguz``), UTF-8."""
    return pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8").fillna("")


def _flores_seed_rows(df: pd.DataFrame, directions: list[str]) -> list[dict]:
    """Expand FLORES pairs into canonical rows for the requested guz dirs."""
    has_sw = "swa" in df.columns
    rows: list[dict] = []
    for direction in directions:
        if direction == "en-guz":
            for _, r in df.iterrows():
                rows.append({
                    "src_text": r["eng"], "tgt_text": r["guz"],
                    "src_lang": "eng", "tgt_lang": "guz",
                    "domain": _FLORES_DOMAIN, "provenance": PROVENANCE_FLORES_SEED,
                })
        elif direction == "sw-guz":
            if not has_sw:
                warnings.warn(
                    "load_flores_seed: no 'swa' column in guz_dev.tsv; "
                    "skipping 'sw-guz' rows", stacklevel=2)
                continue
            for _, r in df.iterrows():
                rows.append({
                    "src_text": r["swa"], "tgt_text": r["guz"],
                    "src_lang": "swa", "tgt_lang": "guz",
                    "domain": _FLORES_DOMAIN, "provenance": PROVENANCE_FLORES_SEED,
                })
        else:
            raise ValueError(
                f"load_flores_seed only supports guz directions, got '{direction}'")
    return rows


# ---------------------------------------------------------------------------
# Public loaders (spec §2.2)
# ---------------------------------------------------------------------------

def load_psa_pairs(splits_dir: Path, split: str, directions: list[str]):
    """Load PSA pairs from ``<splits_dir>/<split>.csv`` as a canonical Dataset.

    For en-sw / sw-en: keeps rows with Kiswahili != "" (and English != "").
    For guz directions (SPEC_REMEDIATION §4): additionally emits pairs from
    rows with non-empty ``Ekegusii`` — en-guz (English->Ekegusii) for every
    such row, sw-guz (Kiswahili->Ekegusii) only when Kiswahili is non-empty.
    Domain is carried; provenance="psa" for all directions.
    """
    splits_dir = Path(splits_dir)
    csv_path = splits_dir / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"split file not found: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str, encoding="utf-8").fillna("")
    paired = df[(df["Kiswahili"].str.strip() != "")
                & (df["English"].str.strip() != "")]
    guz_rows = df[(df["Ekegusii"].str.strip() != "")
                  & (df["English"].str.strip() != "")]
    rows: list[dict] = []
    for direction in directions:
        if direction == "en-sw":
            src_col, tgt_col, src_lang, tgt_lang = ("English", "Kiswahili", "eng", "swa")
            for _, r in paired.iterrows():
                rows.append({
                    "src_text": r[src_col], "tgt_text": r[tgt_col],
                    "src_lang": src_lang, "tgt_lang": tgt_lang,
                    "domain": r["Domain"], "provenance": PROVENANCE_PSA,
                })
        elif direction == "sw-en":
            for _, r in paired.iterrows():
                rows.append({
                    "src_text": r["Kiswahili"], "tgt_text": r["English"],
                    "src_lang": "swa", "tgt_lang": "eng",
                    "domain": r["Domain"], "provenance": PROVENANCE_PSA,
                })
        elif direction == "en-guz":
            for _, r in guz_rows.iterrows():
                rows.append({
                    "src_text": r["English"], "tgt_text": r["Ekegusii"],
                    "src_lang": "eng", "tgt_lang": "guz",
                    "domain": r["Domain"], "provenance": PROVENANCE_PSA,
                })
        elif direction == "sw-guz":
            for _, r in guz_rows.iterrows():
                if r["Kiswahili"].strip() == "":
                    continue  # sw-guz only when Kiswahili is non-empty
                rows.append({
                    "src_text": r["Kiswahili"], "tgt_text": r["Ekegusii"],
                    "src_lang": "swa", "tgt_lang": "guz",
                    "domain": r["Domain"], "provenance": PROVENANCE_PSA,
                })
        else:
            raise ValueError(f"unknown direction '{direction}'")
    return _dataset_from_rows(rows)


def load_flores_seed(flores_dir: Path, n: int, seed: int = 42,
                     directions: list[str] | None = None):
    """Load n few-shot guz seed pairs from FLORES **dev** (never devtest).

    Reads ``<flores_dir>/guz_dev.tsv`` (header ``eng<TAB>guz``), shuffles
    with ``seed``, takes the FIRST n. Emits en-guz rows; sw-guz rows are
    emitted only if a ``swa`` column exists (the plain fixture has none ->
    skipped with a warning, no crash). provenance="flores_dev_seed".
    """
    directions = list(directions) if directions is not None else list(DEFAULT_SEED_DIRS)
    dev_path = Path(flores_dir) / "guz_dev.tsv"
    if not dev_path.exists():
        raise FileNotFoundError(
            f"FLORES dev seed file not found: {dev_path}. guz_devtest.tsv may "
            "never be used for training — run scripts/fetch_flores.py first.")
    df = _read_tsv(dev_path)
    df = df.sample(frac=1.0, random_state=seed).head(n)
    return _dataset_from_rows(_flores_seed_rows(df, directions))


def load_flores_eval(flores_dir: Path, n: int | None = 200, seed: int = 42):
    """Load FLORES **devtest** pairs for evaluation ONLY (never training).

    Reads ``<flores_dir>/guz_devtest.tsv``; deterministic seeded sample of n
    (n=None -> all rows, in file order). Returns raw ``eng``/``guz`` columns;
    evaluate.py normalizes them to src/tgt per eval spec.
    """
    Dataset, _ = _require_datasets()
    devtest_path = Path(flores_dir) / "guz_devtest.tsv"
    if not devtest_path.exists():
        raise FileNotFoundError(
            f"FLORES devtest file not found: {devtest_path} — run "
            "scripts/fetch_flores.py first.")
    df = _read_tsv(devtest_path)
    if n is not None:
        df = df.sample(frac=1.0, random_state=seed).head(n)
    return Dataset.from_dict({"eng": df["eng"].tolist(),
                              "guz": df["guz"].tolist()})


def build_train_dataset(cfg: TrainConfig, splits_dir: Path, flores_dir: Path,
                        augmented_csv: Path | None = None):
    """Build the full training Dataset for ``cfg`` (spec §2.2).

    Concatenates, in order:
      1. PSA pairs from the train split (en-sw / sw-en as requested);
      2. PSA-sourced guz pairs (rows with non-empty Ekegusii), gated by
         ``cfg.fewshot_guz``: 0 = exclude, N = seeded cap, -1 = all;
      3. FLORES dev seed pairs — OPTIONAL extra, only if ``fewshot_guz > 0``
         AND ``guz_dev.tsv`` is present (appended after the PSA guz pairs);
      4. augmented_csv rows (provenance="backtranslation").

    Then applies a seeded shuffle and the cfg.max_samples cap (AFTER
    concatenation). HARD GUARD: if cfg.fewshot_guz > 0 but NO guz pairs are
    available at all (no Ekegusii in the train split and guz_dev.tsv
    absent, e.g. only devtest on disk), raise — devtest must never train.
    """
    _, concatenate_datasets = _require_datasets()
    directions = cfg.directions()
    parts = []

    psa_dirs = [d for d in directions if d in PSA_DIRECTIONS]
    if psa_dirs:
        parts.append(load_psa_pairs(splits_dir, "train", psa_dirs))

    if cfg.fewshot_guz != 0:
        guz_dirs = [d for d in directions if d in GUZ_DIRECTIONS] or list(DEFAULT_SEED_DIRS)
        psa_guz = load_psa_pairs(splits_dir, "train", guz_dirs)
        if cfg.fewshot_guz > 0 and len(psa_guz) > cfg.fewshot_guz:
            psa_guz = psa_guz.shuffle(seed=cfg.seed).select(range(cfg.fewshot_guz))
        if len(psa_guz):
            parts.append(psa_guz)
        if cfg.fewshot_guz > 0:
            flores_dir = Path(flores_dir)
            dev_path = flores_dir / "guz_dev.tsv"
            if dev_path.exists():
                parts.append(load_flores_seed(flores_dir, cfg.fewshot_guz,
                                              seed=cfg.seed, directions=guz_dirs))
            elif len(psa_guz) == 0:
                raise FileNotFoundError(
                    f"fewshot_guz={cfg.fewshot_guz} requested but no guz "
                    f"training pairs are available: the train split has no "
                    f"Ekegusii text and {dev_path} is missing. FLORES "
                    "guz_devtest.tsv is evaluation-only and may never be "
                    "used for training; run scripts/fetch_flores.py to "
                    "fetch guz_dev.tsv.")

    if augmented_csv is not None:
        parts.append(_load_augmented(augmented_csv,
                                     [d for d in directions if d in PSA_DIRECTIONS]
                                     or list(PSA_DIRECTIONS)))

    if not parts:
        raise ValueError(
            f"nothing to train on for direction '{cfg.direction}' "
            f"(fewshot_guz={cfg.fewshot_guz}, augmented_csv={augmented_csv})")
    ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    ds = ds.shuffle(seed=cfg.seed)
    if cfg.max_samples is not None:
        ds = ds.select(range(min(cfg.max_samples, len(ds))))
    return ds


def _load_augmented(augmented_csv: Path, directions: list[str]):
    """Load back-translated rows (dataset-schema CSV) as canonical pairs."""
    augmented_csv = Path(augmented_csv)
    if not augmented_csv.exists():
        raise FileNotFoundError(f"augmented csv not found: {augmented_csv}")
    df = pd.read_csv(augmented_csv, dtype=str, encoding="utf-8").fillna("")
    df = df[(df["Kiswahili"].str.strip() != "") & (df["English"].str.strip() != "")]
    rows: list[dict] = []
    for direction in directions:
        if direction == "en-sw":
            src_col, tgt_col, src_lang, tgt_lang = ("English", "Kiswahili", "eng", "swa")
        elif direction == "sw-en":
            src_col, tgt_col, src_lang, tgt_lang = ("Kiswahili", "English", "swa", "eng")
        else:
            continue
        for _, r in df.iterrows():
            rows.append({
                "src_text": r[src_col], "tgt_text": r[tgt_col],
                "src_lang": src_lang, "tgt_lang": tgt_lang,
                "domain": r["Domain"], "provenance": PROVENANCE_BACKTRANSLATION,
            })
    return _dataset_from_rows(rows)
