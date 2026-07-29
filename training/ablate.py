"""Week 3 ablation matrix, runner and results table (SPEC_WEEK3.md §2.8).

Importable without torch/transformers: training and generation imports are
lazy inside the runner functions.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import MODEL_ZOO, TrainConfig
from .evaluate import EVAL_SPECS, evaluate_checkpoint

# Default location of the back-translation CSV produced by training.augment.
DEFAULT_AUGMENTED_CSV = Path("data/processed/augmented_backtranslation.csv")

# Headline eval specs rendered in the results table.
_DEV_SPEC = "psa_dev_en-sw"
_GUZ_SPEC = "psa_test_en-guz"


def standard_matrix(quick: bool = False) -> list[TrainConfig]:
    """The agreed ablation matrix (SPEC_WEEK3.md §2.8).

    zs_* runs are eval-only (epochs=0); the runner evaluates the base hub
    checkpoints without any training. quick=True caps max_samples=2000 and
    epochs=2 for the fine-tuning runs.
    """
    runs = [
        # 1. zero-shot baselines (no training)
        TrainConfig(run_name="zs_mt5", model_key="mt5_small", epochs=0),
        TrainConfig(run_name="zs_nllb", model_key="nllb_600m", epochs=0),
        # 2./3. base fine-tunes, both EN<->SW directions
        TrainConfig(run_name="ft_mt5_base", model_key="mt5_small",
                    direction="both"),
        TrainConfig(run_name="ft_nllb_base", model_key="nllb_600m",
                    direction="both"),
        # 4. low-resource technique: frozen encoder
        TrainConfig(run_name="ft_nllb_freeze", model_key="nllb_600m",
                    direction="both", freeze_encoder=True),
        # 5. augmentation via back-translation (runner skips if csv absent)
        TrainConfig(run_name="ft_nllb_aug", model_key="nllb_600m",
                    direction="both", use_augmentation=True),
        # 6./7. Ekegusii few-shot from the PSA train split (the only guz
        # source — FLORES-200 has no Ekegusii; NLLB-200 has no guz token)
        TrainConfig(run_name="ft_nllb_guz50", model_key="nllb_600m",
                    direction="all", fewshot_guz=50),
        TrainConfig(run_name="ft_nllb_guz200", model_key="nllb_600m",
                    direction="all", fewshot_guz=200),
        # 8. transfer comparison: same recipe on mt5
        TrainConfig(run_name="ft_mt5_guz200", model_key="mt5_small",
                    direction="all", fewshot_guz=200),
    ]
    if quick:
        quickened = []
        for cfg in runs:
            if cfg.epochs > 0:
                cfg = TrainConfig(**{**cfg.__dict__, "epochs": 2,
                                     "max_samples": 2000})
            quickened.append(cfg)
        runs = quickened
    return runs


def default_eval_specs(cfg: TrainConfig) -> list[str]:
    """Dev-set evals for every run; guz benchmark evals for guz runs.

    Zero-shot runs (epochs=0) also get guz evals on mT5 — a real "no
    Ekegusii pretraining" measurement (~0 expected). NLLB zero-shot guz is
    undefined by design: the hub tokenizer has no guz_Latn token, so there
    is no meaningful unmodified base model to evaluate.
    """
    specs = ["psa_dev_en-sw", "psa_dev_sw-en"]
    has_guz = cfg.direction == "all" or cfg.direction in ("en-guz", "sw-guz")
    zs_mt5 = cfg.epochs == 0 and MODEL_ZOO[cfg.model_key].family == "mt5"
    if has_guz or zs_mt5:
        specs += ["psa_test_en-guz", "psa_test_guz-en"]
    return specs


def _run_dir(cfg: TrainConfig) -> Path:
    path = Path(cfg.output_root) / cfg.run_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_config(cfg: TrainConfig, augmented_csv: Path | None = None,
               eval_specs: list[str] | None = None, eval_n: int | None = 200,
               batch_size: int = 16, skip_existing: bool = True
               ) -> Path | None:
    """Run one matrix entry: train (unless epochs=0) then evaluate.

    Returns the run dir, or None when the run is skipped (missing
    augmentation csv). Zero-shot entries (epochs=0) evaluate the base hub
    checkpoint directly. Results land in runs/<run_name>/evals/*.json.
    """
    cfg = cfg.resolved()
    run = _run_dir(cfg)
    specs = eval_specs or default_eval_specs(cfg)
    evals_dir = run / "evals"

    if skip_existing and all((evals_dir / f"{s}.json").is_file() for s in specs):
        print(f"[ablate] {cfg.run_name}: evals already present, skipping "
              f"(skip_existing=True)")
        return run

    if cfg.epochs == 0:  # zero-shot: eval-only, no training
        ckpt: str | Path = MODEL_ZOO[cfg.model_key].hf_name
        print(f"[ablate] {cfg.run_name}: zero-shot eval of {ckpt}")
    else:
        aug_path = augmented_csv or DEFAULT_AUGMENTED_CSV
        if cfg.use_augmentation and not Path(aug_path).is_file():
            print(f"[ablate] {cfg.run_name}: SKIPPED — augmentation csv not "
                  f"found at {aug_path} (run the back-translation step first)")
            return None
        from .trainer import train  # noqa: lazy import (torch inside)

        ckpt = train(cfg, augmented_csv=aug_path if cfg.use_augmentation
                     else None)

    cfg.to_json(run / "train_config.json")
    for spec in specs:
        print(f"[ablate] {cfg.run_name}: evaluating {spec} ...")
        evaluate_checkpoint(ckpt, spec, n=eval_n, batch_size=batch_size,
                            seed=cfg.seed, out_dir=evals_dir)
    return run


def run_matrix(configs: list[TrainConfig], **kwargs) -> list[Path]:
    """Run a list of matrix entries sequentially; returns run dirs."""
    done: list[Path] = []
    for cfg in configs:
        out = run_config(cfg, **kwargs)
        if out is not None:
            done.append(out)
    return done


# --------------------------------------------------------------------------
# results table
# --------------------------------------------------------------------------

def _get(d: dict, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _fmt(value, nd: int = 2) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{nd}f}"
    except (TypeError, ValueError):
        return str(value)


def _flags(cfg: dict | None) -> str:
    if not cfg:
        return "—"
    if float(cfg.get("epochs", 1) or 0) == 0:
        return "zero-shot"
    parts = [str(cfg.get("direction", "—"))]
    if cfg.get("fewshot_guz"):
        parts.append(f"guz={cfg['fewshot_guz']}")
    if cfg.get("freeze_encoder"):
        parts.append("freeze-enc")
    if cfg.get("freeze_embed"):
        parts.append("freeze-emb")
    if cfg.get("use_augmentation"):
        parts.append("aug")
    if cfg.get("max_samples"):
        parts.append(f"cap={cfg['max_samples']}")
    return ", ".join(parts)


def _collect_run(run: Path) -> dict | None:
    if not run.is_dir():
        return None
    cfg = None
    cfg_path = run / "train_config.json"
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cfg = None
    metrics = {}
    m_path = run / "metrics_dev.json"
    if m_path.is_file():
        try:
            metrics = json.loads(m_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metrics = {}
    evals: dict[str, dict] = {}
    evals_dir = run / "evals"
    if evals_dir.is_dir():
        for j in sorted(evals_dir.glob("*.json")):
            try:
                evals[j.stem] = json.loads(j.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
    if not cfg and not metrics and not evals:
        return None
    return {"name": run.name, "cfg": cfg, "metrics": metrics, "evals": evals}


def write_results_table(runs_root: Path, out_md: Path) -> Path:
    """Scan runs/*/metrics_dev.json + runs/*/evals/*.json into a markdown
    table at out_md (missing values rendered "—")."""
    runs_root = Path(runs_root)
    rows = []
    if runs_root.is_dir():
        for child in sorted(runs_root.iterdir()):
            rec = _collect_run(child)
            if rec is not None:
                rows.append(rec)

    order = {cfg.run_name: i for i, cfg in enumerate(standard_matrix())}
    rows.sort(key=lambda r: (order.get(r["name"], len(order)), r["name"]))

    header = ("| Run | Model | Config | Dev BLEU | Dev chrF | "
              "Guz BLEU | Guz chrF | Trainable % | Seconds |")
    sep = "|---|---|---|---|---|---|---|---|---|"
    lines = [
        "# Week 3 ablation results",
        "",
        f"Scanned from `{runs_root}` "
        "(metrics_dev.json + evals/*.json per run).",
        "",
        header,
        sep,
    ]
    for rec in rows:
        cfg, metrics, evals = rec["cfg"], rec["metrics"], rec["evals"]
        model = (cfg or {}).get("model_key") or metrics.get("model_key")
        dev = evals.get(_DEV_SPEC, {})
        dev_bleu = _get(dev, "bleu") or _get(
            metrics, "sacrebleu", "eval_sacrebleu", "bleu", "eval_bleu")
        dev_chrf = _get(dev, "chrf") or _get(
            metrics, "chrf", "eval_chrf", "chrF")
        guz = evals.get(_GUZ_SPEC, {})
        trainable = _get(metrics, "trainable_pct", "trainable_percent")
        seconds = _get(metrics, "seconds", "train_seconds", "train_runtime")
        lines.append(
            f"| {rec['name']} | {model or '—'} | {_flags(cfg)} "
            f"| {_fmt(dev_bleu)} | {_fmt(dev_chrf)} "
            f"| {_fmt(_get(guz, 'bleu'))} | {_fmt(_get(guz, 'chrf'))} "
            f"| {_fmt(trainable, 1)} | {_fmt(seconds, 0)} |")

    out_md = Path(out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_md
