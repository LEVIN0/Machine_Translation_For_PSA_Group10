"""Training engine — HF Seq2SeqTrainer wiring (SPEC_WEEK3.md §2.5, §3).

Heavy dependencies (torch/transformers/datasets/sacrebleu) are imported lazily
inside functions so this module is importable without the GPU stack installed.
`training.data` is imported lazily inside train() against the frozen §2.2
interface (built in parallel by another agent).
"""

from __future__ import annotations

import os
from pathlib import Path

from .config import LANGS, MODEL_ZOO, NLLB_CODES, TrainConfig
from .utils import Timer, device_info, run_dir, save_json, set_seed


def _resolve_report_to(cfg: TrainConfig) -> str:
    """W&B only when requested AND usable; never a hard dependency (§4)."""
    if cfg.report_to != "wandb":
        return "none"
    try:
        import wandb  # noqa: F401
    except Exception:
        print("[trainer] wandb not installed; falling back to report_to='none'")
        return "none"
    if os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_MODE") == "offline":
        os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)
        return "wandb"
    print("[trainer] WANDB_API_KEY not set; falling back to report_to='none'")
    return "none"


def _apply_freezing(model, cfg: TrainConfig) -> float:
    """Apply freeze_encoder / freeze_embed; return trainable-param percentage."""
    if cfg.freeze_encoder:
        for p in model.get_encoder().parameters():
            p.requires_grad_(False)
    if cfg.freeze_embed:
        embed = model.get_shared() if hasattr(model, "get_shared") else model.get_input_embeddings()
        for p in embed.parameters():
            p.requires_grad_(False)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / total if total else 0.0
    print(f"[trainer] trainable params: {trainable:,}/{total:,} ({pct:.2f}%)")
    try:
        import wandb

        if getattr(wandb, "run", None) is not None:
            wandb.summary["trainable_pct"] = pct
    except Exception:
        pass
    return pct


def _forced_bos_id(tokenizer, lang_code: str) -> int | None:
    """NLLB forced BOS id; convert_tokens_to_ids with lang_code_to_id fallback (§3)."""
    try:
        tid = tokenizer.convert_tokens_to_ids(lang_code)
        if isinstance(tid, int) and tid >= 0:
            return tid
    except Exception:
        pass
    code_map = getattr(tokenizer, "lang_code_to_id", None)
    if code_map is not None:
        try:
            return int(code_map[lang_code])
        except Exception:
            return None
    return None


def _tokenize_dataset(ds, tokenizer, family: str, max_length: int):
    """Tokenize per §3. Returns a datasets.Dataset with input_ids/labels."""
    remove_cols = list(ds.column_names)

    if family == "mt5":
        def _mt5(batch):
            texts = [
                f"translate {LANGS[s]} to {LANGS[t]}: {x}"
                for s, t, x in zip(batch["src_lang"], batch["tgt_lang"], batch["src_text"])
            ]
            return tokenizer(
                text=texts, text_target=batch["tgt_text"],
                max_length=max_length, truncation=True,
            )

        return ds.map(_mt5, batched=True, remove_columns=remove_cols)

    if family == "nllb":
        # src_lang/tgt_lang are tokenizer state -> tokenize per language-pair group.
        pairs = sorted(set(zip(ds["src_lang"], ds["tgt_lang"])))
        parts = []
        for s, t in pairs:
            idx = [i for i, (a, b) in enumerate(zip(ds["src_lang"], ds["tgt_lang"]))
                   if (a, b) == (s, t)]
            sub = ds.select(idx)
            tokenizer.src_lang = NLLB_CODES[s]

            def _nllb(batch):
                return tokenizer(
                    text=batch["src_text"], text_target=batch["tgt_text"],
                    max_length=max_length, truncation=True,
                )

            if hasattr(tokenizer, "tgt_lang"):
                tokenizer.tgt_lang = NLLB_CODES[t]
                parts.append(sub.map(_nllb, batched=True, remove_columns=remove_cols))
            else:  # older transformers fallback
                with tokenizer.as_target_tokenizer():
                    parts.append(sub.map(_nllb, batched=True, remove_columns=remove_cols))
        if len(parts) == 1:
            return parts[0]
        from datasets import concatenate_datasets

        return concatenate_datasets(parts).shuffle(seed=42)

    raise ValueError(f"unknown model family '{family}'")


def _make_compute_metrics(tokenizer):
    def compute_metrics(eval_pred):
        import numpy as np
        import sacrebleu

        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        preds = np.where(np.asarray(preds) < 0, pad_id, preds)
        labels = np.where(np.asarray(labels) != -100, labels, pad_id)
        decoded_preds = [p.strip() for p in tokenizer.batch_decode(preds, skip_special_tokens=True)]
        decoded_labels = [l.strip() for l in tokenizer.batch_decode(labels, skip_special_tokens=True)]
        bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
        chrf = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels])
        return {"sacrebleu": float(bleu.score), "chrf": float(chrf.score)}

    return compute_metrics


def train(cfg: TrainConfig,
          splits_dir: Path = Path("data/processed/splits"),
          flores_dir: Path = Path("data/external/flores200"),
          augmented_csv: Path | None = None) -> Path:
    """Run one training run; returns Path to run_dir/checkpoint-best (§2.5)."""
    import torch
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                              DataCollatorForSeq2Seq, Seq2SeqTrainer,
                              Seq2SeqTrainingArguments)

    from . import data  # lazy: frozen §2.2 interface, built in parallel

    cfg = cfg.resolved()
    set_seed(cfg.seed)
    splits_dir, flores_dir = Path(splits_dir), Path(flores_dir)
    out = run_dir(cfg)

    train_ds = data.build_train_dataset(cfg, splits_dir, flores_dir, augmented_csv)
    n_train_pairs = len(train_ds)
    try:
        eval_ds = data.load_psa_pairs(splits_dir, "dev", cfg.directions())
        if len(eval_ds) == 0:
            raise ValueError("empty dev split")
    except Exception as e:
        print(f"[trainer] dev split unavailable ({e}); evaluating on train subset")
        eval_ds = train_ds

    model_cfg = MODEL_ZOO[cfg.model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.hf_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_cfg.hf_name)

    trainable_pct = _apply_freezing(model, cfg)

    train_tok = _tokenize_dataset(train_ds, tokenizer, model_cfg.family, model_cfg.max_length)
    eval_tok = _tokenize_dataset(eval_ds, tokenizer, model_cfg.family, model_cfg.max_length)

    if model_cfg.family == "nllb":
        tgts = sorted(set(train_ds["tgt_lang"]))
        if len(tgts) == 1:
            bos = _forced_bos_id(tokenizer, NLLB_CODES[tgts[0]])
            if bos is not None:
                if getattr(model, "generation_config", None) is not None:
                    model.generation_config.forced_bos_token_id = bos
                else:
                    model.config.forced_bos_token_id = bos
        else:
            print(f"[trainer] multiple target langs {tgts}; no forced_bos_token_id set")

    report_to = _resolve_report_to(cfg)

    arg_kwargs = dict(
        output_dir=str(out),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="sacrebleu",
        greater_is_better=True,
        predict_with_generate=True,
        fp16=bool(cfg.fp16 and torch.cuda.is_available()),
        learning_rate=cfg.lr,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        report_to=report_to,
        run_name=cfg.run_name,
        seed=cfg.seed,
        logging_steps=20,
        save_total_limit=2,
    )
    # transformers-version compatibility (eval_strategy vs evaluation_strategy, run_name)
    import inspect

    params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    if "eval_strategy" not in params and "evaluation_strategy" in params:
        arg_kwargs["evaluation_strategy"] = arg_kwargs.pop("eval_strategy")
    arg_kwargs = {k: v for k, v in arg_kwargs.items() if k in params}
    args = Seq2SeqTrainingArguments(**arg_kwargs)

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=collator,
        compute_metrics=_make_compute_metrics(tokenizer),
    )
    # transformers >=5 renamed Trainer's `tokenizer` arg to `processing_class`
    if "processing_class" in inspect.signature(Seq2SeqTrainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Seq2SeqTrainer(**trainer_kwargs)

    with Timer() as timer:
        trainer.train()

    best_dir = out / "checkpoint-best"
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_dir))  # best weights (load_best_model_at_end=True)
    tokenizer.save_pretrained(str(best_dir))

    cfg.to_json(out / "train_config.json")
    try:
        eval_metrics = {k: float(v) for k, v in trainer.evaluate().items()
                        if isinstance(v, (int, float))}
    except Exception as e:
        print(f"[trainer] final evaluate failed ({e}); writing metrics without eval")
        eval_metrics = {}
    metrics = dict(eval_metrics)
    metrics.update({
        "n_train_pairs": n_train_pairs,
        "trainable_pct": trainable_pct,
        "seconds": timer.seconds,
        "device_info": device_info(),
    })
    save_json(metrics, out / "metrics_dev.json")

    try:
        import wandb

        if getattr(wandb, "run", None) is not None:
            wandb.summary["metrics_dev"] = metrics
            wandb.finish()
    except Exception:
        pass

    print(f"[trainer] done in {timer.seconds:.1f}s -> {best_dir}")
    return best_dir
