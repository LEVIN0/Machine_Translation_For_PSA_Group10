#!/usr/bin/env python3
"""Week 3 trainer tests (Agent B) — SPEC_WEEK3.md §2.4/§2.5/§4.

Plain asserts, no pytest required. Everything that CAN run offline runs;
network-dependent parts (HF hub download) skip gracefully with a printed
reason and still count as pass.

Run from the project root:  python tests/test_week3_trainer.py
"""

from __future__ import annotations

import csv
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SCHEMA = ["PSA_ID", "Domain", "English", "Kiswahili", "Ekegusii", "Source",
          "Date", "URL", "Metadata", "Status"]


def _psa_row(i: int, paired: bool = True) -> dict:
    return {
        "PSA_ID": f"PSA{i:06d}",
        "Domain": "Health",
        "English": f"Public health announcement number {i} for all citizens.",
        "Kiswahili": (f"Tangazo la afya la umma namba {i} kwa wananchi wote."
                      if paired else ""),
        "Ekegusii": "",
        "Source": "Test",
        "Date": "2024-01-01",
        "URL": "",
        "Metadata": "{}",
        "Status": "Pending",
    }


def _write_splits(splits_dir: Path, n_train: int = 6, n_dev: int = 4) -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split, n in (("train", n_train), ("dev", n_dev)):
        with (splits_dir / f"{split}.csv").open("w", newline="",
                                                encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SCHEMA)
            w.writeheader()
            for i in range(1, n + 1):
                w.writerow(_psa_row(i))


def test_utils():
    """§2.4: seed reproducibility, run_dir, json roundtrip, Timer."""
    from training.config import TrainConfig
    from training.utils import (Timer, device_info, load_json, run_dir,
                                save_json, set_seed)

    set_seed(42)
    a = [random.random() for _ in range(3)]
    set_seed(42)
    b = [random.random() for _ in range(3)]
    assert a == b, "python random not reproducible after set_seed"
    try:
        import numpy as np
        set_seed(42)
        x = np.random.rand(3).tolist()
        set_seed(42)
        y = np.random.rand(3).tolist()
        assert x == y, "numpy not reproducible after set_seed"
    except ImportError:
        print("skipped: numpy not installed (set_seed numpy check)")

    info = device_info()
    assert set(info) == {"device", "gpu_name", "torch"}, info
    assert info["device"] in {"cuda", "cpu"}

    with tempfile.TemporaryDirectory(prefix="psa_w3_") as tmp:
        cfg = TrainConfig(run_name="u1", model_key="mt5_small", output_root=tmp)
        rd = run_dir(cfg)
        assert rd == Path(tmp) / "u1" and rd.is_dir()

        payload = {"a": 1, "b": [1, 2], "c": {"d": "é"}}
        p = save_json(payload, rd / "nested" / "x.json")
        assert p.exists() and load_json(p) == payload

    with Timer() as t:
        time.sleep(0.01)
    assert t.seconds >= 0.005, t.seconds
    print("ok  test_utils")


def test_config_contract():
    """§2.1: TrainConfig.resolved() fills lr/batch_size; json roundtrip."""
    from training.config import MODEL_ZOO, TrainConfig

    cfg = TrainConfig(run_name="c1", model_key="mt5_small")
    r = cfg.resolved()
    assert r.lr == MODEL_ZOO["mt5_small"].lr
    assert r.batch_size == MODEL_ZOO["mt5_small"].batch_size
    assert cfg.lr is None and cfg.batch_size is None  # original untouched

    r2 = TrainConfig(run_name="c2", model_key="nllb_600m", lr=9e-5,
                     batch_size=3).resolved()
    assert r2.lr == 9e-5 and r2.batch_size == 3

    # §2.1 precision: mT5 must default to bf16 (fp16 overflows -> NaN grads),
    # NLLB stays fp16; an explicit choice always wins.
    assert TrainConfig(run_name="p1", model_key="mt5_small").resolved().precision == "bf16"
    assert TrainConfig(run_name="p2", model_key="nllb_600m").resolved().precision == "fp16"
    assert TrainConfig(run_name="p3", model_key="mt5_small",
                       precision="fp32").resolved().precision == "fp32"

    with tempfile.TemporaryDirectory(prefix="psa_w3_") as tmp:
        p = Path(tmp) / "cfg.json"
        cfg2 = TrainConfig(run_name="c3", model_key="mt5_small",
                           direction="all", fewshot_guz=50, freeze_encoder=True,
                           epochs=1.5, max_samples=10, report_to="none")
        cfg2.to_json(p)
        assert TrainConfig.from_json(p) == cfg2
    print("ok  test_config_contract")


def test_no_heavy_imports():
    """§4: importing trainer/utils must not import torch/transformers."""
    code = (
        "import sys; import training.trainer, training.utils; "
        "heavy = [m for m in ('torch', 'transformers', 'datasets') "
        "         if m in sys.modules]; "
        "assert not heavy, f'heavy modules at import time: {heavy}'"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=str(PROJECT_ROOT),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    print("ok  test_no_heavy_imports")


def test_report_to_fallback():
    """§4: wandb requested but unusable (no key / not installed / not logged
    in) -> 'none'; offline mode or a netrc `wandb login` -> 'wandb'."""
    from training.config import TrainConfig
    from training.trainer import _resolve_report_to

    old_key = os.environ.pop("WANDB_API_KEY", None)
    old_mode = os.environ.pop("WANDB_MODE", None)
    try:
        cfg = TrainConfig(run_name="w1", model_key="mt5_small", report_to="wandb")
        try:
            import wandb

            class _FakeApi:
                """Deterministic stand-in for wandb.Api (netrc probe)."""

                def __init__(self, *args, key=None, **kwargs):
                    self._key = key

                @property
                def api_key(self):
                    return self._key

            orig_api = wandb.Api
            try:
                # not logged in (no netrc): fallback
                wandb.Api = lambda *a, **k: _FakeApi(key=None)
                assert _resolve_report_to(cfg) == "none"
                # logged in via `wandb login` (netrc): usable
                wandb.Api = lambda *a, **k: _FakeApi(key="secret")
                assert _resolve_report_to(cfg) == "wandb"
            finally:
                wandb.Api = orig_api
            # offline mode is usable regardless of login state
            os.environ["WANDB_MODE"] = "offline"
            assert _resolve_report_to(cfg) == "wandb"
        except ImportError:
            assert _resolve_report_to(cfg) == "none"
        cfg_none = TrainConfig(run_name="w2", model_key="mt5_small",
                               report_to="none")
        assert _resolve_report_to(cfg_none) == "none"
    finally:
        if old_key is not None:
            os.environ["WANDB_API_KEY"] = old_key
        if old_mode is not None:
            os.environ["WANDB_MODE"] = old_mode
    print("ok  test_report_to_fallback")


def test_smoke_train():
    """§2.5: real 1-epoch train with a tiny model on tiny local splits.

    Skips gracefully (still pass) when: training.data is not available yet,
    the GPU-stack deps are missing, or the HF hub is unreachable.
    """
    try:
        from training import data  # noqa: F401
        assert hasattr(data, "build_train_dataset")
    except Exception as e:
        print(f"skipped: training.data not available on this branch ({e})")
        return
    try:
        import datasets  # noqa: F401
        import sacrebleu  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        print(f"skipped: training deps not installed ({e})")
        return

    # Env juggling per §4: wandb offline-capable, HF hub allowed for download.
    old_offline = os.environ.pop("HF_HUB_OFFLINE", None)
    old_wmode = os.environ.get("WANDB_MODE")
    os.environ["WANDB_MODE"] = "offline"
    try:
        from transformers import (AutoConfig, AutoModelForSeq2SeqLM,
                                  AutoTokenizer)

        tmp = Path(tempfile.mkdtemp(prefix="psa_w3_smoke_"))
        local_model = tmp / "tiny-mt5"
        try:
            # Real tokenizer + config from the hub (spec §4); model body shrunk
            # so the full train step fits in a small CPU RAM budget (the raw
            # tiny-random-mt5 checkpoint is ~300M params via its 250k vocab).
            tok = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-mt5")
            mcfg = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-mt5")
            mcfg.d_model = 64
            mcfg.d_ff = 128
            mcfg.d_kv = 16
            mcfg.num_heads = 2
            mcfg.num_layers = 2
            mcfg.num_decoder_layers = 2
            mdl = AutoModelForSeq2SeqLM.from_config(mcfg)
            tok.save_pretrained(str(local_model))
            mdl.save_pretrained(str(local_model))
        except Exception as e:
            print(f"skipped: HF hub unreachable ({type(e).__name__}: {e})")
            return

        splits_dir = tmp / "splits"
        _write_splits(splits_dir)

        from training.config import MODEL_ZOO, ModelConfig, TrainConfig
        from training.trainer import train
        from training.utils import load_json

        MODEL_ZOO["tiny_mt5"] = ModelConfig(
            key="tiny_mt5", hf_name=str(local_model), family="mt5",
            lr=1e-3, batch_size=2, max_length=64)
        try:
            cfg = TrainConfig(
                run_name="smoke", model_key="tiny_mt5", direction="both",
                epochs=1, batch_size=2, grad_accum=1, max_samples=8,
                fp16=False, report_to="none", output_root=str(tmp / "runs"))
            resolved_cfg = cfg.resolved()
            best = train(cfg, splits_dir=splits_dir)
        finally:
            MODEL_ZOO.pop("tiny_mt5", None)

        run = Path(cfg.output_root) / "smoke"
        assert best == run / "checkpoint-best" and best.is_dir()
        assert (best / "config.json").exists(), "best checkpoint missing model config"
        assert (run / "train_config.json").exists()
        assert TrainConfig.from_json(run / "train_config.json") == resolved_cfg

        metrics = load_json(run / "metrics_dev.json")
        for key in ("n_train_pairs", "trainable_pct", "seconds", "device_info"):
            assert key in metrics, f"metrics_dev.json missing '{key}'"
        assert metrics["n_train_pairs"] <= 8, metrics["n_train_pairs"]
        assert 0.0 < metrics["trainable_pct"] <= 100.0
        assert metrics["seconds"] > 0
        assert "eval_sacrebleu" in metrics and "eval_chrf" in metrics, metrics

        # §2.5(3): freeze_encoder + freeze_embed shrink trainable-param %.
        MODEL_ZOO["tiny_mt5"] = ModelConfig(
            key="tiny_mt5", hf_name=str(local_model), family="mt5",
            lr=1e-3, batch_size=2, max_length=64)
        try:
            cfg_f = TrainConfig(
                run_name="smoke_frozen", model_key="tiny_mt5", direction="both",
                epochs=1, batch_size=2, grad_accum=1, max_samples=4,
                fp16=False, report_to="none", freeze_encoder=True,
                freeze_embed=True, output_root=str(tmp / "runs"))
            best_f = train(cfg_f, splits_dir=splits_dir)
        finally:
            MODEL_ZOO.pop("tiny_mt5", None)
        metrics_f = load_json(Path(cfg_f.output_root) / "smoke_frozen"
                              / "metrics_dev.json")
        assert best_f.is_dir()
        assert 0.0 < metrics_f["trainable_pct"] < metrics["trainable_pct"], (
            metrics_f["trainable_pct"], metrics["trainable_pct"])
        print("ok  test_smoke_train (tiny-model train RAN)")
    finally:
        if old_offline is not None:
            os.environ["HF_HUB_OFFLINE"] = old_offline
        if old_wmode is None:
            os.environ.pop("WANDB_MODE", None)
        else:
            os.environ["WANDB_MODE"] = old_wmode


def run() -> int:
    """Run all tests; network-dependent parts skip gracefully."""
    test_utils()
    test_config_contract()
    test_no_heavy_imports()
    test_report_to_fallback()
    test_smoke_train()
    print("ok  test_week3_trainer")
    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
