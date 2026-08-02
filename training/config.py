"""Week 3 training configuration — frozen contract (see SPEC_WEEK3.md §2.1).

This module is owned by the main agent. Do not modify without updating
SPEC_WEEK3.md; every other training module codes against these interfaces.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

LANGS = {"eng": "English", "swa": "Swahili", "guz": "Ekegusii"}
NLLB_CODES = {"eng": "eng_Latn", "swa": "swh_Latn", "guz": "guz_Latn"}

DIRECTIONS_SINGLE = ("en-sw", "sw-en", "en-guz", "sw-guz")


@dataclass
class ModelConfig:
    key: str            # "mt5_small" | "nllb_600m"
    hf_name: str        # hub id
    family: str         # "mt5" | "nllb"
    lr: float           # default learning rate
    batch_size: int     # default per-device batch size
    max_length: int = 128
    precision: str = "fp16"  # "fp16" | "bf16" | "fp32" (GPU mixed precision;
                             # mT5 needs bf16 — its activations overflow fp16
                             # and produce NaN gradients)


MODEL_ZOO: dict[str, ModelConfig] = {
    "mt5_small": ModelConfig(
        key="mt5_small", hf_name="google/mt5-small", family="mt5",
        lr=1e-4, batch_size=16, max_length=128, precision="bf16",
    ),
    "nllb_600m": ModelConfig(
        key="nllb_600m", hf_name="facebook/nllb-200-distilled-600M",
        family="nllb", lr=5e-5, batch_size=8, max_length=128,
        precision="fp16",
    ),
}


@dataclass
class TrainConfig:
    run_name: str
    model_key: str
    direction: str = "both"     # "en-sw"|"sw-en"|"en-guz"|"sw-guz"|"both"|"all"
    fewshot_guz: int = 0        # guz pairs: 0=exclude, N=seeded cap on
                                # PSA-sourced (Ekegusii) pairs, -1=all.
                                # (FLORES-200 was evaluated and dropped: it
                                # has no Ekegusii — the PSA train split is
                                # the only guz source.)
    use_augmentation: bool = False
    freeze_encoder: bool = False
    freeze_embed: bool = False
    epochs: float = 3.0
    lr: float | None = None     # None -> MODEL_ZOO default
    batch_size: int | None = None
    grad_accum: int = 2
    max_samples: int | None = None   # cap training pairs (smoke/quick runs)
    fp16: bool = True     # legacy: superseded by `precision` below; kept so
                          # old train_config.json files still load
    precision: str | None = None     # None -> MODEL_ZOO default
                                     # ("fp16" | "bf16" | "fp32")
    seed: int = 42
    output_root: str = "runs"
    report_to: str = "wandb"    # "wandb" | "none"
    wandb_project: str = "psa-mt-group10"

    def resolved(self, zoo: dict[str, ModelConfig] | None = None) -> "TrainConfig":
        """Return a copy with lr/batch_size filled from the model zoo."""
        zoo = zoo or MODEL_ZOO
        if self.model_key not in zoo:
            raise KeyError(
                f"unknown model_key '{self.model_key}' (have: {sorted(zoo)})")
        mc = zoo[self.model_key]
        out = TrainConfig(**asdict(self))
        out.lr = self.lr if self.lr is not None else mc.lr
        out.batch_size = self.batch_size if self.batch_size is not None else mc.batch_size
        out.precision = self.precision if self.precision is not None else mc.precision
        return out

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def directions(self) -> list[str]:
        """Expand direction aliases to explicit direction codes."""
        if self.direction == "both":
            return ["en-sw", "sw-en"]
        if self.direction == "all":
            return ["en-sw", "sw-en", "en-guz", "sw-guz"]
        if self.direction in DIRECTIONS_SINGLE:
            return [self.direction]
        raise ValueError(f"unknown direction '{self.direction}'")
