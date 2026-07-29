"""Training utilities — seeding, device info, run dirs, timing, JSON IO.

Frozen contract (SPEC_WEEK3.md §2.4). Heavy deps (numpy/torch) are imported
lazily inside functions so this module stays importable without the GPU stack.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

from .config import TrainConfig


def set_seed(seed: int = 42) -> None:
    """Seed python-random, numpy and torch (if available) for reproducibility."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def device_info() -> dict:
    """Return {"device": "cuda|cpu", "gpu_name": str|None, "torch": str|None}."""
    try:
        import torch
    except ImportError:
        return {"device": "cpu", "gpu_name": None, "torch": None}
    if torch.cuda.is_available():
        return {
            "device": "cuda",
            "gpu_name": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
        }
    return {"device": "cpu", "gpu_name": None, "torch": torch.__version__}


def run_dir(cfg: TrainConfig) -> Path:
    """Return <output_root>/<run_name>, creating it (and parents) if needed."""
    path = Path(cfg.output_root) / cfg.run_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Any, path: str | Path) -> Path:
    """Serialize obj as UTF-8 JSON (indent=2), creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return path


def load_json(path: str | Path) -> Any:
    """Load a JSON file written with UTF-8 encoding."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Timer:
    """Context manager measuring wall-clock seconds.

    Usage:
        with Timer() as t:
            ...
        print(t.seconds)
    """

    def __init__(self) -> None:
        self.seconds: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> bool:
        self.seconds = time.perf_counter() - self._start
        return False
