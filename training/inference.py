"""Week 3 inference — MTTranslator + DEMO_PSAS (see SPEC_WEEK3.md §2.7, §3).

Import-time never requires torch/transformers: heavy imports happen lazily
inside ``MTTranslator.__init__`` / ``translate`` so the rest of the package
(metrics, ablation matrix, CLIs with --help) works on a CPU-only machine
without the GPU stack installed.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import LANGS, MODEL_ZOO, NLLB_CODES
from .lang_tokens import ensure_lang_token

# 8 original short PSA demo lines (>=1 per domain, mix of eng/swa sources).
DEMO_PSAS: list[dict] = [
    {"domain": "Health", "src": "eng",
     "text": "Wash your hands with soap and clean water for at least twenty seconds."},
    {"domain": "Health", "src": "swa",
     "text": "Hakikisha mtoto wako amepata chanjo zote muhimu kwa wakati."},
    {"domain": "Education", "src": "eng",
     "text": "Every child has the right to free primary education."},
    {"domain": "Education", "src": "swa",
     "text": "Rudisha watoto shuleni; elimu ndiyo ufunguo wa maendeleo."},
    {"domain": "Agriculture", "src": "eng",
     "text": "Plant certified drought-tolerant seeds before the long rains begin."},
    {"domain": "Agriculture", "src": "swa",
     "text": "Hifadhi mbolea mahali salama ili kupata mavuno bora msimu huu."},
    {"domain": "Security", "src": "eng",
     "text": "Report any suspicious activity to your nearest police station immediately."},
    {"domain": "Governance", "src": "swa",
     "text": "Jitokeze kupiga kura; sauti yako ndiyo nguvu ya mabadiliko."},
]

# config.json model_type -> family (see SPEC_WEEK3.md §2.7 sniffing rule).
_MODEL_TYPE_TO_FAMILY = {
    "t5": "mt5", "mt5": "mt5", "umt5": "mt5",
    "nllb": "nllb", "m2m_100": "nllb", "m2m100": "nllb",
    "mbart": "nllb", "mbart50": "nllb",
}
_FAMILY_TO_MODEL_KEY = {"mt5": "mt5_small", "nllb": "nllb_600m"}


def _sniff_model_key(checkpoint: Path) -> str | None:
    """Resolve model_key for a checkpoint dir.

    Priority: train_config.json in the checkpoint dir or its parent (the
    trainer writes it into runs/<run>/), else config.json model_type
    sniffing. Returns None if nothing can be determined.
    """
    for cand in (checkpoint / "train_config.json",
                 checkpoint.parent / "train_config.json"):
        if cand.is_file():
            try:
                cfg = json.loads(cand.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            key = cfg.get("model_key")
            if key in MODEL_ZOO:
                return key
    cfg_path = checkpoint / "config.json"
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        family = _MODEL_TYPE_TO_FAMILY.get(str(cfg.get("model_type", "")).lower())
        if family:
            return _FAMILY_TO_MODEL_KEY[family]
    return None


class MTTranslator:
    """Translate texts with a fine-tuned (or base) seq2seq checkpoint.

    checkpoint: hub id or local dir (e.g. runs/<run>/checkpoint-best).
    model_key: "mt5_small" | "nllb_600m"; None -> sniffed from
    train_config.json / config.json (falls back to mt5_small with a warning).
    """

    def __init__(self, checkpoint: str | Path, model_key: str | None = None):
        self.checkpoint = str(checkpoint)
        if model_key is None:
            model_key = _sniff_model_key(Path(self.checkpoint))
            if model_key is None:
                print(f"[inference] could not sniff model_key for "
                      f"'{self.checkpoint}'; assuming 'mt5_small'")
                model_key = "mt5_small"
        if model_key not in MODEL_ZOO:
            raise KeyError(
                f"unknown model_key '{model_key}' (have: {sorted(MODEL_ZOO)})")
        self.model_key = model_key
        self.model_cfg = MODEL_ZOO[model_key]
        self.family = self.model_cfg.family

        import torch  # noqa: lazy import (SPEC_WEEK3.md §4)
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint)
        self.model.to(self.device)
        self.model.eval()

    # -- family-specific encoding (SPEC_WEEK3.md §3, frozen) ---------------

    def _encode_sources(self, texts: list[str], src: str, tgt: str):
        tok = self.tokenizer
        if self.family == "mt5":
            prefix = f"translate {LANGS[src]} to {LANGS[tgt]}: "
            return tok([prefix + t for t in texts], return_tensors="pt",
                       padding=True, truncation=True,
                       max_length=self.model_cfg.max_length)
        # nllb
        tok.src_lang = NLLB_CODES[src]
        return tok(texts, return_tensors="pt", padding=True, truncation=True,
                   max_length=self.model_cfg.max_length)

    def _ensure_nllb_langs(self, src: str, tgt: str) -> None:
        """Add any missing NLLB language tokens once (guz_Latn is absent from
        the hub tokenizer; fine-tuned checkpoints already carry it -> no-op).
        Only meaningful for checkpoints that were fine-tuned with guz pairs —
        on a base hub model this just enables the mechanics, not competence.
        """
        if self.family != "nllb":
            return
        ensured = getattr(self, "_ensured_langs", set())
        for code in (NLLB_CODES[src], NLLB_CODES[tgt]):
            if code not in ensured:
                ensure_lang_token(self.model, self.tokenizer, code)
                ensured.add(code)
        self._ensured_langs = ensured

    def _forced_bos_id(self, tgt: str) -> int | None:
        """forced_bos_token_id for nllb generation; None for mt5."""
        if self.family != "nllb":
            return None
        code = NLLB_CODES[tgt]
        tok = self.tokenizer
        bos = tok.convert_tokens_to_ids(code)
        if bos is None or bos == getattr(tok, "unk_token_id", None):
            lang_map = getattr(tok, "lang_code_to_id", None) or {}
            bos = lang_map.get(code)
        return bos

    # -- public API ---------------------------------------------------------

    def translate(self, texts: list[str], src: str, tgt: str,
                  max_length: int = 128, num_beams: int = 4) -> list[str]:
        """Translate texts from src to tgt ({eng, swa, guz}) with batching."""
        if src not in LANGS or tgt not in LANGS:
            raise ValueError(f"src/tgt must be in {sorted(LANGS)}, got {src}->{tgt}")
        if src == tgt:
            return list(texts)
        torch = self._torch
        out: list[str] = []
        batch_size = 16
        self._ensure_nllb_langs(src, tgt)
        forced_bos = self._forced_bos_id(tgt)
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                chunk = [str(t) for t in texts[i:i + batch_size]]
                enc = self._encode_sources(chunk, src, tgt).to(self.device)
                gen_kwargs = dict(max_length=max_length, num_beams=num_beams)
                if forced_bos is not None:
                    gen_kwargs["forced_bos_token_id"] = forced_bos
                ids = self.model.generate(**enc, **gen_kwargs)
                out.extend(tok for tok in self.tokenizer.batch_decode(
                    ids, skip_special_tokens=True))
        return out
