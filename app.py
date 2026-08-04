#!/usr/bin/env python3
"""Week 4 deployment — Streamlit demo wrapping training.inference.MTTranslator
(docs/SPEC_WEEK4.md). This is the "deployable web app" success criterion.

Setup:
    pip install -r requirements.txt -r requirements-training.txt -r requirements-app.txt

Run:
    streamlit run app.py

The newest runs/*/checkpoint-best is auto-discovered on load (same rule as
scripts/translate.py); you can override it in the sidebar with any local
checkpoint dir or hub id.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.config import LANGS  # noqa: E402
from training.inference import DEMO_PSAS  # noqa: E402


def discover_checkpoint(runs_root: Path = PROJECT_ROOT / "runs") -> Path | None:
    """Newest runs/*/checkpoint-best by modification time (scripts/translate.py rule)."""
    if not runs_root.is_dir():
        return None
    candidates = [p for p in runs_root.glob("*/checkpoint-best") if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


@st.cache_resource(show_spinner="Loading model checkpoint...")
def load_translator(checkpoint: str):
    from training.inference import MTTranslator
    return MTTranslator(checkpoint)


def lang_label(code: str) -> str:
    return LANGS[code]


st.set_page_config(page_title="PSA Translator — Group 10", page_icon="🌍",
                    layout="centered")
st.title("🌍 PSA Machine Translation")
st.caption("DSA4020A Group 10 — few-shot cross-lingual transfer for Kenyan "
           "Public Service Announcements (English / Kiswahili / Ekegusii).")

st.sidebar.header("Model")
auto_ckpt = discover_checkpoint()
ckpt_path = st.sidebar.text_input(
    "Checkpoint path or hub id",
    value=str(auto_ckpt) if auto_ckpt else "",
    help="A local runs/<run>/checkpoint-best directory, or any Hugging Face "
         "hub id (base models will translate with zero-shot quality only).")

if not ckpt_path:
    st.warning(
        "No checkpoint found under `runs/*/checkpoint-best`. Train a model "
        "first (`scripts/run_training.py`, see `docs/week3_kinesis_guide.md`) "
        "or enter a checkpoint path / hub id in the sidebar.")
    st.stop()

try:
    translator = load_translator(ckpt_path)
except Exception as exc:  # noqa: BLE001 — surface any load error to the user
    st.error(f"Could not load checkpoint `{ckpt_path}`:\n\n{exc}")
    st.stop()

st.sidebar.success(f"{translator.model_key} ({translator.family}) on "
                    f"{translator.device}")
st.sidebar.markdown("---")
st.sidebar.caption(
    "Glossary terms (harambee, matatu, nyumba kumi, ...) are kept "
    "untranslated by design — see `data/glossary.json`.")
st.sidebar.caption(
    "Decoding uses no_repeat_ngram_size=3 as a guardrail against the "
    "repetition-loop failure mode documented in reports/week4_report.md; "
    "evaluation numbers in that report were generated without it.")

tab_translate, tab_demo = st.tabs(["Translate", "Sample PSAs"])

with tab_translate:
    langs = list(LANGS)
    col1, col2 = st.columns(2)
    src = col1.selectbox("From", langs, format_func=lang_label, index=0)
    tgt = col2.selectbox("To", langs, format_func=lang_label, index=1)
    if src == tgt:
        st.info("Choose two different languages.")
    text = st.text_area("Text to translate", height=120,
                        placeholder="Type an English, Kiswahili, or "
                                    "Ekegusii sentence...")
    if st.button("Translate", type="primary",
                 disabled=(not text.strip() or src == tgt)):
        with st.spinner("Translating..."):
            output = translator.translate([text], src=src, tgt=tgt,
                                          no_repeat_ngram_size=3)[0]
        st.text_area(f"{lang_label(tgt)} translation", value=output, height=120)

with tab_demo:
    st.caption("The 8 sample PSAs across all five domains — the Week 3 "
               "demo success criterion (`scripts/translate.py --demo`).")
    domains = sorted({p["domain"] for p in DEMO_PSAS})
    domain_filter = st.multiselect("Filter by domain", domains, default=domains)
    for i, psa in enumerate(DEMO_PSAS):
        if psa["domain"] not in domain_filter:
            continue
        targets = ["swa", "guz"] if psa["src"] == "eng" else ["eng", "guz"]
        st.markdown(f"**{psa['domain']}** ({lang_label(psa['src'])}): "
                    f"{psa['text']}")
        cols = st.columns(len(targets))
        for col, tgt_code in zip(cols, targets):
            with col:
                if st.button(f"→ {lang_label(tgt_code)}", key=f"demo-{i}-{tgt_code}"):
                    with st.spinner("Translating..."):
                        out = translator.translate([psa["text"]], src=psa["src"],
                                                   tgt=tgt_code,
                                                   no_repeat_ngram_size=3)[0]
                    st.write(out)
        st.divider()
