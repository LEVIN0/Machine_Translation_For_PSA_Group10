# Week 2 Report — Preprocessing & Exploratory Data Analysis

**Course:** DSA4020A Natural Language Processing
**Group:** 10
**Week:** 2 — Data Processing & EDA
**Date:** 29 July 2026

---

## 1. Overview

This week we took the Week 1 dataset (12,705 rows, 13 sources) through a full
preprocessing and EDA pass. We built a reusable preprocessing pipeline
(tokenization, normalization, code-switching detection, cultural-term glossary),
produced the EDA report with six figures, created leakage-safe train/dev/test
splits for Week 3, and prepared the 500-row native-speaker validation subset.
All code is in the repo and rerunnable end-to-end with one command
(`python scripts/run_week2.py`).

## 2. What we built

**Preprocessing pipeline** (`SRC/preprocessing.py`, `scripts/run_week2.py`)

- Deep normalization: Unicode NFC, curly quotes/dashes converted to ASCII,
  whitespace collapsed (this handles the orthographic variation we saw across
  sources, e.g. different quote and dash conventions).
- Regex word tokenizer (language-agnostic). We deliberately did not use subword
  tokenization here — SentencePiece/BPE belongs to Week 3 with the model
  tokenizers.
- Code-switching detection: a stopword-ratio heuristic flags sentences that mix
  English and Kiswahili. Only 1 row in 12,705 was flagged, so code-switching is
  not a major issue in our data (it is mostly clean monolingual text).
- Cultural glossary (`data/glossary.json`): 24 Kenyan institutional/cultural
  terms (harambee, nyumba kumi, matatu, M-Pesa, boda boda, SHA, KCSE…) with
  Kiswahili equivalents and notes. 52 rows contain at least one glossary term —
  these get consistent terminology treatment in Week 3.
- Output: `data/processed/psa_preprocessed.csv` with per-row token counts,
  code-switch flags and glossary tags. Original columns untouched.

**EDA** (`SRC/eda.py`, output in `reports/week2_eda_report.md` + `reports/figures/`)

Six figures: domain bar and pie charts, English and Kiswahili length histograms,
source distribution, paired vs unpaired breakdown — plus computed statistics and
auto-generated observations.

**Splits** (`SRC/splits.py`, output in `data/processed/splits/`)

Train/dev/test at 90/5/5 (11,435 / 635 / 635 rows), stratified by domain, seeded
(seed=42) for reproducibility. Crucially, rows are grouped by normalized text
before splitting so duplicates and near-duplicates cannot leak across splits —
we verified zero overlap. Leaky splits would have made our Week 4 BLEU scores
untrustworthy, so we treated this as a correctness requirement, not a nicety.

**Validation subset** (`data/validation/validation_subset.csv`)

500 stratified rows with reviewer columns (Fluency, Adequacy, Issues, Notes) and
a review guide (`docs/validation_guide.md`). Native-speaker feedback collection
is in progress — each member reviews ~165 rows against the guide.

## 3. EDA findings

| Measure | Value |
|---|---|
| Total rows | 12,705 |
| Paired EN–SW rows | 2,987 (23.5%) |
| English-only rows | 9,718 (76.5%) |
| Ekegusii rows | 0 (placeholder until Week 3 few-shot) |
| Vocabulary EN / SW | 15,737 / 8,307 types |
| Mean length EN / SW | 19.3 / 23.8 words |
| Code-switched rows | 1 |

**Domain distribution:** Health 8,655 (68.1%), Agriculture 2,409 (19.0%),
Governance 1,336 (10.5%), Security 277 (2.2%), Education 28 (0.2%).

## 4. What the EDA told us (and what we decided)

1. **Only 23.5% of rows are actually parallel.** Our supervised training signal
   is ~3k EN–SW pairs, mostly TICO-19. Plan: import Tatoeba EN–SW pairs now, have
   the team write Kiswahili versions of their team-written PSAs, and use
   back-translation (NLLB-200) in Week 3 to create synthetic pairs from vetted
   English-only rows — a standard low-resource augmentation technique.
2. **Health dominates (68.1%).** We are keeping the full dataset (the data is
   good), but Week 3 training will use a balanced "training view" — downsampling
   Health within the training split only, or domain-balanced sampling.
3. **Education and Security remain thin** (28 and 277 rows). These sub-topics
   barely exist as scrapeable English web text in Kenya, so we are covering them
   with team-written PSAs (kit in `docs/team_written_psa_kit.md`, 25 sub-topics).
4. **Kiswahili runs ~23% longer than English** (23.8 vs 19.3 mean words) —
   normal translation expansion, but it matters for the Week 3 subword budget
   and max-sequence-length settings.
5. **A handful of very short rows (min 1 token)** survive despite the 4-word
   cleaning minimum — these are rows made mostly of initials and punctuation,
   which the two tokenizers count differently. We reviewed them and they are
   harmless, but Week 3 data loading will filter below 3 tokens.

## 5. Issues handled this week

- **Missing translations:** quantified (76.5%) rather than assumed; mitigation
  plan above (Tatoeba + team-written + back-translation).
- **Orthographic variation:** handled by deep normalization (quotes, dashes,
  Unicode) so the same word doesn't appear as multiple vocabulary types.
- **Domain imbalance:** documented with figures; strategy decided (balanced
  training view, no data deletion).
- **Split leakage:** prevented by grouping duplicate texts before splitting;
  verified zero overlap.
- **Reproducibility:** seeded splits (42), seeded langdetect, one-command
  pipeline (`run_week2.py`), auto-regenerated EDA report.
- **Version control:** dataset files total ~3 MB, so we track them directly in
  Git; DVC/Git LFS is unnecessary at this size (documented decision).

## 6. Challenges

- The biggest surprise was how little of our "parallel dataset" is actually
  parallel — 76.5% is English-only. Week 1 optimized for volume and quality;
  Week 2 made clear that Week 3's real constraint is *paired* volume.
- Writing a code-switch detector that works without a trained model is harder
  than it sounds; our stopword heuristic is simple but only flagged 1 row, which
  matched our manual impression of the data (clean, monolingual sources).
- Near-duplicate leakage across splits is easy to get wrong: naive random
  splitting put duplicate texts in both train and test until we grouped by
  normalized text first.

## 7. Next steps (Week 3 — Modeling)

- Experiment tracking setup (Weights & Biases or MLflow)
- Baselines: mT5-small and NLLB-200-distilled, zero-shot vs few-shot
- Back-translation augmentation for English-only rows
- Ekegusii few-shot transfer (EN/SW → guz); collect a small native-speaker seed
  set if possible (~100–200 sentences)
- Balanced training view per the EDA decision
