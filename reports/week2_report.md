# Week 2 Report — Preprocessing & Exploratory Data Analysis

**Course:** DSA4020A Natural Language Processing
**Group:** 10
**Week:** 2 — Data Processing & EDA
**Date:** 29 July 2026

---

## 1. Overview

This week we built a full preprocessing and EDA pipeline and ran it over our
6,823-row PSA dataset (lecturer gold + TICO-19 + audited scraped PSAs +
team-written pairs — composition and the framework audit are in the Week 1
report and `reports/framework_audit.md`). Every number in this report
reflects the dataset as it now stands.

The pipeline is rerunnable end-to-end with one command
(`python scripts/run_week2.py`).

## 2. What we built

**Preprocessing pipeline** (`src/preprocessing.py`, `scripts/run_week2.py`)

- Deep normalization: Unicode NFC, curly quotes/dashes converted to ASCII,
  whitespace collapsed (this handles the orthographic variation we saw across
  sources, e.g. different quote and dash conventions).
- Regex word tokenizer (language-agnostic). We deliberately did not use subword
  tokenization here — SentencePiece/BPE belongs to Week 3 with the model
  tokenizers.
- Code-switching detection: a stopword-ratio heuristic flags sentences that mix
  English and Kiswahili. **0 rows in 6,823 were flagged** — the dataset
  (gold PSAs + TICO-19 + vetted scraped rows) is clean monolingual
  text in each column.
- Cultural glossary (`data/glossary.json`): 24 Kenyan institutional/cultural
  terms (harambee, nyumba kumi, matatu, M-Pesa, boda boda, SHA, KCSE…) with
  Kiswahili equivalents and notes. **120 rows** contain at least one glossary
  term — these get consistent terminology treatment in Week 3.
- Output: `data/processed/psa_preprocessed.csv` with per-row token counts,
  code-switch flags and glossary tags. Original columns untouched.

**EDA** (`src/eda.py`, output in `reports/week2_eda_report.md` + `reports/figures/`)

Six figures: domain bar and pie charts, English and Kiswahili length histograms,
source distribution, paired vs unpaired breakdown — plus computed statistics and
auto-generated observations.

**Splits** (`src/splits.py`, output in `data/processed/splits/`)

Train/dev/test at 90/5/5 (**6,141 / 341 / 341 rows**), stratified by domain,
seeded (seed=42) for reproducibility. Crucially, rows are grouped by normalized
text before splitting so duplicates and near-duplicates cannot leak across
splits — we verified zero overlap. Leaky splits would have made our Week 4 BLEU
scores untrustworthy, so we treated this as a correctness requirement, not a
nicety.

**Validation subset** (`data/validation/validation_subset.csv`)

500 stratified rows with reviewer columns (Fluency, Adequacy, Issues, Notes) and
a review guide (`docs/validation_guide.md`). Native-speaker feedback collection
is in progress.

## 3. EDA findings

| Measure | Value |
|---|---|
| Total rows | 6,823 |
| Paired EN–SW rows | 5,972 (87.5%) |
| English-only rows | 851 (12.5%) |
| Rows with Ekegusii | 2,848 (41.7%) |
| Vocabulary EN / SW | 11,551 / 12,177 types |
| Mean length EN / SW | 19.6 / 20.9 words |
| Code-switched rows | 0 |

**Domain distribution:** Health 3,944 (57.8%), Security 952 (14.0%),
Education 828 (12.1%), Agriculture 559 (8.2%), Governance 540 (7.9%).

**Composition:** lecturer gold 2,852 · TICO-19 corpus 3,004 · audited scraped
PSA 817 · team-written 150.

## 4. What the EDA told us (and what we decided)

1. **The dataset is genuinely parallel.** 87.5% of rows pair English with
   Kiswahili — and 2,848 rows have Ekegusii, our Week 3 transfer target.
   The remaining 851 English-only rows (mostly TICO-19) are candidates for
   Week 3 back-translation.
2. **Health still leads, but the balance is workable** (57.8%). Security and
   Education — the thinnest domains in raw web scraping — stand at 952 and
   828 rows thanks to the gold dataset's Security/Education-heavy
   composition. Week 3 training will still use a domain-balanced training
   view rather than deleting data.
3. **Kiswahili now runs only ~7% longer than English** (20.9 vs 19.6 mean
   words — the gold PSAs are tighter than TICO-19's clinical text). Still
   relevant for the Week 3 subword budget and max-sequence-length settings.
4. **A handful of very short rows (min 3 tokens)** survive the 4-word cleaning
   minimum because the two tokenizers count initials/punctuation differently.
   Harmless; Week 3 data loading filters below 3 tokens.

## 5. Issues handled this week

- **Missing translations:** 851 English-only rows (12.5%) remain; handled by
  the Week 3 back-translation plan.
- **Non-PSA content:** every row is classified against the lecturer's
  framework in the Week 1 build (`psa_class` in Metadata); scraped rows that
  fail are deleted, documented in `reports/framework_audit.md`.
- **Orthographic variation:** handled by deep normalization (quotes, dashes,
  Unicode) so the same word doesn't appear as multiple vocabulary types.
- **Domain imbalance:** documented with figures; strategy decided (balanced
  training view, no further data deletion).
- **Split leakage:** prevented by grouping duplicate texts before splitting;
  verified zero overlap.
- **Reproducibility:** seeded splits (42), seeded langdetect, one-command
  pipeline (`run_week2.py`), auto-regenerated EDA report.
- **Version control:** dataset files total ~3 MB, so we track them directly in
  Git; DVC/Git LFS is unnecessary at this size (documented decision).

## 6. Challenges

- The framework audit forced a hard call: deleting 9,548 scraped rows we had
  spent two weeks collecting. The deciding argument was the brief itself — a
  PSA dataset should contain PSAs — plus the gold dataset replacing volume
  with quality (and bringing Ekegusii with it).
- Calibrating the PSA classifier was subtler than expected: the lecturer's own
  file applies a broader standard than the framework text (announcements count,
  not just imperatives). We documented the classifier rules and the per-row
  outcomes so every deletion is auditable.
- Writing a code-switch detector that works without a trained model is harder
  than it sounds; our stopword heuristic flagged zero rows in the dataset,
  which matches the data's nature (clean, professional translations).
- Near-duplicate leakage across splits is easy to get wrong: naive random
  splitting put duplicate texts in both train and test until we grouped by
  normalized text first.

## 7. Next steps (Week 3 — Modeling)

- Experiment tracking setup (Weights & Biases)
- Baselines: mT5-small and NLLB-200-distilled, zero-shot vs fine-tuned
- Ekegusii training from the 2,848 real PSA pairs; the Ekegusii benchmark is
  built from our held-out test split (FLORES-200 was evaluated and dropped —
  it contains no Ekegusii; see Week 3 notes)
- Back-translation augmentation for the 851 English-only rows
- Domain-balanced training view per the EDA decision
