# Week 2 Report — Preprocessing & Exploratory Data Analysis

**Course:** DSA4020A Natural Language Processing
**Group:** 10
**Week:** 2 — Data Processing & EDA
**Date:** 29 July 2026

---

## 1. Overview

This week we built a full preprocessing and EDA pipeline and ran it over our
PSA dataset. Mid-week, the lecturer issued the PSA Framework and a gold
dataset (`PSA_KE_Final.csv`); we audited our scraped rows against that
standard, deleted 9,548 non-PSA scraped rows, and merged 2,852 gold PSAs with
Ekegusii translations (details in the Week 1 report addendum and
`reports/framework_audit.md`). All Week 2 outputs below were **regenerated on
the remediated 6,823-row dataset**, so every number in this report reflects
the dataset as it now stands.

The pipeline itself is unchanged and rerunnable end-to-end with one command
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
  English and Kiswahili. **0 rows in 6,823 were flagged** — the remediated
  dataset (gold PSAs + TICO-19 + vetted scraped rows) is clean monolingual
  text in each column.
- Cultural glossary (`data/glossary.json`): 24 Kenyan institutional/cultural
  terms (harambee, nyumba kumi, matatu, M-Pesa, boda boda, SHA, KCSE…) with
  Kiswahili equivalents and notes. **120 rows** contain at least one glossary
  term — these get consistent terminology treatment in Week 3.
- Output: `data/processed/psa_preprocessed.csv` with per-row token counts,
  code-switch flags and glossary tags. Original columns untouched.

**EDA** (`SRC/eda.py`, output in `reports/week2_eda_report.md` + `reports/figures/`)

Six figures: domain bar and pie charts, English and Kiswahili length histograms,
source distribution, paired vs unpaired breakdown — plus computed statistics and
auto-generated observations.

**Splits** (`SRC/splits.py`, output in `data/processed/splits/`)

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

1. **The dataset is now genuinely parallel.** Before the framework audit only
   23.2% of rows had a Kiswahili translation; after deleting non-PSA scraped
   text and merging the gold dataset, 87.5% are paired — and 2,848 rows have
   Ekegusii, our Week 3 transfer target. The remaining 851 English-only rows
   (mostly TICO-19) are candidates for Week 3 back-translation.
2. **Health still leads, but the balance is workable** (57.8%, down from 64%).
   Security and Education — our thinnest domains at the end of Week 1 (277 and
   28 rows) — are now 952 and 828 rows thanks to the gold dataset's
   Security/Education-heavy composition. Week 3 training will still use a
   domain-balanced training view rather than deleting data.
3. **Kiswahili now runs only ~7% longer than English** (20.9 vs 19.6 mean
   words — the gold PSAs are tighter than TICO-19's clinical text). Still
   relevant for the Week 3 subword budget and max-sequence-length settings.
4. **A handful of very short rows (min 3 tokens)** survive the 4-word cleaning
   minimum because the two tokenizers count initials/punctuation differently.
   Harmless; Week 3 data loading filters below 3 tokens.

## 5. Issues handled this week

- **Missing translations:** dropped from 76.8% to 12.5% through remediation;
  residual handled by Week 3 back-translation plan.
- **Non-PSA content:** identified via the lecturer's framework, classified per
  row (`psa_class` in Metadata), deleted for scraped rows, documented in
  `reports/framework_audit.md`.
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

- The framework audit forced a hard call: deleting 9,548 rows we had spent two
  weeks collecting. The deciding argument was the brief itself — a PSA dataset
  should contain PSAs — plus the gold dataset replacing volume with quality
  (and bringing Ekegusii with it).
- Calibrating the PSA classifier was subtler than expected: the lecturer's own
  file applies a broader standard than the framework text (announcements count,
  not just imperatives). We documented the classifier rules and the per-row
  outcomes so every deletion is auditable.
- Writing a code-switch detector that works without a trained model is harder
  than it sounds; our stopword heuristic flagged zero rows in the remediated
  dataset, which matches the data's nature (clean, professional translations).
- Near-duplicate leakage across splits is easy to get wrong: naive random
  splitting put duplicate texts in both train and test until we grouped by
  normalized text first.

## 7. Next steps (Week 3 — Modeling)

- Experiment tracking setup (Weights & Biases)
- Baselines: mT5-small and NLLB-200-distilled, zero-shot vs fine-tuned
- Ekegusii training from the 2,848 real PSA pairs (FLORES-200 devtest reserved
  purely as a benchmark)
- Back-translation augmentation for the 851 English-only rows
- Domain-balanced training view per the EDA decision
