# Week 4 spec — final evaluation, error analysis, human eval, deployment

Scope per the course brief: evaluation, deployment and documentation. This
supersedes nothing in `SPEC_WEEK3.md` (training/inference/config stay
frozen); it only adds new, independent artifacts on top of them.

## 1. Final evaluation — `scripts/run_week4_eval.py`

Difference from Week 3's `scripts/run_eval.py`: that script evaluates a
capped sample (`--n`, default 200) per registered spec, for fast iteration
during the ablation matrix. This script evaluates the **full** held-out
test split (`data/processed/splits/test.csv`) for all four directions
(en-sw, sw-en, en-guz, guz-en), and additionally:

- keeps `PSA_ID` + `Domain` alongside every row, so results can be broken
  down per domain (Health / Education / Agriculture / Security /
  Governance), not just overall
- computes a per-row sentence-chrF and a repetition-loop heuristic flag
  (`repetition_flag`), so the predictions double as error-analysis input
- writes `reports/week4_eval/summary.json` (overall + per-domain BLEU/chrF
  per direction) and `reports/week4_eval/predictions/<direction>.csv`
  (one row per example: source, reference, hypothesis, domain, sentence
  chrF, repetition flag)

The en-guz direction's predictions file doubles as the source for the
Ekegusii benchmark result (equivalent to Week 3's `psa_test_en-guz` /
`psa_test_guz-en` specs, but full-size and domain-tagged) and for the
human-eval sample (§3).

## 2. Error analysis — `scripts/error_analysis.py`

Pure pandas + sacrebleu over the predictions CSVs from §1 — no torch, no
checkpoint, no GPU. Can run on any machine once the predictions exist.
Writes `reports/week4_error_analysis.md`: per-domain score table, the k
lowest sentence-chrF examples per direction, and repetition-flagged
examples. The repetition heuristic is a candidate list for a human read,
not a verdict — every flagged/worst example still needs manual
confirmation before it goes in `reports/week4_report.md`.

## 3. Human evaluation — `scripts/build_human_eval_sheet.py` +
`docs/week4_human_eval_guide.md`

Distinct from the Week 2 `data/validation/validation_subset.csv` /
`docs/validation_guide.md` (which rate the *dataset's* professional EN-SW
translations). This rates the **model's** Ekegusii output: a seeded,
domain-stratified sample of `reports/week4_eval/predictions/en-guz.csv`,
written to `data/validation/week4_model_output_review.csv` with reviewer
columns (`Reviewer`, `Fluency_1to5`, `Adequacy_1to5`, `Issues`, `Notes`) —
same 1-5 fluency/adequacy scale as Week 2, plus two Ekegusii-specific issue
tags (`repetition_loop`, `language_confusion`) drawn directly from the
Week 3 report's documented failure modes.

## 4. Deployment — `app.py`

Streamlit app (`requirements-app.txt`) wrapping `training.inference.
MTTranslator` exactly as `scripts/translate.py` does for the CLI: same
checkpoint auto-discovery rule (newest `runs/*/checkpoint-best`), same
`DEMO_PSAS` table for the sample-PSA tab. Two tabs: free-text translate
between any two of {English, Kiswahili, Ekegusii}, and the 8-PSA demo table
from `training/inference.py` with per-target translate buttons. No new
model logic — this is purely a UI over the frozen Week 3 inference
contract, so it stays correct as long as `MTTranslator` does.

## 5. Documentation — `reports/week4_report.md`
Skeleton written with the final-eval, error-analysis and human-eval
sections stubbed for the actual numbers (produced by running §1-§3 on the
Kinesis node against the best Week 3 checkpoint, `ft_nllb_guz_all` per
`reports/week3_results.md`). Fill in and remove the `TODO` markers once
those scripts have been run.

## 6. Ekegusii augmentation experiment (added post-eval)

Motivated by the §1-§2 results (16.7% repetition-loop rate and one
copy-through in en-guz; the Week 3 finding that guz quality scales with
pair count): extend `training/augment.py` from eng->swa to **eng->guz**
back-translation and retrain.

- `training/augment.py::backtranslate(tgt="guz")`: train-split rows that
  lack Ekegusii (English-only + EN-SW rows; 3,577 in the canonical split)
  are translated eng->guz by `ft_nllb_guz_all/checkpoint-best` itself with
  no_repeat_ngram_size=3; empty and copy-through generations are dropped.
  Existing Kiswahili text is carried over, yielding sw-guz pairs too.
  Synthetic rows are marked Source="Back-translation", Status="Synthetic",
  PSA_ID suffixed `-BTG`. Train split only — the test split and the guz
  benchmark stay untouched (no leakage).
- `training/data.py::_load_augmented`: now builds all four directions
  from an augmented CSV (was en-sw/sw-en only).
- Retrain: `scripts/run_training.py --model nllb_600m --direction all
  --fewshot-guz -1 --use-augmentation --augmented-csv
  data/processed/augmented_guz.csv --run-name ft_nllb_guz_aug`.
- Comparison: `scripts/run_week4_eval.py` against the new checkpoint,
  reported side-by-side with `ft_nllb_guz_all` in
  `reports/week4_report.md`. Evaluation decoding stays unassisted in both
  (guardrails are a deployment-only choice, §4).
