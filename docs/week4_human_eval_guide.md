# Native-Speaker Human Evaluation Guide (Week 4 — model output)

File to review: `data/validation/week4_model_output_review.csv`, built by
`python scripts/build_human_eval_sheet.py` from the Week 4 en→guz
predictions. This is **different from** the Week 2
`docs/validation_guide.md`: that guide rates the *dataset's* professional
Kiswahili translations; this one rates our **fine-tuned model's Ekegusii
output** against the English source (and the held-out gold Ekegusii, given
for reference only).

Each reviewer fills the empty columns: `Reviewer`, `Fluency_1to5`,
`Adequacy_1to5`, `Issues`, `Notes`.

## What to rate

For each row, read the **English** source and the model's
**Model_Ekegusii** output. `Reference_Ekegusii` (the held-out gold
translation) is provided for context only — score the model's output on
its own merits; a fluent Ekegusii sentence that says the same thing in
different words is not a failure just because it diverges from the
reference wording.

- **Fluency_1to5** — is `Model_Ekegusii` grammatical, natural Ekegusii,
  ignoring the English? 5 = flawless native Ekegusii; 1 = ungrammatical /
  incomprehensible.
- **Adequacy_1to5** — does `Model_Ekegusii` convey the same meaning as the
  **English**? 5 = full meaning preserved; 1 = meaning lost or wrong.

## Issues to flag (comma-separated in `Issues`)

- `mistranslation` — the Ekegusii says something different from the English.
- `omission` — part of the English meaning is missing.
- `addition` — the Ekegusii adds information not in the English.
- `grammar` — spelling, agreement, or word-order errors.
- `repetition_loop` — the model repeats a word/phrase instead of finishing
  the sentence (the low-resource failure mode documented in
  `reports/week3_report.md` §8; also auto-flagged for you in
  `reports/week4_eval/predictions/en-guz.csv`'s `repetition_flag` column,
  but the heuristic is a candidate list, not a verdict — confirm by ear).
- `language_confusion` — the output is fluent but in the *wrong* language
  (e.g. Swahili instead of Ekegusii — see the Week 3 finding that donor-
  language dominance appears at very low guz data scales).
- `cultural term` — a glossary term (see `data/glossary.json`) is handled
  inconsistently or wrongly.

Use `Notes` for anything else (awkward register, dialect variant,
orthographic variation worth normalizing, or a comment on how
`Model_Ekegusii` compares to `Reference_Ekegusii` if it's informative).

## Rules

1. Rate independently — do not discuss scores while reviewing.
2. Do not edit the `English`, `Reference_Ekegusii`, or `Model_Ekegusii`
   columns; only fill the reviewer columns.
3. When unsure between two scores, choose the lower one and explain in
   `Notes`.
4. Save the completed file as
   `week4_model_output_review_reviewed_<name>.csv` and share it with the
   integration lead for `reports/week4_report.md`.
