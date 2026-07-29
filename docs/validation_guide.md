# Native-Speaker Validation Guide (Week 2)

File to review: `data/validation/validation_subset.csv` — a stratified
(by Domain) random sample of the dataset. Each reviewer fills the empty
columns: `Reviewer`, `Fluency_1to5`, `Adequacy_1to5`, `Issues`, `Notes`.

## What to rate

For each row, compare the **English** text with its **Kiswahili**
translation and score two dimensions on a 1–5 scale:

- **Fluency_1to5** — is the Kiswahili grammatical and natural, ignoring
  the English? 5 = flawless native Kiswahili; 1 = ungrammatical/
  incomprehensible.
- **Adequacy_1to5** — does the Kiswahili convey the same meaning as the
  English? 5 = full meaning preserved; 1 = meaning lost or wrong.

Rows with an **empty Kiswahili cell are not ratable**: leave the score
columns blank and write `missing translation` under `Issues`.

## Issues to flag (comma-separated in `Issues`)

- `mistranslation` — the Kiswahili says something different.
- `omission` — part of the English meaning is missing.
- `addition` — the Kiswahili adds information not in the English.
- `grammar` — spelling, agreement, or word-order errors.
- `cultural term` — a glossary term (see `data/glossary.json`, e.g.
  harambee, matatu, NHIF) is translated inconsistently or wrongly.

Use `Notes` for anything else (e.g. awkward register, dialect variant,
orthographic variation worth normalizing). Write your name or initials
in `Reviewer` on every row you score.

## Rules

1. Rate independently — do not discuss scores while reviewing.
2. Do not edit the English, Kiswahili, or Ekegusii columns; only fill
   the reviewer columns.
3. When unsure between two scores, choose the lower one and explain in
   `Notes`.
4. Save the completed file as `validation_subset_reviewed_<name>.csv`
   and share it with the integration lead.
