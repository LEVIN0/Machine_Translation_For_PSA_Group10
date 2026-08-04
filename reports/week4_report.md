# Week 4 report — evaluation, deployment & documentation

DSA4020A Group 10. Scope: final evaluation of the best Week 3 checkpoint,
error analysis, native-speaker human evaluation of model output, the
deployment demo, and this write-up.

## 1. Recap

Week 3 selected **NLLB-200-600M, fine-tuned on all directions with the full
~2.5k PSA-sourced Ekegusii pairs** (`ft_nllb_guz_all`) as the strongest
transfer configuration: dev BLEU 49.56 / chrF 71.94 on EN-SW, and Ekegusii
chrF 27.92 on the 138-pair held-out benchmark (see
`reports/week3_results.md`, `reports/week3_report.md`). Week 4 evaluates
this checkpoint properly (full test split, not the Week 3 ablation-matrix
sample) and turns it into a demo.

## 2. Final evaluation

Run (Kinesis node, same environment as Week 3 —
`docs/week3_kinesis_guide.md`):

```
python scripts/run_week4_eval.py --checkpoint runs/ft_nllb_guz_all/checkpoint-best
```

| Direction | n | BLEU | chrF | Repetition-flagged |
|---|---:|---:|---:|---:|
| en-sw  | 304 | 49.31 | 72.23 | 0 |
| sw-en  | 304 | 47.63 | 68.64 | 0 |
| en-guz | 138 | 3.56 | 27.92 | 23 |
| guz-en | 138 | 3.43 | 19.54 | 0 |

(n differs because only test rows with text on *both* sides score per
direction: 304 EN↔SW pairs, 138 EN↔guz pairs.)

The full-test-split numbers hold up against the Week 3 dev numbers almost
exactly: en-sw 49.31 vs 49.56 dev BLEU — drift of a third of a point on an
unseen split. The en-guz chrF of 27.92 is *numerically identical* to the
Week 3 benchmark result, which is expected and reassuring: the benchmark's
138 eng–guz pairs are exactly these test rows, so two independently written
eval pipelines (`scripts/run_eval.py` in Week 3, `scripts/run_week4_eval.py`
here) produced the same score on the same pairs — a free cross-validation
of the whole eval stack. Total runtime on the A100: ~70 seconds for all
four directions.

## 3. Error analysis

Run: `python scripts/error_analysis.py` (reads §2's predictions; no GPU
needed). Full detail in `reports/week4_error_analysis.md`.

**Per-domain.** The pattern is consistent across all four directions
(full tables in `reports/week4_error_analysis.md`); the en-guz breakdown:

| Domain | n | BLEU | chrF | Repetition-flagged |
|---|---:|---:|---:|---:|
| Agriculture | 19 | 4.03 | 28.34 | 3 |
| Education | 38 | 3.91 | 29.30 | 3 |
| Governance | 18 | 6.42 | 34.16 | 1 |
| Health | 25 | 3.07 | 24.30 | 8 |
| Security | 38 | 2.01 | 26.15 | 8 |

- **EN↔SW: Health is the weakest domain in both directions** (44.76 /
  44.39 BLEU, vs 57–62 for Agriculture/Governance). Counter-intuitive
  given Health is 57.8% of training data — but its test rows are long,
  clinical TICO-19 COVID sentences ("bilateral multilobar ground-glass
  opacities…") whose terminology is sparse in the training pairs. Reading
  the lowest-chrF examples shows most are *acceptable paraphrases* scored
  down for diverging from reference wording (e.g. "Vilabu vitatoa tamasha
  kwa ajili ya wiki ya kiraia" vs the reference's "Vilabu vyatumbuiza kwa
  wiki ya uraia"), plus a few genuine terminology failures. Short,
  formulaic PSA sentences (Agriculture, Governance) score highest —
  sentence length and terminology density, not data volume, drive the gap.
- **EN→guz: Governance strongest (34.16 chrF), Health (24.30) and
  Security (26.15) weakest** — and the repetition flags land exactly
  there (8 + 8 of 23). Reading the flagged examples confirms the heuristic:
  the flagged rows are genuine loops (`…bwobotuki bwobotuki bwobotuki…`,
  `…chiria chiria chiria…`, `…ekero ekero ekero…`), always on longer
  inputs — the model starts a well-formed Ekegusii sentence, then falls
  into a loop when it should plan a long continuation. **16.7% of en-guz
  outputs are affected.** One different failure also surfaced: a KRA tax
  PSA came out as the English source *verbatim* (copy-through, chrF 9.9) —
  the same low-confidence fallback family as Week 3's language-confusion
  finding, here expressed as giving up rather than switching language.
- **guz→en is the weakest direction overall** (chrF 19.54 vs 27.92 for
  en-guz; Security BLEU 1.0) — a real asymmetry: generating Ekegusii
  scores partial credit at character level (chrF rewards the morphology
  the model does get right), while translating *from* noisy, sparse
  Ekegusii into English requires full comprehension with no partial
  credit. No repetition flags here — loops only appear when *generating*
  the low-resource language.
- **Data-quality note:** the error analysis also catches gold noise —
  one Security row's *reference* Ekegusii itself contains English
  ("…ime yerori yao - keep them outside"). This is the known lecturer-gold
  noise discussed in Week 1/3, visible here as an artificially capped
  sentence score, not a model failure.

## 4. Human evaluation

Native-speaker rating of the model's Ekegusii output (distinct from the
Week 2 dataset-quality review) — see `docs/week4_human_eval_guide.md`.

Sheet built with:

```
python scripts/build_human_eval_sheet.py --n-per-domain 6
```

<!-- TODO: once data/validation/week4_model_output_review_reviewed_<name>.csv
     comes back — mean Fluency_1to5 / Adequacy_1to5 overall and per domain,
     and a tally of the Issues tags (mistranslation, omission, addition,
     grammar, repetition_loop, language_confusion, cultural term). -->

| Metric | Score |
|---|---:|
| Mean fluency (1-5) | TODO |
| Mean adequacy (1-5) | TODO |
| Rows with no issues flagged | TODO / TODO |

## 5. Deployment

`app.py` — a Streamlit app wrapping `training.inference.MTTranslator`
(same checkpoint auto-discovery and demo-PSA table as
`scripts/translate.py`). Two tabs: free-text translation between any two
of {English, Kiswahili, Ekegusii}, and the 8-PSA demo across all five
domains.

```
pip install -r requirements.txt -r requirements-training.txt -r requirements-app.txt
streamlit run app.py
```

<!-- TODO: screenshot(s) of the running app, once deployed / run locally. -->

## 6. Limitations

- Same core limitation as Week 3: ~2.5k Ekegusii training pairs is enough
  to demonstrate transfer, not native fluency — expect the guz directions
  to still show more issues than en-sw/sw-en in both the automatic and
  human evaluation.
- The human-evaluation sample (§4) is small and stratified by domain, not
  a full audit; treat its scores as indicative, not a certified quality
  bound.
- The repetition-loop flag (§3) is a cheap n-gram heuristic, not a
  learned classifier — but on this checkpoint every flagged example we
  read was a genuine loop, so the 23-flag count for en-guz can be treated
  as close to a true count (16.7% of outputs).
- A second, rarer failure mode surfaced in the final eval: copy-through
  (English source returned verbatim) on low-confidence inputs — related
  to Week 3's language-confusion finding.
- The guz-en direction is markedly weaker than en-guz (chrF 19.54 vs
  27.92); any deployment framing should present Ekegusii *generation* as
  the demonstrated capability, with guz→en understood as preliminary.

## 7. Conclusion

<!-- TODO: one paragraph — does the final evaluation support Week 3's
     headline claim (successful few-shot transfer to an unseen language),
     what the human evaluation adds/complicates, and what the deployment
     demo shows end-to-end. -->
