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

<!-- TODO: paste the overall table printed by run_week4_eval.py / from
     reports/week4_eval/summary.json -->

| Direction | n | BLEU | chrF | Repetition-flagged |
|---|---:|---:|---:|---:|
| en-sw  | TODO | TODO | TODO | TODO |
| sw-en  | TODO | TODO | TODO | TODO |
| en-guz | TODO | TODO | TODO | TODO |
| guz-en | TODO | TODO | TODO | TODO |

<!-- TODO: one or two sentences comparing these full-test-split numbers to
     the Week 3 dev-set numbers above (expect some drift; test is a
     different, unseen split). -->

## 3. Error analysis

Run: `python scripts/error_analysis.py` (reads §2's predictions; no GPU
needed). Full detail in `reports/week4_error_analysis.md`.

<!-- TODO: per-domain summary — which domain is weakest per direction and
     why (data volume? sentence length? domain-specific vocabulary?). Pull
     the per-domain table for at least en-guz from
     reports/week4_error_analysis.md. -->

<!-- TODO: qualitative read of the worst-chrF and repetition-flagged
     examples — confirm or reject the repetition-loop heuristic by ear,
     note any other recurring failure pattern (language confusion,
     truncation, glossary-term mishandling). -->

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
- The repetition-loop flag (§3) is a cheap n-gram heuristic, not a
  learned classifier — treat its output as a shortlist for a human read.
- The human-evaluation sample (§4) is small and stratified by domain, not
  a full audit; treat its scores as indicative, not a certified quality
  bound.
<!-- TODO: add anything else that surfaces once §2-§4 are actually run. -->

## 7. Conclusion

<!-- TODO: one paragraph — does the final evaluation support Week 3's
     headline claim (successful few-shot transfer to an unseen language),
     what the human evaluation adds/complicates, and what the deployment
     demo shows end-to-end. -->
