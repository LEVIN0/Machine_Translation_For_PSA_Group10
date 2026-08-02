# Week 3 Report — Modeling with Transfer Learning

**Course:** DSA4020A Natural Language Processing
**Group:** 10
**Week:** 3 — Modeling with Transfer Learning (Sub-objective 2)
**Date:** 29 July – 2 August 2026

---

## 1. Overview

This week we moved from data to models. We fine-tuned two pre-trained
sequence-to-sequence models — **mT5-small** (~300M parameters) and
**NLLB-200-distilled-600M** — on our curated PSA parallel data, ran the
ablation matrix (zero-shot vs fine-tuned, layer freezing, few-shot Ekegusii
transfer at 50/200/all pairs), tracked every run in Weights & Biases, and
shipped a command-line translation demo (`scripts/translate.py`) as the
week's success criterion.

The strategic choice of these two models was deliberate: we expected
NLLB-200 to give better coverage of our language pair, while mT5 offered a
contrasting architecture. **Verified finding that shaped the whole design:**
FLORES-200 contains no Ekegusii (204 languages checked, `guz_Latn` absent)
and NLLB-200's tokenizer ships **without** a `guz_Latn` token — so *neither*
model has any Ekegusii pretraining. Every Ekegusii result below is true
unseen-language transfer: we added the missing `guz_Latn` token to NLLB,
initialised its embedding from Swahili (`swh_Latn`, a close Bantu relative),
and taught the model the language from our own PSA pairs only.

## 2. Setup

| Item | Value |
|---|---|
| Training environment | Navon Cloud / Kinesis Network (USIU grid), 1× NVIDIA A100-SXM4-80GB, 22 vCPUs, 118 GB RAM (Helsinki) |
| Framework | PyTorch 2.13 + Hugging Face Transformers 5.14 |
| Experiment tracking | Weights & Biases (project `psa-mt-group10`) + per-run JSON logs |
| Training data | PSA pairs from `data/processed/splits/train.csv` (both directions) + PSA-sourced Ekegusii pairs from the same train split (capped per run: 0 / 50 / 200 / all) |
| Evaluation | sacreBLEU + chrF on PSA dev (EN↔SW) and the held-out Ekegusii benchmark `guz_test.tsv` (138 EN–guz pairs built from the test split, never used in training) |
| Seed | 42 everywhere |

## 3. Models and hyperparameters

| Hyperparameter | mT5-small | NLLB-200-distilled-600M |
|---|---|---|
| HF checkpoint | `google/mt5-small` | `facebook/nllb-200-distilled-600M` |
| Parameters | ~300M | ~600M |
| Learning rate | 1e-4 | 5e-5 |
| Batch size × grad-accum | 16 × 2 | 8 × 2 |
| Epochs | 3 | 3 |
| Max sequence length | 128 | 128 |
| Precision | **bf16** (see §8) | fp16 |
| Early selection | best dev sacreBLEU | best dev sacreBLEU |
| Ekegusii handling | none (task-prefix trick only) | `guz_Latn` token added + embedding donor-initialised from `swh_Latn` |

Low-resource techniques used: **encoder freezing** (ablation run),
**few-shot capping** (50 / 200 / all PSA-sourced Ekegusii pairs). The planned
back-translation augmentation run was skipped by the matrix runner because
the augmented CSV had not been generated in time — deferred to future work
(§9).

## 4. Runs and training time

All runs on one A100-80GB; total matrix wall time ≈ 80 minutes of training
(plus evaluation and smoke runs). Every run is logged in W&B project
[`psa-mt-group10`](https://wandb.ai/clairewangui02-university-system-of-georgia/psa-mt-group10).

| Run | Config summary | Trainable % | Wall time (s) |
|---|---|---:|---:|
| zs_mt5 | mT5 zero-shot (eval only) | — | — |
| zs_nllb | NLLB zero-shot (eval only) | — | — |
| ft_mt5_base | mT5 fine-tune, EN↔SW | 100 | 297 |
| ft_nllb_base | NLLB fine-tune, EN↔SW | 100 | 654 |
| ft_nllb_freeze | NLLB + frozen encoder | 32.8 | 601 |
| ft_nllb_guz50 | NLLB, all directions + 50 guz pairs | 100 | 758 |
| ft_nllb_guz200 | NLLB, all directions + 200 guz pairs | 100 | 788 |
| ft_mt5_guz200 | mT5, all directions + 200 guz pairs | 100 | 323 |
| ft_nllb_guz_all | NLLB, all directions + **all** guz pairs | 100 | 1,096 |

## 5. Results

From `reports/week3_results.md` (auto-generated from run logs; smoke runs
omitted here):

| Run | Dev BLEU | Dev chrF | Guz BLEU | Guz chrF |
|---|---:|---:|---:|---:|
| zs_mt5 | 0.04 | 2.36 | 0.02 | 1.46 |
| zs_nllb | 5.57 | 12.62 | — | — |
| ft_mt5_base | 3.63 | 21.95 | — | — |
| ft_nllb_base | **50.93** | **72.70** | — | — |
| ft_nllb_freeze | 48.75 | 71.45 | — | — |
| ft_nllb_guz50 | 51.36 | 73.20 | 3.01 | 17.21 |
| ft_nllb_guz200 | 51.88 | **73.53** | 3.01 | 17.24 |
| ft_mt5_guz200 | 4.09 | 22.51 | 1.77 | 11.86 |
| ft_nllb_guz_all | 49.56 | 71.94 | **3.56** | **27.92** |

*(Dev = PSA dev split, EN↔SW combined; Guz = 138-pair held-out Ekegusii
benchmark. NLLB zero-shot guz is undefined — the base tokenizer has no
`guz_Latn` to evaluate.)*

## 6. What the ablations showed

1. **Zero-shot vs fine-tuned:** domain adaptation is everything. NLLB jumps
   from 5.57 to 50.93 dev BLEU (+45.4) after fine-tuning on our PSA data;
   mT5 goes from 0.04 to 3.63. The PSA register is far from generic
   pretraining text, and three epochs on 6k pairs closes most of that gap
   for NLLB.
2. **NLLB ≫ mT5 on this task.** Fine-tuned mT5-small reaches only 3.63
   dev BLEU where NLLB reaches 50.93 — NLLB's many-to-many translation
   pretraining transfers to our pair far better than mT5's span-corruption
   objective, even though both models "cover" English and Swahili.
3. **Ekegusii few-shot transfer works, and scales with data.** Adding guz
   pairs *improves* EN↔SW slightly (50.93 → 51.88 at guz200 — regularisation
   from multi-directional training) while unlocking Ekegusii translation.
   The guz chrF scaling curve is the headline: **1.46 (zero-shot floor) →
   17.21 (50 pairs) → 17.24 (200) → 27.92 (all ~2.5k pairs)** — a 62% jump
   from using our full scraped + lecturer-gold Ekegusii data, and direct
   evidence that Week 1's data collection is the lever that matters.
4. **Freezing the encoder** costs 2.2 BLEU (50.93 → 48.75) while training
   only 32.8% of parameters — a viable parameter-efficient option when
   compute is constrained.
5. **Language confusion is real at very low data scales.** With only 200
   guz pairs, the model produced fluent *Swahili* when asked for Ekegusii
   (the donor language dominates). With all guz pairs it produces genuinely
   Ekegusii-flavoured morphology (see §7) — fluency is bounded by data
   scale, not by the transfer mechanism.

## 7. Demo (success criterion)

`python scripts/translate.py --checkpoint runs/ft_nllb_guz_all/checkpoint-best --demo`
translates 8 sample PSAs across all five domains (EN↔SW, EN→guz, SW→guz).
Sample outputs:

| Direction | Source | Model output |
|---|---|---|
| EN→SW | Wash your hands with soap and clean water for at least twenty seconds. | Osha mikono yako kwa sabuni na maji safi kwa angalau sekunde ishirini. |
| SW→EN | Jitokeze kupiga kura; sauti yako ndiyo nguvu ya mabadiliko. | Emerge to vote; your voice is the force of change. |
| SW→guz | Jitokeze kupiga kura; sauti yako ndiyo nguvu ya mabadiliko. | Kabe omoroberio bwokoruta chibesa; ensemo yao nigo endagera yogoikerania. |

The Ekegusii output shows genuine Ekegusii noun-class morphology
(**omo-**roberio, **bwo-**koruta, **chi-**besa — prefixes absent from
Swahili), which is the key qualitative result: the model learned a language
it never saw in pretraining. Remaining issues at this data scale: occasional
repetition loops and uncertain word choices on longer sentences.

## 8. Challenges

- **mT5 + fp16 = NaN.** Our first mT5 runs produced `loss=0,
  grad_norm=nan` from step 1: mT5's activations overflow fp16 (Google
  trained it in bf16). Fix: per-model precision defaults — mT5 trains in
  bf16 (A100-native), NLLB keeps fp16 (`training/config.py`,
  `--precision` flag). The NaN run is preserved in the results table
  (`smoke_mt5`, 0.00/0.00) as evidence.
- **Bleeding-edge dependency stack.** The fresh environment pulled
  transformers 5.14 / datasets 5.0.1 / pandas 3.0.5 / numpy 2.5.1, and
  `datasets` fingerprint pickling crashes on pyarrow ≥ 21 + numpy ≥ 2.4
  (dill recurse-mode serialises module namespaces by value and dies on
  pyarrow's `MonthDayNano` and numpy's `*_with_like` helpers). Fixed with
  a small shim forcing reference-mode pickling in `training/data.py`; the
  full 44-test suite passes on the exact production stack.
- **Environment is ephemeral.** The Kinesis container loses all
  non-committed state on restart, so results live in three places: git
  (code + reports), W&B (metrics + a results-archive artifact), and a
  downloaded results archive.
- **Tiny Ekegusii corpus.** ~2.5k guz training pairs is enough to
  demonstrate transfer but not fluency — every low-resource lever
  (donor initialisation, few-shot scaling, multi-directional training)
  was necessary, not decorative.

## 9. Next steps (Week 4)

- Final evaluation on the held-out PSA test split + full guz benchmark
- Human evaluation of Ekegusii output using the Week 2 validation subset
  (fluency/adequacy by a native speaker)
- Error analysis per domain (repetition loops on long guz sentences first)
- Back-translation augmentation run (planned, deferred)
- Deployment: wrap `MTTranslator` in the demo web app
