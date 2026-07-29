# Week 3 Report — Modeling with Transfer Learning

**Course:** DSA4020A Natural Language Processing
**Group:** 10
**Week:** 3 — Modeling with Transfer Learning (Sub-objective 2)
**Date:** 29th of July 2026

> **How to use this template:** every `{{PLACEHOLDER}}` gets replaced with the
> measured value after the Colab runs. `reports/week3_results.md` is generated
> automatically from the run logs (`write_results_table`) — paste its table in
> §5. Delete this note before submission.

---

## 1. Overview

This week we moved from data to models. We fine-tuned two pre-trained
sequence-to-sequence models — **mT5-small** (~300M parameters) and
**NLLB-200-distilled-600M** — on our curated PSA parallel data, ran the
planned ablation matrix (zero-shot vs fine-tuned, layer freezing, few-shot
Ekegusii transfer, back-translation augmentation), tracked every run in
Weights & Biases, and shipped a command-line translation demo
(`scripts/translate.py`) as the week's success criterion.

The strategic choice of these two models is deliberate: NLLB-200 includes
Ekegusii (`guz_Latn`) among its 200 languages, while mT5 does not — so the pair
lets us measure few-shot transfer to Ekegusii both *with* and *without*
pretraining exposure to the language.

## 2. Setup

| Item | Value |
|---|---|
| Training environment | Google Colab (free tier), GPU: {{GPU_NAME, e.g. Tesla T4}} |
| Framework | PyTorch + Hugging Face Transformers |
| Experiment tracking | Weights & Biases (project `psa-mt-group10`) + per-run JSON logs |
| Training data | PSA pairs from `data/processed/splits/train.csv` (paired rows, both directions) + FLORES-200 `guz_Latn` **dev** seed for few-shot runs |
| Evaluation | sacreBLEU + chrF on PSA dev (EN→SW, SW→EN) and FLORES-200 `guz_Latn` **devtest** (EN→guz, guz→EN; devtest never used in training) |
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
| Precision | fp16 | fp16 |
| Early selection | best dev sacreBLEU | best dev sacreBLEU |
| Language coverage relevant to us | EN, SW (no Ekegusii) | EN, SW, **Ekegusii (`guz_Latn`)** |

Low-resource techniques used: **encoder freezing** (ablation run), **few-shot
seed injection** (50/200 FLORES dev pairs), and **back-translation
augmentation** of English-only PSA rows (ablation run).

## 4. Runs and training time

| Run | Config summary | Train pairs | Trainable % | Wall time | W&B link |
|---|---|---:|---:|---:|---|
| zs_mt5 | zero-shot, no training | — | — | — | {{link}} |
| zs_nllb | zero-shot, no training | — | — | — | {{link}} |
| ft_mt5_base | fine-tune, EN↔SW | {{n}} | 100% | {{mm:ss}} | {{link}} |
| ft_nllb_base | fine-tune, EN↔SW | {{n}} | 100% | {{mm:ss}} | {{link}} |
| ft_nllb_freeze | + frozen encoder | {{n}} | {{pct}} | {{mm:ss}} | {{link}} |
| ft_nllb_aug | + back-translation pairs | {{n}} | 100% | {{mm:ss}} | {{link}} |
| ft_nllb_guz50 | + 50 Ekegusii seed pairs | {{n}} | 100% | {{mm:ss}} | {{link}} |
| ft_nllb_guz200 | + 200 Ekegusii seed pairs | {{n}} | 100% | {{mm:ss}} | {{link}} |
| ft_mt5_guz200 | mT5 + 200 Ekegusii seed pairs | {{n}} | 100% | {{mm:ss}} | {{link}} |

## 5. Preliminary results

*(Auto-generated table from `reports/week3_results.md` goes here.)*

| Run | PSA dev EN→SW BLEU/chrF | PSA dev SW→EN BLEU/chrF | FLORES EN→guz BLEU/chrF | FLORES guz→EN BLEU/chrF |
|---|---|---|---|---|
| zs_mt5 | {{/}} | {{/}} | {{/}} | {{/}} |
| zs_nllb | {{/}} | {{/}} | {{/}} | {{/}} |
| ft_mt5_base | {{/}} | {{/}} | {{/}} | {{/}} |
| ft_nllb_base | {{/}} | {{/}} | {{/}} | {{/}} |
| ft_nllb_freeze | {{/}} | {{/}} | — | — |
| ft_nllb_aug | {{/}} | {{/}} | — | — |
| ft_nllb_guz50 | {{/}} | {{/}} | {{/}} | {{/}} |
| ft_nllb_guz200 | {{/}} | {{/}} | {{/}} | {{/}} |
| ft_mt5_guz200 | {{/}} | {{/}} | {{/}} | {{/}} |

## 6. What the ablations showed

1. **Zero-shot vs fine-tuned:** {{2-3 sentences — the domain-adaptation gain
   from fine-tuning on PSA data vs the zero-shot baseline.}}
2. **Ekegusii few-shot transfer:** {{2-3 sentences — NLLB at 0/50/200 seed
   pairs; whether mT5 (no Ekegusii pretraining) moves at all; chrF is the
   headline metric here.}}
3. **Freezing:** {{1-2 sentences — quality vs speed/trainable-params tradeoff.}}
4. **Back-translation:** {{1-2 sentences — did synthetic pairs help dev BLEU.}}

## 7. Demo (success criterion)

`python scripts/translate.py --demo` translates 8 sample PSAs across all five
domains (EN↔SW, EN→guz, SW→guz). Sample outputs:

| Source | Model output |
|---|---|
| {{sample EN PSA}} | {{SW / guz translation}} |
| {{sample SW PSA}} | {{EN translation}} |

## 8. Challenges

- **Colab free-tier limits:** {{session timeouts / VRAM — what actually happened
  and how you mitigated (smaller batch + grad-accum, splitting the matrix
  across sessions, resuming from checkpoint-best).}}
- **Tiny parallel corpus:** only ~3.1k EN–SW pairs and a 997-sentence Ekegusii
  seed — every low-resource lever (freezing, few-shot, augmentation) was
  necessary, not decorative.
- **W&B on Colab:** {{any API-key/offline issues; JSON fallback logs.}}

## 9. Next steps (Week 4)

- Final evaluation on the held-out PSA test split + full FLORES devtest
- Human evaluation using the Week 2 validation subset (fluency/adequacy)
- Error analysis per domain
- Deployment: wrap `MTTranslator` in the demo web app
