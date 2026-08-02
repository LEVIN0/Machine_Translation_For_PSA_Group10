# Week 3 ablation results

Scanned from `runs` (metrics_dev.json + evals/*.json per run).

| Run | Model | Config | Dev BLEU | Dev chrF | Guz BLEU | Guz chrF | Trainable % | Seconds |
|---|---|---|---|---|---|---|---|---|
| zs_mt5 | mt5_small | zero-shot | 0.04 | 2.36 | 0.02 | 1.46 | — | — |
| zs_nllb | nllb_600m | zero-shot | 5.57 | 12.62 | — | — | — | — |
| ft_mt5_base | mt5_small | both | 3.63 | 21.95 | — | — | 100.0 | 297 |
| ft_nllb_base | nllb_600m | both | 50.93 | 72.70 | — | — | 100.0 | 654 |
| ft_nllb_freeze | nllb_600m | both, freeze-enc | 48.75 | 71.45 | — | — | 32.8 | 601 |
| ft_nllb_guz50 | nllb_600m | all, guz=50 | 51.36 | 73.20 | 3.01 | 17.21 | 100.0 | 758 |
| ft_nllb_guz200 | nllb_600m | all, guz=200 | 51.88 | 73.53 | 3.01 | 17.24 | 100.0 | 788 |
| ft_mt5_guz200 | mt5_small | all, guz=200 | 4.09 | 22.51 | 1.77 | 11.86 | 100.0 | 323 |
| ft_nllb_guz_all | nllb_600m | all, guz=-1 | 49.56 | 71.94 | 3.56 | 27.92 | 100.0 | 1096 |
| smoke_mt5 | mt5_small | both, cap=2000 | 0.00 | 0.00 | — | — | 100.0 | 91 |
| smoke_mt5_bf16 | mt5_small | both, cap=2000 | 0.16 | 3.81 | — | — | 100.0 | 82 |
| smoke_nllb | nllb_600m | both, cap=300 | 38.95 | 62.73 | — | — | 100.0 | 86 |
| smoke_nllb_guz | nllb_600m | en-guz, guz=50, cap=50 | 1.92 | 15.56 | — | — | 100.0 | 28 |
