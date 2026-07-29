# Week 3 Colab guide — training the PSA MT models

How to run `notebooks/week3_colab.ipynb` on Google Colab's free GPU, what to
expect, and what to do when things go wrong.

## Prerequisites

- A **Google account** (for Colab).
- A **Weights & Biases account** (wandb.ai) — *optional*. Without it, runs log
  JSON to `runs/` and the results table still works; W&B is only the dashboard.
- Our repo is public: `github.com/LEVIN0/Machine_Translation_For_PSA_Group10`
  (the notebook clones it — no setup needed).

## Step by step

1. Open [colab.research.google.com](https://colab.research.google.com) →
   **File → Open notebook → GitHub** and paste the repo URL, then pick
   `notebooks/week3_colab.ipynb` (or upload the file).
2. **Runtime → Change runtime type → T4 GPU** → Save.
3. Run the cells top to bottom (**Runtime → Run all** works too):
   - Cell 1 checks the GPU — stop here if it says "no GPU detected".
   - Cell 2 clones the repo and `pip install`s `requirements.txt` +
     `requirements-training.txt` (~3–4 min).
   - Cell 3 is wandb login — **skip if you want JSON-only logs**.
   - Cell 4 builds the held-out Ekegusii benchmark (`guz_test.tsv`, from our
     own test split — **evaluation only**). FLORES-200 was evaluated and
     dropped: it contains no Ekegusii, and NLLB-200 has no `guz_Latn` token.
   - Cell 5 runs the zero-shot baselines (eval-only, ~10–15 min including
     model downloads).
   - Cell 6 group fine-tunes the matrix, **one cell per run with a timer**.
   - Cell 7 (optional) back-translates EN-only rows and trains `ft_nllb_aug`.
   - Cell 8 is the demo (`translate.py --demo`) — our success criterion.
   - Cell 9 renders the results table from `runs/`.
   - Cell 10 zips `runs/` + `reports/` for download and shows how to commit
     results back to GitHub.
4. **Download `week3_results.zip` before closing the tab** — Colab wipes the
   disk when the runtime ends.

## Expected runtimes (Colab T4)

| Step | Time |
|---|---|
| Setup (clone + pip install) | ~3–4 min |
| `build_guz_benchmark.py` | seconds |
| Zero-shot evals (`zs_mt5` + `zs_nllb`) | ~10–15 min |
| mt5-small fine-tune (3 epochs, EN↔SW) | ~20–30 min per run |
| NLLB-600M fine-tune (3 epochs) | ~35–50 min per run |
| Back-translation + `ft_nllb_aug` | ~30–60 min |
| Full standard matrix (all 9 runs) | roughly 4–6 hours |

The free tier caps sessions around 12 h (often less), so plan to split the
matrix — see below.

## GPU troubleshooting

**Runtime disconnects / "session crashed".**
Colab kills idle or long sessions. All progress lives in `runs/`, and every
notebook cell is rerunnable: after reconnecting, re-run cells 1–4 (setup) and
then the training cells you still need — `run_ablations.py` skips runs whose
`evals/*.json` already exist, so finished work is never redone.

**Out of memory (CUDA OOM).**
Halve the batch size and raise gradient accumulation so the effective batch
stays the same, e.g. for a custom run:

```bash
python scripts/run_training.py --model nllb_600m --run-name ft_nllb_base \
    --batch-size 4 --epochs 3
```

(`grad_accum` defaults to 2; if OOM persists, restart the runtime first —
**Runtime → Restart** — to free fragmented GPU memory, then re-run the cell.)

**wandb problems.**
No API key, network blocks, or login hangs: skip cell 3. The trainer falls
back to `report_to="none"` automatically and writes `metrics_dev.json` +
`evals/*.json` per run, and `write_results_table` builds the report from
those files. To silence wandb completely, set `WANDB_MODE=offline`.

**Resuming / re-using a checkpoint.**
Every run saves its best model to `runs/<run>/checkpoint-best/`. You can
evaluate or demo any checkpoint directly without retraining:

```bash
python scripts/run_eval.py --checkpoint runs/ft_nllb_base/checkpoint-best
python scripts/translate.py --checkpoint runs/ft_nllb_base/checkpoint-best --demo
```

**Splitting the matrix across sessions.**
Use `--only` to run a subset per session, e.g.:

```bash
# session 1
python scripts/run_ablations.py --only zs_mt5,zs_nllb,ft_nllb_base
# session 2
python scripts/run_ablations.py --only ft_nllb_freeze,ft_nllb_guz50
# session 3
python scripts/run_ablations.py --only ft_nllb_guz200,ft_mt5_guz200,ft_mt5_base
```

Each session's `runs/` folder must be kept (download the zip in cell 10 and
merge locally) — or commit results back to the repo after each session
(cell 10) and pull at the start of the next one. Finally, rebuild the table
over the merged `runs/` with `python scripts/run_ablations.py --table-only`.

**Slow first cell.**
The first training/eval cell downloads model weights (mt5-small ~1 GB,
NLLB-600M ~2.4 GB) — that's a one-time cost per runtime.
