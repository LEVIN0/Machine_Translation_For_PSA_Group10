# Week 3 training runbook — Navon Cloud / Kinesis Network

How Week 3 training was actually run: a GPU app on the **Navon Cloud
pilot (Kinesis Network)**, USIU grid. Node specs: **1× NVIDIA
A100-SXM4-80GB**, 22 vCPUs, ~118 GB RAM, ~1 TB storage (Helsinki).

This replaces the earlier Colab plan (superseded — see `SPEC_WEEK3.md`).

## 0. The two rules that will bite you

1. **GPU must be requested at app creation time.** When creating the app
   (code-server gallery image), the creation wizard's *Resources* step must
   have **Accelerator: A100, Count: 1**. Quick Launch defaults to **0 GPUs**,
   and you cannot add a GPU afterwards in the Config UI — if you forgot,
   delete the app and create it again properly.
2. **The container filesystem is ephemeral.** Restarting/recreating the app
   wipes everything that is not in git, W&B, or downloaded. Export results
   *before* stopping anything (§6).

## 1. First-time environment setup

The code-server image is bare — no Python, no tmux. In the code-server
terminal (`` Ctrl+` `` or Terminal → New Terminal):

```bash
apt-get update && apt-get install -y python3 python3-venv python3-pip tmux zip
git clone https://github.com/LEVIN0/Machine_Translation_For_PSA_Group10.git
cd ~/Machine_Translation_For_PSA_Group10
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-training.txt   # ~10 min (torch is ~2.5 GB)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
# expect: True NVIDIA A100-SXM4-80GB
```

## 2. Weights & Biases

```bash
wandb login          # paste API key from https://wandb.ai/authorize
```

All runs then land in project **`psa-mt-group10`** automatically
(`report_to="wandb"` is the default; without a login the trainer falls back
to JSON logs and prints a notice).

## 3. Sanity checks (run these first, every fresh container)

```bash
python tests/test_smoke.py            # expect: ALL SMOKE TESTS PASSED
python scripts/build_guz_benchmark.py # expect: 138 eng-guz pairs
```

## 4. Training

Smoke runs first (~2 min each, models already cached after the first):

```bash
python scripts/run_training.py --model mt5_small --run-name smoke_mt5 --quick
python scripts/run_training.py --model nllb_600m --run-name smoke_nllb --max-samples 300 --epochs 1
```

Then the full ablation matrix inside **tmux** (so a browser disconnect
doesn't kill training):

```bash
tmux new -s train
source .venv/bin/activate      # venv activation is per-shell — tmux panes are new shells!
python scripts/run_ablations.py --matrix standard 2>&1 | tee runs/ablation_log.txt
# detach: Ctrl+B then D   ·   reattach: tmux attach -t train
```

The matrix is resumable: completed runs are skipped on rerun
(`--no-skip-existing` to force). Notes:

- `zs_*` runs are eval-only (no training).
- `ft_nllb_aug` skips itself if `data/processed/augmented.csv` doesn't exist.
- Mixed precision is per-model by default: **mT5 trains in bf16** (fp16
  overflows to NaN gradients), **NLLB in fp16**. Override with
  `--precision {fp16,bf16,fp32}`.

## 5. Evaluation, results table, demo

```bash
python scripts/run_ablations.py --matrix standard --table-only  # writes reports/week3_results.md
python scripts/run_eval.py --checkpoint runs/<run>/checkpoint-best
python scripts/translate.py --checkpoint runs/<run>/checkpoint-best --demo
```

## 6. Export results BEFORE stopping the app

```bash
tar -czf week3_results.tar.gz reports/week3_results.md runs/ablation_log.txt \
    runs/*/metrics*.json runs/*/train_config.json runs/*/evals/*.json
```

Then either download it via the code-server file explorer (right-click →
Download; **refresh the browser tab** if new files don't show) or push it to
W&B as an artifact:

```bash
python -c "
import wandb
run = wandb.init(project='psa-mt-group10', name='results-archive', job_type='export')
art = wandb.Artifact('week3-results', type='export'); art.add_file('week3_results.tar.gz')
run.log_artifact(art); run.finish()"
```

The trained checkpoints (multi-GB) stay on the container; the small metrics
files + W&B are what the report needs.

## 7. Troubleshooting

| Symptom | Cause → fix |
|---|---|
| `python: command not found` | venv not active in *this* shell (`source .venv/bin/activate`) — remember tmux panes are new shells |
| `nvidia-smi: command not found` + `torch.cuda.is_available()` False | app was created with 0 GPUs → recreate it with A100 ×1 (§0) |
| `wandb` falls back to `report_to='none'` | not logged in → `wandb login` |
| training loss = 0, `grad_norm = nan` (mT5) | fp16 overflow — fixed in the repo (mT5 defaults to bf16) |
| new files invisible in the file explorer | code-server quirk — refresh the **browser tab** (F5) |
| `zip: command not found` | bare image — `apt-get install -y zip` or use `tar` |
| `tmux attach` says "sessions should be nested" | you're already inside tmux — no need to attach |
| HF download hangs | check `curl -sI -m 15 https://huggingface.co \| head -1`; the suite's model tests skip gracefully if the hub is unreachable |
