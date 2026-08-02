# SPEC_WEEK3.md — Modeling with Transfer Learning (single source of truth)

Project: DSA4020A Group 10 — PSA Machine Translation (EN/SW ↔ Ekegusii).
Repo layout this spec ADDS to the existing project (existing Week 1/2 code is
untouched except `tests/test_smoke.py` wiring, done by the main agent at merge).

Decisions (locked with the team): training on **Google Colab free GPU**,
tracking via **Weights & Biases**. Ekegusii data: **PSA gold pairs only** —
the FLORES-200 seed/benchmark plan was dropped after verification (§8):
FLORES-200 contains no `guz_Latn` and NLLB-200 has no `guz_Latn` token.

> **Update (executed):** training actually ran on a **Navon Cloud / Kinesis
> A100-80GB** node (USIU grid) via the CLI scripts below — the Colab
> notebook (§5) was dropped and the Colab guide (§6) was replaced by
> `docs/week3_kinesis_guide.md`. One further deviation: mT5 required **bf16**
> (fp16 overflows to NaN gradients), so precision became a per-model
> `MODEL_ZOO` default (§2.1). The back-translation run (`ft_nllb_aug`) was
> skipped by the runner (no augmented CSV generated in time). An extra
> `ft_nllb_guz_all` run (all PSA guz pairs) was added after the 200-pair run
> exhibited donor-language confusion. Results: `reports/week3_report.md`.
The Ekegusii benchmark is our own held-out test split (evaluation-only —
project rule).

## 1. New files and ownership

| File | Owner | Purpose |
|---|---|---|
| `training/__init__.py` | A | package marker, exports version |
| `training/config.py` | A | ModelConfig / TrainConfig dataclasses, MODEL_ZOO, LANGS |
| `training/data.py` | A | splits → paired HF datasets; guz benchmark loading |
| `training/augment.py` | A | back-translation: EN-only rows → synthetic EN–SW pairs |
| `training/lang_tokens.py` | main | NLLB unseen-language extension (adds `guz_Latn`, donor-init from `swh_Latn`) |
| `scripts/build_guz_benchmark.py` | main | build `guz_test.tsv` from the held-out test split |
| `tests/fixtures/guz_benchmark/guz_test.tsv` | A | tiny benchmark fixture (8 lines) |
| `tests/test_week3_data.py` | A | offline tests for config/data/augment |
| `training/utils.py` | B | seeding, device info, run dirs, timing, JSON IO |
| `training/trainer.py` | B | `train()` — HF Seq2SeqTrainer wiring, freezing, W&B, checkpoints |
| `scripts/run_training.py` | B | CLI for one training run |
| `requirements-training.txt` | B | torch, transformers, datasets, accelerate, sentencepiece, wandb, sacrebleu, protobuf |
| `tests/test_week3_trainer.py` | B | tiny-model train step (skip-graceful if offline) |
| `training/evaluate.py` | C | sacreBLEU + chrF evaluation of a checkpoint |
| `training/inference.py` | C | MTTranslator + DEMO_PSAS |
| `training/ablate.py` | C | ablation matrix, runner, results-table writer |
| `scripts/run_eval.py` | C | CLI: evaluate checkpoint(s) |
| `scripts/translate.py` | C | CLI demo (success criterion) |
| `scripts/run_ablations.py` | C | CLI: run the matrix |
| ~~`notebooks/week3_colab.ipynb`~~ (dropped) | C | one-click Colab runbook |
| `docs/week3_kinesis_guide.md` | C | Kinesis training guide incl. GPU troubleshooting (replaces the Colab guide) |
| `tests/test_week3_eval.py` | C | metrics + inference + matrix tests (skip-graceful) |

No two agents edit the same file. Shared contracts below are frozen — implement
exactly, no unilateral changes. Main agent merges branches and wires
`tests/test_smoke.py` to discover `tests/test_week3_*.py`.

## 2. Frozen interface contracts

### 2.1 `training/config.py`
```python
LANGS = {"eng": "English", "swa": "Swahili", "guz": "Ekegusii"}
NLLB_CODES = {"eng": "eng_Latn", "swa": "swh_Latn", "guz": "guz_Latn"}

@dataclass
class ModelConfig:
    key: str            # "mt5_small" | "nllb_600m"
    hf_name: str        # "google/mt5-small" | "facebook/nllb-200-distilled-600M"
    family: str         # "mt5" | "nllb"
    lr: float           # default lr: mt5_small 3e-4? NO -> 1e-4 ; nllb_600m 5e-5
    batch_size: int     # mt5_small 16 ; nllb_600m 8
    max_length: int = 128
    precision: str = "fp16"  # mt5_small "bf16" (fp16 -> NaN grads) ; nllb_600m "fp16"

MODEL_ZOO: dict[str, ModelConfig]  # the two entries above

@dataclass
class TrainConfig:
    run_name: str
    model_key: str
    direction: str = "both"     # "en-sw"|"sw-en"|"en-guz"|"sw-guz"|"both"|"all"
    fewshot_guz: int = 0        # cap on PSA-sourced guz train pairs (0 = none, -1 = all)
    use_augmentation: bool = False
    freeze_encoder: bool = False
    freeze_embed: bool = False
    epochs: float = 3.0
    lr: float | None = None     # None -> MODEL_ZOO default
    batch_size: int | None = None
    grad_accum: int = 2
    max_samples: int | None = None   # cap training pairs (smoke/quick runs)
    fp16: bool = True     # legacy: superseded by `precision`; kept for old config files
    precision: str | None = None     # None -> MODEL_ZOO default ("fp16"|"bf16"|"fp32")
    seed: int = 42
    output_root: str = "runs"
    report_to: str = "wandb"    # "wandb" | "none"
    wandb_project: str = "psa-mt-group10"
    def resolved(self, zoo=MODEL_ZOO) -> "TrainConfig": ...  # fills lr/batch_size/precision
    def to_json(self, path) / @classmethod from_json(path)
```
Direction expansion (single place, in `data.py`): `"both"` -> `["en-sw","sw-en"]`;
`"all"` -> `["en-sw","sw-en","en-guz","sw-guz"]`; single codes stay as-is.

### 2.2 `training/data.py`
Canonical example columns (HF `datasets.Dataset`): 
`src_text:str, tgt_text:str, src_lang:str, tgt_lang:str, domain:str, provenance:str`
(`src_lang`/`tgt_lang` in {"eng","swa","guz"}; provenance in
{"psa","backtranslation"}).
```python
def load_psa_pairs(splits_dir: Path, split: str, directions: list[str]) -> Dataset
    # reads data/processed/splits/{split}.csv; keeps rows with Kiswahili != "";
    # "en-sw" row + (if requested) "sw-en" row per pair; domain carried; provenance="psa"
def load_guz_benchmark(benchmark_dir: Path, n: int | None = 200, seed: int = 42) -> Dataset
    # reads guz_test.tsv ONLY (evaluation-only, never train); deterministic
    # seeded sample of n; columns: eng, guz (raw texts) -> normalized to
    # src/tgt by evaluate.py
def build_train_dataset(cfg: TrainConfig, splits_dir: Path,
                        augmented_csv: Path | None = None) -> Dataset
    # psa pairs (train split) + PSA-sourced guz pairs gated by cfg.fewshot_guz
    # + optional augmented_csv rows (provenance="backtranslation");
    # applies cfg.max_samples cap AFTER concatenation with seeded shuffle;
    # HARD ASSERT: raise if fewshot_guz>0 but the train split has no Ekegusii
    # (the guz benchmark is built from the test split — never trainable)
```
TSV canonical format: header `eng\tguz`, UTF-8, one sentence pair per line.
`scripts/build_guz_benchmark.py` reads `data/processed/splits/test.csv`,
keeps rows with non-empty English AND Ekegusii, and writes
`data/external/guz_benchmark/guz_test.tsv` (138 pairs on the final
dataset). Prints the pair count; missing split or zero pairs -> clear error,
exit 1. The output is EVALUATION-ONLY.

### 2.3 `training/augment.py`
```python
def backtranslate(model_ckpt: str, splits_dir: Path, out_csv: Path,
                  max_rows: int = 3000, batch_size: int = 32, seed: int = 42) -> Path
    # takes ENGLISH-ONLY rows from the TRAIN split, translates eng->swa with the
    # given checkpoint via training/inference.MTTranslator, writes CSV with the
    # dataset schema columns (Source="Back-translation", Status="Synthetic",
    # Metadata={"type":"backtranslation","model":<ckpt>}) plus nothing else;
    # rows failing generation are skipped. Deterministic: seeded row order.
```
Module must be importable without torch installed (import torch lazily inside
functions; same pattern everywhere in the package).

### 2.4 `training/utils.py`
```python
def set_seed(seed: int = 42) -> None            # random/np/torch if available
def device_info() -> dict                       # {"device": "cuda|cpu", "gpu_name": str|None, "torch": str|None}
def run_dir(cfg: TrainConfig) -> Path           # runs/<run_name>, mkdirs, returns
def save_json(obj, path) / load_json(path)
class Timer:  # context manager, .seconds float
```

### 2.5 `training/trainer.py`
```python
def train(cfg: TrainConfig, splits_dir: Path = Path("data/processed/splits"),
          augmented_csv: Path | None = None) -> Path
```
Behaviour:
1. `cfg = cfg.resolved()`; `set_seed(cfg.seed)`; build dataset via `data.build_train_dataset`.
2. Tokenize per family (see §3). HF `Seq2SeqTrainer`, `Seq2SeqTrainingArguments`:
   `output_dir=run_dir(cfg)`, `eval_strategy="epoch"`, `save_strategy="epoch"`,
   `load_best_model_at_end=True`, `metric_for_best_model="sacrebleu"`,
   `predict_with_generate=True`,
   mixed precision from `cfg.precision` (resolved per model: mT5 `bf16` — fp16
   overflows to NaN grads; NLLB `fp16`; bf16 falls back to fp32 if unsupported),
   `learning_rate`, `num_train_epochs=cfg.epochs`,
   `per_device_train_batch_size`, `gradient_accumulation_steps=cfg.grad_accum`,
   `report_to="wandb" if cfg.report_to=="wandb" and WANDB_API_KEY set else "none"`,
   `run_name=cfg.run_name`, `seed=cfg.seed`, `logging_steps=20`,
   `save_total_limit=2`. W&B project = cfg.wandb_project; never crash when wandb
   is absent or offline (fallback "none" + print).
3. Freezing (low-resource techniques): `freeze_encoder` -> `requires_grad_(False)`
   on `model.get_encoder()`; `freeze_embed` -> on shared embeddings.
   Log trainable-param % to stdout and W&B summary.
4. `compute_metrics`: sacrebleu corpus BLEU (+ chrF in the same dict, key "chrf";
   metric_for_best_model stays "sacrebleu").
5. After training: save best to `run_dir/checkpoint-best/` (model + tokenizer),
   write `train_config.json` (cfg.to_json), `metrics_dev.json`
   (final eval metrics + n_train_pairs + trainable_pct + seconds + device_info).
   Returns the `checkpoint-best` Path.
Must be CPU-runnable for a 1-step smoke run with a tiny local model (see tests).

### 2.6 `training/evaluate.py`
```python
EVAL_SPECS = {  # name -> (loader, src, tgt)
  "psa_dev_en-sw":  ("psa", "dev",  "eng", "swa"),
  "psa_dev_sw-en":  ("psa", "dev",  "eng", "swa"),   # reversed at load
  "psa_test_en-sw": ("psa", "test", "eng", "swa"),
  "psa_test_sw-en": ("psa", "test", "eng", "swa"),
  "psa_test_en-guz": ("guzbench", "test", "eng", "guz"),
  "psa_test_guz-en": ("guzbench", "test", "guz", "eng"),
}
def evaluate_checkpoint(ckpt: str | Path, eval_spec: str, n: int | None = 200,
                        batch_size: int = 16, seed: int = 42,
                        out_dir: Path | None = None) -> dict
    # -> {"eval_spec":..., "ckpt": str, "n": int, "bleu": float, "chrf": float,
    #     "seconds": float, "model_key": str|None}
    # uses inference.MTTranslator for generation; sacrebleu for metrics;
    # if out_dir: writes <out_dir>/<eval_spec>.json
```
PSA eval reads `data/processed/splits/<split>.csv` (paired rows only).
Guz benchmark eval reads `data/external/guz_benchmark/guz_test.tsv` via
`data.load_guz_benchmark` (built from the held-out test split).

### 2.7 `training/inference.py`
```python
class MTTranslator:
    def __init__(self, checkpoint: str | Path, model_key: str | None = None):
        # model_key None -> read runs/<...>/train_config.json if present, else
        # sniff config.json (model_type "t5"->mt5 family; "nllb"/m2m100 -> nllb)
    def translate(self, texts: list[str], src: str, tgt: str,
                  max_length: int = 128, num_beams: int = 4) -> list[str]
        # src/tgt in {"eng","swa","guz"}; batching; family-specific encoding (§3)
DEMO_PSAS: list[dict]  # exactly 8 dicts: {"domain":..., "src": "eng"|"swa", "text":...}
    # >=1 per each of the 5 domains; short realistic PSA lines; original text.
```
Import-time must not require torch (lazy imports in __init__/translate).

### 2.8 `training/ablate.py`
```python
def standard_matrix(quick: bool = False) -> list[TrainConfig]
```
Matrix (run_name, model, notes) — `quick=True` caps `max_samples=2000, epochs=2`:
1. `zs_mt5` / `zs_nllb` — zero-shot: NO training, eval-only entries (flag via
   TrainConfig with `epochs=0`; ablate runner handles by calling evaluate only)
2. `ft_mt5_base` — direction "both"
3. `ft_nllb_base` — direction "both"
4. `ft_nllb_freeze` — + freeze_encoder
5. `ft_nllb_aug` — + use_augmentation (needs augmented csv; runner warns+skips if absent)
6. `ft_nllb_guz50` — direction "all", fewshot_guz=50
7. `ft_nllb_guz200` — direction "all", fewshot_guz=200
8. `ft_mt5_guz200` — direction "all", fewshot_guz=200 (transfer comparison)
```python
def write_results_table(runs_root: Path, out_md: Path) -> Path
    # scans runs/*/metrics_dev.json + runs/*/evals/*.json -> markdown table
    # (run, model, config flags, dev BLEU/chrF, guz BLEU/chrF, trainable %,
    #  seconds) -> reports/week3_results.md ; missing values rendered "—"
```

### 2.9 CLIs (argparse, `python scripts/<x>.py --help` self-explanatory)
- `run_training.py --model mt5_small --run-name ft_mt5_base [--direction both]
  [--fewshot-guz 0] [--freeze-encoder] [--use-augmentation] [--epochs 3]
  [--max-samples N] [--report-to none] [--quick]`
- `run_eval.py --checkpoint runs/x/checkpoint-best [--specs psa_dev_en-sw,psa_test_en-guz] [--n 200]`
- `run_ablations.py [--matrix standard|quick] [--only zs_nllb,ft_nllb_base]`
- `translate.py --checkpoint runs/x/checkpoint-best [--text "..."] [--src eng]
  [--tgt swa] [--demo] [--interactive]` — `--demo` translates DEMO_PSAS into all
  applicable targets (eng↔swa + eng→guz, swa→guz), prints a clean table.
  No `--checkpoint` + `--demo` -> auto-discover newest runs/*/checkpoint-best.

## 3. Family-specific tokenization (frozen)
- **mt5** (`google/mt5-small`, T5Tokenizer): input = 
  `f"translate {LANGS[src]} to {LANGS[tgt]}: {text}"`; labels = target text;
  generation: same prefix, no forced BOS.
- **nllb** (`facebook/nllb-200-distilled-600M`): `tokenizer.src_lang = NLLB_CODES[src]`;
  labels tokenized with `tokenizer(text_target=...)` after
  `tokenizer.tgt_lang = NLLB_CODES[tgt]` (or `as_target_tokenizer` ctx);
  generation: `forced_bos_token_id=tokenizer.convert_tokens_to_ids(NLLB_CODES[tgt])`
  (fallback `tokenizer.lang_code_to_id[...]` for older transformers — handle both).
Both: `tokenizer(text=..., text_target=...)` pattern, truncation at
`model_cfg.max_length`, `DataCollatorForSeq2Seq`.

## 4. Non-negotiables
- **The guz benchmark (`guz_test.tsv`, from the test split) never in training**
  — assert in build_train_dataset.
- Seeded everything (seed=42 default); rerunnable one-command flows.
- No torch/transformers import at package import time (lazy) — data/config/eval
  metric paths stay usable without GPU stack installed.
- W&B must never be a hard dependency of a run: no API key / offline ->
  `report_to="none"` + JSON logs (still satisfies tracking via runs/*.json +
  results table; W&B is the dashboard layer).
- Windows-compatible paths (pathlib everywhere), UTF-8 everywhere.
- Network-dependent tests SKIP gracefully (try/except around hub download,
  print "skipped: <reason>", count as pass). Set `WANDB_MODE=offline` and
  `HF_HUB_OFFLINE` unset-controlled inside tests via env juggling.
- Tests must run fast on CPU: tiny models only — build a LOCAL tiny model in
  the test (create ~1M-param T5-via-transformers `T5ForConditionalGeneration`
  with a small `T5Config` + save a real tokenizer IF hub reachable; else skip),
  or use `hf-internal-testing/tiny-random-mt5` when reachable.
- Existing tests/test_smoke.py behaviour unchanged; main agent adds discovery
  of `tests/test_week3_*.py` (each exposes `run()` printing "ok  <name>").

## 5. Notebook (C) — ~~`notebooks/week3_colab.ipynb`~~ (dropped: training ran via the CLI scripts on Kinesis; see `docs/week3_kinesis_guide.md`)
1. GPU check (`torch.cuda.get_device_name`, assert T4-or-better else warning)
2. `git clone` the team repo + `pip install -r requirements.txt -r requirements-training.txt`
3. `wandb login` (env var or interactive) — with "skip if you want JSON-only logs" note
4. `python scripts/build_guz_benchmark.py`
5. Zero-shot eval runs (zs_mt5, zs_nllb)
6. Fine-tune runs (base, freeze, guz50/200) — one cell per run with time print
7. Optional augmentation cell (back-translate with best EN→SW ckpt, retrain ft_nllb_aug)
8. `python scripts/translate.py --demo` (success criterion)
9. `write_results_table` -> show reports/week3_results.md
10. Zip `runs/` + results for download; cell to commit results back to the repo
Keep markdown explanations student-voiced and brief; every cell rerunnable.

## 6. Docs (C) — ~~`docs/week3_colab_guide.md`~~ (replaced by `docs/week3_kinesis_guide.md`)
Prereqs (Google account, wandb account), step-by-step Colab usage, expected
run times on T4 (mt5-small ~20–30 min, nllb-600M ~35–50 min per 3-epoch run),
**GPU troubleshooting section** (mid-week check-in item): runtime disconnects,
OOM (halve batch + raise grad-accum), wandb offline mode, resuming from
checkpoint-best, how to split the matrix across Colab sessions.

## 7. Addendum — Ekegusii from PSA gold (supersedes §2.2 seed note)

Following the framework audit (`reports/framework_audit.md`), Ekegusii
training pairs come from **real PSA data** (lecturer gold dataset merged
into `psa_parallel_week1.csv`), not from FLORES:

- `TrainConfig.fewshot_guz` semantics changed (fields/defaults unchanged):
  `0` = exclude guz pairs, `N` = seeded cap on PSA-sourced guz pairs,
  `-1` = all PSA-sourced guz pairs.
- `load_psa_pairs` additionally emits `en-guz` / `sw-guz` pairs from split
  rows with non-empty `Ekegusii` (provenance stays `"psa"`; `sw-guz` only
  when Kiswahili is non-empty).
- FLORES **dev** seeding was later dropped entirely (see §8): FLORES-200
  contains no Ekegusii, so the PSA train split is the only guz source.
- The Ekegusii benchmark is built from the held-out **test** split
  (`scripts/build_guz_benchmark.py` → `guz_test.tsv`); still never in
  training — the guard in `build_train_dataset` raises when guz pairs are
  requested but the train split has no Ekegusii text.

## 8. Addendum — No Ekegusii in FLORES-200 or NLLB-200 (verified)

While preparing the Week 3 Colab runs we verified both assumptions behind
the original FLORES plan, and both failed:

1. **FLORES-200 has no Ekegusii.** The published archive
   (`flores200_dataset.tar.gz`) contains 204 languages; `guz_Latn` is absent
   (Kenyan languages present: kik, kam, luo; plus som and swh). There is no
   off-the-shelf Ekegusii benchmark.
2. **NLLB-200 has no `guz_Latn` token.** The tokenizer's language list has
   kik/luo/swh but no guz — NLLB-200 has zero Ekegusii pretraining. mT5's
   pretraining corpus (mC4) does not cover Ekegusii either.

Consequences, now implemented:

- `scripts/build_guz_benchmark.py` builds
  `data/external/guz_benchmark/guz_test.tsv` (138 eng–guz pairs) from our
  own held-out test split — there is no off-the-shelf benchmark to fetch.
  Eval specs are `psa_test_en-guz` / `psa_test_guz-en`.
- The FLORES dev seed loader was **removed** (`load_flores_seed`,
  `flores_dev_seed` provenance gone); `fewshot_guz` caps PSA-sourced guz
  train pairs, the only legitimate guz source.
- `training/lang_tokens.py::ensure_lang_token` adds the missing `guz_Latn`
  token to NLLB (resize embeddings, donor-initialised from `swh_Latn` — a
  close Bantu relative) before fine-tuning; MTTranslator does the same at
  inference time (no-op for fine-tuned checkpoints that already carry it).
- **NLLB zero-shot Ekegusii is undefined by design** (no unmodified base
  model can be evaluated on a language its tokenizer lacks); the ablation
  matrix therefore only runs guz zero-shot evals for mT5.
- Reporting narrative: both models face a **truly unseen language**; all
  Ekegusii results come from fine-tuning on the 2,848 real PSA guz pairs
  (train split), benchmarked on 138 held-out test pairs.
