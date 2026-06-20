<div align="center">
<h2>🚀 Unified Runner — <code>vtr_run.py</code></h2>
<p>One command to evaluate any supported <b>model × method</b> — without remembering each subproject's per-env argument quirks.</p>
</div>

> 🧩 Part of [**PriorTR**](../README.md) · setup: [LLaVA](../image/LLaVA/README.md) · [InternVL](../image/InternVL/README.md) · [Qwen3-VL](../image/Qwen3-VL/README.md) · [Video-LLaVA](../video/Video-LLaVA/README.md) · [add a method](adding-a-method.md)

## 🧩 Why a launcher (not one package)

Each base model pins a **mutually-incompatible** `transformers` (LLaVA `4.37.2`, InternVL `≤4.49`,
Qwen3-VL `5.2.0.dev0`), so every subproject has its own conda env and they cannot coexist in one
process. The launcher therefore **does not load models** — it builds the right command and dispatches
it into the matching env via `conda run -n <env>`. Same per-env isolation lmms-eval already uses, with
a single uniform front-end.

## 🗺️ Capability matrix

```
model        env                 priortr  priortr_2f  fastv  sparsevlm  vispruner  baseline
llava        PriorTRllava           ✓                                                  ✓
internvl     PriorTRinternvl        ✓                  ✓                               ✓
qwen3vl      PriorTRqwen3vl         ✓         ✓        ✓        ✓          ✓           ✓
video-llava  PriorTRvideollava                ✓        ✓                               ✓
```

`python vtr_run.py --list` prints it (and marks each env ✓ present / ✗ missing). Illegal combinations
(e.g. `--model llava --method fastv`) are rejected with the supported list.

- **`priortr_2f`** — two-forward variant of PriorTR (explicit question-free prior forward instead of the
  single-forward causal-mask shortcut). Video-LLaVA has **no single-forward `priortr`** (video lacks the shortcut).
- **Video-LLaVA** runs through a second **`native_video`** backend: its own `run_inference_video_qa.py`
  (not lmms-eval). Pass a video dataset via `--video-dir` / `--gt-question` / `--gt-answers` instead of
  `--tasks`, cap with `--num-samples`, and use `--keep-tokens` (no `--keep-ratio`).

## 🚀 Usage

```bash
python vtr_run.py --list                                  # what's available

# PriorTR on Qwen3-VL — 2/9 keep ratio, 2-GPU accelerate
python vtr_run.py --model qwen3vl --method priortr --tasks mme,mmbench_en_dev \
    --keep-ratio 0.2222 --prune-layer 3 --gpus 0,1 --num-processes 2

# FastV on InternVL — keep 192 tokens
python vtr_run.py --model internvl --method fastv --tasks mme --keep-tokens 192 --prune-layer 2 --gpus 0

# Baseline (no pruning) on LLaVA
python vtr_run.py --model llava --method baseline --tasks pope --gpus 0

# PriorTR-2F on Video-LLaVA — native backend, a video dataset (not --tasks)
python vtr_run.py --model video-llava --method priortr_2f \
    --video-dir /data/MSVD_Zero_Shot_QA/videos \
    --gt-question /data/MSVD_Zero_Shot_QA/test_q.json \
    --gt-answers  /data/MSVD_Zero_Shot_QA/test_a.json \
    --keep-tokens 64 --prune-layer 3 --param query_aggregation=question --num-samples 500 --gpus 0

# Inspect the exact command without running it
python vtr_run.py --model qwen3vl --method priortr --tasks mme --keep-ratio 0.2222 --dry-run
```

The launcher translates unified flags into each subproject's own names (`strategy=` vs `vtr_strategy=`,
`keep_tokens=` vs `vtr_keep_ratio=`, the `PYTHONPATH` / `attn_implementation=sdpa` quirks, etc.).

## 🎛️ Method-specific hyperparameters

Common knobs (`--keep-tokens` / `--keep-ratio` / `--prune-layer`) apply to every method. Each method
*also* has its own, passed via repeatable `--param NAME=VALUE` and **validated against the chosen
method** (e.g. `token_merge` on `priortr` is rejected — only SparseVLM reads it). Discover them with
`--describe`:

```bash
$ python vtr_run.py --describe qwen3vl priortr
qwen3vl / priortr   (env: PriorTRqwen3vl, wrapper: qwen3_vl_vtr)
  common knobs: --keep-tokens | --keep-ratio, --prune-layer
  --param options for this method:
    query_aggregation= {last|question|auto}  query attention aggregation
    head_aggregation=  {mean|max}            aggregation across attention heads
```

| method | `--param` options |
|---|---|
| `priortr` / `priortr_2f` / `fastv` | `query_aggregation`, `head_aggregation` |
| `priortr_2f` | above; also runs an extra prior forward (`prior_prompt`/`prior_mode` use config defaults) |
| `sparsevlm` | `token_merge` (defaults to `True`) |
| `vispruner` | `important_ratio` (note: `prune_layer` is forced to 1 internally) |
| `internvl` (any) | `max_num` (image tiles) |

> **Default *values* live in each subproject's config** (single source of truth) — the launcher does
> not duplicate them. It only injects an *intended* default where it differs from the bare config
> default (currently just SparseVLM's `token_merge=True`); override with `--param token_merge=False`.

## 🚩 Key flags

| Flag | Meaning |
|---|---|
| `--model` | `llava` \| `internvl` \| `qwen3vl` \| `video-llava` |
| `--method` | `priortr` \| `priortr_2f` \| `fastv` \| `sparsevlm` \| `vispruner` \| `baseline` (per matrix) |
| `--tasks` | lmms-eval task list, comma-separated (image models) |
| `--keep-tokens` / `--keep-ratio` | token budget (mutually exclusive; video-llava: `--keep-tokens` only) |
| `--prune-layer` | pruning layer (subproject default if unset) |
| `--param NAME=VALUE` | method-specific hyperparameter (repeatable, validated per method) |
| `--describe MODEL METHOD` | list a combo's tunable hyperparameters and exit |
| `--gpus` | `CUDA_VISIBLE_DEVICES` value |
| `--num-processes` | `>1` uses `accelerate launch` for multi-GPU throughput (image models; speed only) |
| `--env` | override the conda env name for `--model` (else `envs.json`, else default) |
| `--pretrained` | override the HF checkpoint |
| `--extra` | raw extra `model_args`, appended verbatim (unvalidated escape hatch) |
| `--limit N` | lmms-eval `--limit`: cap #samples (int or fraction) — handy for smoke tests |
| `--dry-run` | print the command, don't execute |
| **video-llava only** | *(native_video backend)* |
| `--video-dir` | directory of video files |
| `--gt-question` / `--gt-answers` | ground-truth questions / answers JSON |
| `--num-samples N` | cap #QA samples (native analogue of `--limit`) |
| `--cache-dir` | value for the script's required `--cache_dir` |

## 📦 Environments

The launcher dispatches into an env **by name** — it never creates one. Build each env **once** per its
subproject README. The name is resolved in this order:

```
--env <NAME>          (per-invocation; needs --model)
  ▸ envs.json[model]  (per-checkout file: {"qwen3vl": "MyEnv"}; git-ignored)
  ▸ REGISTRY default  (PriorTRllava / PriorTRinternvl / PriorTRqwen3vl / PriorTRvideollava)
```

So a machine whose envs are named differently just points `--env`/`envs.json` at the real names — no
launcher edits. Before a real run the launcher **preflights** the resolved env and, if absent, prints a
clear error naming the README to follow (instead of a raw `conda run` failure).

- **Reproducibility:** each subproject ships a locked `environment.yml` (`conda env export`, linux-64 /
  CUDA 12.8). It's a **record, not a one-command rebuild** — `torch` (cu128 index), `transformers`
  (PyPI pin or git commit), and the editable `lmms-eval` are off default channels, so install those per
  the README first; the `.yml` pins the rest.
- **lmms-eval is not bundled:** clone it under `<subproject>/lmms-eval` per each README before a real
  run. `--dry-run` prints the command even when lmms-eval is absent.
