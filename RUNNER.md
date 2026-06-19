# Unified Runner (`vtr_run.py`)

One command to evaluate any supported **model × method** combination, without
remembering each subproject's per-env argument quirks.

## Why a launcher (and not one package)

Each base model pins a **mutually-incompatible** `transformers` version
(LLaVA `4.37.2`, InternVL `≤4.49`, Qwen3-VL `5.2.0.dev0`), so every subproject
has its own conda env and they cannot coexist in one Python process. The
launcher therefore does **not** load models itself — it builds the correct
`lmms-eval` command and dispatches it into the matching env via
`conda run -n <env>`. This is the same per-env isolation lmms-eval already uses;
the launcher just gives it a single, uniform front-end.

## Capability matrix

```
model     env                 priortr  fastv  sparsevlm  vispruner  baseline
llava     PriorTRllava           ✓                                     ✓
internvl  PriorTRinternvl        ✓      ✓                              ✓
qwen3vl   PriorTRqwen3vl         ✓      ✓        ✓          ✓          ✓
```

Run `python vtr_run.py --list` to print it. Illegal combinations (e.g.
`--model llava --method fastv`) are rejected with the supported list.

> **InfoVTR** and **Video-LLaVA** are intentionally not wired in yet — they
> are handled separately later.

## Usage

```bash
# See what's available
python vtr_run.py --list

# PriorTR on Qwen3-VL, 2/9 keep ratio, 2-GPU accelerate
python vtr_run.py --model qwen3vl --method priortr --tasks mme,mmbench_en_dev \
    --keep-ratio 0.2222 --prune-layer 3 --gpus 0,1 --num-processes 2

# FastV on InternVL, keep 192 tokens
python vtr_run.py --model internvl --method fastv --tasks mme \
    --keep-tokens 192 --prune-layer 2 --gpus 0

# Baseline (no pruning) on LLaVA
python vtr_run.py --model llava --method baseline --tasks pope --gpus 0

# Inspect the exact command without running it
python vtr_run.py --model qwen3vl --method priortr --tasks mme --keep-ratio 0.2222 --dry-run
```

The launcher translates unified flags into each subproject's own argument names
(`strategy=` vs `vtr_strategy=`, `keep_tokens=` vs `vtr_keep_ratio=`, the
`PYTHONPATH` / `attn_implementation=sdpa` quirks, etc.).

## Key flags

| Flag | Meaning |
|---|---|
| `--model` | `llava` \| `internvl` \| `qwen3vl` |
| `--method` | `priortr` \| `fastv` \| `sparsevlm` \| `vispruner` \| `baseline` (per matrix) |
| `--tasks` | lmms-eval task list, comma-separated |
| `--keep-tokens` / `--keep-ratio` | token budget (mutually exclusive) |
| `--prune-layer` | pruning layer (subproject default if unset) |
| `--gpus` | `CUDA_VISIBLE_DEVICES` value |
| `--num-processes` | `>1` uses `accelerate launch` for multi-GPU |
| `--pretrained` | override the HF checkpoint |
| `--extra` | raw extra `model_args`, appended verbatim |
| `--dry-run` | print the command, don't execute |

## Prerequisites

`lmms-eval` is **not bundled**; clone it under `<subproject>/lmms-eval` per each
subproject README before running for real. With `--dry-run` the launcher prints
the command even if lmms-eval is absent.
