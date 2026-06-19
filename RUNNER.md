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
model     env                 priortr  priortr_2f  fastv  sparsevlm  vispruner  baseline
llava     PriorTRllava           ✓                                                  ✓
internvl  PriorTRinternvl        ✓                  ✓                               ✓
qwen3vl   PriorTRqwen3vl         ✓         ✓        ✓        ✓          ✓           ✓
```

Run `python vtr_run.py --list` to print it. Illegal combinations (e.g.
`--model llava --method fastv`) are rejected with the supported list.

`priortr_2f` is the **two-forward variant of PriorTR** (explicit question-free prior
forward instead of the single-forward causal-mask shortcut); only Qwen3-VL implements it.

> **Video-LLaVA** is a separate subproject (its own non-lmms-eval pipeline) and is
> intentionally not wired into this launcher yet.

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

## Method-specific hyperparameters

Common knobs (`--keep-tokens` / `--keep-ratio` / `--prune-layer`) apply to every
method. Each method *also* has its own hyperparameters, passed via repeatable
`--param NAME=VALUE` and **validated against the chosen method** — e.g. giving
`token_merge` to `priortr` is rejected, because only SparseVLM reads it.

Discover what a combination accepts with `--describe`:

```bash
$ python vtr_run.py --describe qwen3vl priortr
qwen3vl / priortr   (env: PriorTRqwen3vl, wrapper: qwen3_vl_vtr)
  common knobs: --keep-tokens | --keep-ratio, --prune-layer
  --param options for this method:
    query_aggregation= {last|question|auto}  query attention aggregation
    head_aggregation=  {mean|max}            aggregation across attention heads
```

```bash
# tune priortr's attention aggregation
python vtr_run.py --model qwen3vl --method priortr --tasks mme --keep-ratio 0.2222 \
    --param query_aggregation=last --param head_aggregation=max

# vispruner's importance/diversity split
python vtr_run.py --model qwen3vl --method vispruner --tasks mme --keep-ratio 0.2222 \
    --param important_ratio=0.6
```

Per-method tunables (run `--describe <model> <method>` for the authoritative list):

| method | `--param` options |
|---|---|
| priortr / priortr_2f / fastv | `query_aggregation`, `head_aggregation` |
| priortr_2f | (above; also runs an extra prior forward — `prior_prompt`/`prior_mode` use config defaults, not exposed here) |
| sparsevlm | `token_merge` (defaults to `True`) |
| vispruner | `important_ratio` (note: `prune_layer` is forced to 1 internally) |
| internvl (any) | `max_num` (image tiles) |

**Default *values* live in each subproject's own config** (single source of
truth) — the launcher does not duplicate them. It only injects an *intended*
default where it differs from the bare config default (currently just SparseVLM's
`token_merge=True`); pass `--param token_merge=False` to override.

## Key flags

| Flag | Meaning |
|---|---|
| `--model` | `llava` \| `internvl` \| `qwen3vl` |
| `--method` | `priortr` \| `priortr_2f` \| `fastv` \| `sparsevlm` \| `vispruner` \| `baseline` (per matrix) |
| `--tasks` | lmms-eval task list, comma-separated |
| `--keep-tokens` / `--keep-ratio` | token budget (mutually exclusive) |
| `--prune-layer` | pruning layer (subproject default if unset) |
| `--param NAME=VALUE` | method-specific hyperparameter (repeatable, validated per method) |
| `--describe MODEL METHOD` | list a combo's tunable hyperparameters and exit |
| `--gpus` | `CUDA_VISIBLE_DEVICES` value |
| `--num-processes` | `>1` uses `accelerate launch` for multi-GPU eval throughput (speed only, not a hyperparameter) |
| `--env` | override the conda env name for `--model` (else `envs.json`, else the default) |
| `--pretrained` | override the HF checkpoint |
| `--extra` | raw extra `model_args`, appended verbatim (unvalidated escape hatch) |
| `--limit N` | lmms-eval `--limit`: cap #samples (int) or fraction — handy for smoke tests |
| `--dry-run` | print the command, don't execute |

## Environments (how dispatch finds them)

The launcher does **not** create environments — it dispatches into one *by name*
(`conda run -n <name>`). Each model needs its conda env built **once**, by
following that subproject's README (`conda create -n PriorTR<model>` → torch →
pinned `transformers` → `setup.py develop` / `pip install -e .` → clone lmms-eval
→ register the wrapper). The env name the launcher uses is resolved as:

```
--env <NAME>            (per-invocation; needs --model)
  > envs.json[model]    (per-checkout file: {"qwen3vl": "MyEnv"}; git-ignored)
  > REGISTRY default    (PriorTRllava / PriorTRinternvl / PriorTRqwen3vl)
```

So another machine whose envs are named differently doesn't have to edit the
launcher — point `--env`/`envs.json` at the real names. `python vtr_run.py --list`
marks each env ✓ present / ✗ missing, and before a real run the launcher
**preflights** the resolved env: if it's absent you get a clear error naming the
README to follow (instead of a raw `conda run` failure).

### Reproducibility

Each subproject ships a locked `environment.yml` (`conda env export`, captured on
linux-64 / CUDA 12.8). It pins every transitive version, but is a **record, not a
one-command rebuild**: `torch` (cu128 index), `transformers` (PyPI pin for
LLaVA/InternVL, a git commit for Qwen3-VL) and the editable `lmms-eval` are not on
default channels — install those per the README first; the `.yml` then pins the
rest. The README remains the authoritative setup; the `.yml` is the exact-version
companion.

## Prerequisites

`lmms-eval` is **not bundled**; clone it under `<subproject>/lmms-eval` per each
subproject README before running for real. With `--dry-run` the launcher prints
the command even if lmms-eval is absent.
