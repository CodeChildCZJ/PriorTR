<div align="center">
<h2>PriorTR on Qwen2-VL</h2>
<p><b>Single-forward V-Information visual token pruning.</b> Estimate the prior <code>Q</code> and posterior <code>P</code> in one forward, score by <code>S = P · log(P / Q)</code>, keep Top-K — plus FastV and the native home of <a href="../../docs/CLSE.md">CLSE</a>. Self-contained: runs on <b>stock</b> <code>pip install transformers</code>, no library patch.</p>
<p>
  <img src="https://img.shields.io/badge/conda-PriorTRqwen2vl-44A833?logo=anaconda&logoColor=white" alt="env">
  <img src="https://img.shields.io/badge/transformers-4.57.x-FFD21E?logo=huggingface&logoColor=black" alt="transformers">
  <img src="https://img.shields.io/badge/methods-PriorTR%20%C2%B7%20FastV%20%C2%B7%20CLSE-3776AB" alt="methods">
</p>
</div>

> 🧩 Part of [**PriorTR**](../../README.md) · [unified runner](../../docs/RUNNER.md) · [add a method](../../docs/adding-a-method.md) · [CLSE pruning](../../docs/CLSE.md)

## ⚙️ Environment Setup

```bash
conda create -n PriorTRqwen2vl python=3.10 -y -c conda-forge --override-channels
conda activate PriorTRqwen2vl

# 1. PyTorch — cu121 for standard GPUs, cu128 for Blackwell / RTX PRO (SM_120+)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#   or: --index-url https://download.pytorch.org/whl/cu128

# 2. Qwen2-VL stack (the DynamicCache `.layers` KV-trim API requires transformers >= 4.57)
pip install "transformers==4.57.*" accelerate qwen-vl-utils pillow decord
```

> ✅ **Stock transformers, no patch needed.** `VTRQwen2VLForConditionalGeneration` sets the visual mask
> and image grid on its (prunable) text model itself inside `forward`, so the upstream package works
> as-is — plain `pip install`, no symlink and no source edits. The project is used via `PYTHONPATH`
> (there is nothing to `pip install -e`).

> ⚠️ Two install gotchas seen on a clean machine: the conda `defaults` channel can 403 (hence
> `-c conda-forge --override-channels`), and lmms-eval imports `decord` (easy to miss — included above).

**Verify**

```bash
cd <repo>/image/Qwen2-VL
PYTHONPATH=$PWD python -c "
import torch, transformers
print(torch.__version__, transformers.__version__)             # expect transformers 4.57.x
from visual_token_pruning import VTRConfig
from visual_token_pruning.model import VTRQwen2VLForConditionalGeneration
print('Qwen2-VL VTR OK')
"
```

## 📦 lmms-eval Setup

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
cd lmms-eval && pip install -e . --no-deps && cd ..    # --no-deps keeps the pinned transformers

# runtime deps
pip install jiwer rouge_score nltk sacrebleu evaluate datasets loguru tenacity spacy
pip install pycocoevalcap texttable protobuf sqlitedict openai pytablewriter

# register the wrapper
cp ./lmms_eval_model/qwen2_vl_vtr.py ./lmms-eval/lmms_eval/models/simple/qwen2_vl_vtr.py
```

Then add `"qwen2_vl_vtr": "Qwen2_VL_VTR",` to `AVAILABLE_SIMPLE_MODELS` in
`./lmms-eval/lmms_eval/models/__init__.py`.

## 🚀 Usage

Run from `lmms-eval/` with `PYTHONPATH` pointing at this project so the wrapper finds
`visual_token_pruning`. `Qwen/Qwen2-VL-7B-Instruct` downloads from HuggingFace on first use. Defaults:
`vtr_strategy=priortr`, `vtr_query_aggregation=auto` (→ `question` for priortr, `last` for others).

```bash
cd lmms-eval
export PYTHONPATH=<repo>/image/Qwen2-VL
M=Qwen/Qwen2-VL-7B-Instruct

# Baseline (no pruning)
python -m lmms_eval --model qwen2_vl_vtr \
    --model_args "pretrained=$M,vtr_enabled=False" \
    --tasks mme,gqa --batch_size 1 --output_path ../eval_results/baseline

# PriorTR (single-forward) — swap vtr_strategy=fastv for FastV
python -m lmms_eval --model qwen2_vl_vtr \
    --model_args "pretrained=$M,vtr_strategy=priortr,vtr_keep_ratio=0.2222,vtr_prune_layer=3" \
    --tasks mme,gqa --batch_size 1 --output_path ../eval_results/priortr_0.2222

# CLSE (progressive 3-stage) — one budget knob; see docs/CLSE.md for the full guide
python -m lmms_eval --model qwen2_vl_vtr \
    --model_args "pretrained=$M,vtr_strategy=clse,vtr_retain_ratio=0.334" \
    --tasks mme,gqa --batch_size 1 --output_path ../eval_results/clse_0.334
```

> **Multi-GPU** (data-parallel): prefix with `accelerate launch --num_processes=N` and keep
> `PYTHONPATH` exported in the same shell.

## 🎛️ VTR Parameters

Passed via `--model_args` as comma-separated `key=value` pairs. List-valued params use `;`
(e.g. `vtr_prune_layer=1;10;19`).

**Core**

| Parameter | Type | Default | Description |
|---|:---:|:---:|---|
| `vtr_enabled` | bool | `True` | Enable/disable visual token pruning |
| `vtr_strategy` | str | `priortr` | `priortr`, `fastv`, or `clse` |
| `vtr_prune_layer` | int \| list | `3` | Layer(s) to prune (CLSE auto-resolves to `1;10;19` if left scalar) |
| `vtr_keep_tokens` | int \| list | `None` | Exact tokens to keep (overrides `vtr_keep_ratio`) |
| `vtr_keep_ratio` | float \| list | `0.5` | Fraction to keep (used when `vtr_keep_tokens` is unset) |
| `vtr_query_aggregation` | str | `auto` | `auto`, `last`, or `question` (auto → `question` for priortr) |
| `vtr_head_aggregation` | str | `mean` | Aggregate across heads: `mean` or `max` |

**CLSE-specific** (see [docs/CLSE.md](../../docs/CLSE.md))

| Parameter | Default | Description |
|---|:---:|---|
| `vtr_retain_ratio` | `None` | Budget knob (`0.334` / `0.223` / `0.112`) activating the hard-coded 3-stage schedule |
| `vtr_clse_cutoff_ratio` | `0.1` | 2D-FFT high-pass cutoff |
| `vtr_clse_temp` | `0.1` | Evolution-factor sigmoid temperature |

## 📄 License

Built on [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL); released under the Apache 2.0 License
(root [LICENSE](../../LICENSE)).
