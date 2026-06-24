<div align="center">
<h2>PriorTR on Qwen3-VL</h2>
<p><b>Single-forward V-Information visual token pruning.</b> Estimate the prior <code>Q</code> (newline token after the image) and the posterior <code>P</code> (query tokens) in one forward, score by <code>S = P · log(P / Q)</code>, keep Top-K. Ships the richest method zoo.</p>
<p>
  <img src="https://img.shields.io/badge/conda-PriorTRqwen3vl-44A833?logo=anaconda&logoColor=white" alt="env">
  <img src="https://img.shields.io/badge/transformers-5.2.0.dev0-FFD21E?logo=huggingface&logoColor=black" alt="transformers">
  <img src="https://img.shields.io/badge/methods-PriorTR%20%C2%B7%20PriorTR--2F%20%C2%B7%20FastV%20%C2%B7%20SparseVLM%20%C2%B7%20VisPruner-3776AB" alt="methods">
</p>
</div>

> 🧩 Part of [**PriorTR**](../../README.md) · [unified runner](../../docs/RUNNER.md) · [add a method](../../docs/adding-a-method.md) · [CLSE pruning](../../docs/CLSE.md)

## ⚙️ Environment Setup

```bash
conda create -n PriorTRqwen3vl python=3.10 -y
conda activate PriorTRqwen3vl

# 1. PyTorch — cu121 for standard GPUs, cu128 for Blackwell / RTX PRO (SM_120+)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#   or: --index-url https://download.pytorch.org/whl/cu128

# 2. transformers — PINNED dev commit (see note)
pip install git+https://github.com/huggingface/transformers.git@f8f2834e1a

# 3. Install the project (registers visual_token_pruning + creates the qwen3 symlink)
python setup.py develop
```

> ⚠️ **Pinned transformers commit.** The custom `qwen3/` code depends on internal APIs (e.g.
> `check_model_inputs` in `transformers.utils.generic`) present in `5.2.0.dev0` (commit `f8f2834e1a`)
> but removed/renamed later. The wrong version causes import errors — install from this exact commit.

> ⚠️ **Use `python setup.py develop`, not `pip install -e .`.** It registers `visual_token_pruning`
> and symlinks `transformers/models/qwen3_vl → ./qwen3/` so the VTR-hooked model replaces the upstream
> one. Modern pip (PEP 660) skips the post-install hook that creates this symlink. The hook is robust:
> if the dependency step aborts (e.g. a transitive `typer` conflict), the symlink is still created in a
> `finally` block.

**Verify** — the `qwen3_vl` path must point to *this project's* `qwen3/`, not the backup:

```bash
python -c "
import torch, transformers, os
print(torch.__version__, transformers.__version__)        # expect transformers 5.2.0.dev0
from transformers.utils.generic import check_model_inputs  # must import
import transformers.models.qwen3_vl.modeling_qwen3_vl as m
print('Qwen3VL path:', os.path.realpath(m.__file__))       # must be .../Qwen3-VL/qwen3/...
from visual_token_pruning import VTRConfig; print('VTRConfig OK')
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
cp ./lmms_eval_model/qwen3_vl_vtr.py ./lmms-eval/lmms_eval/models/simple/qwen3_vl_vtr.py
```

Then add `"qwen3_vl_vtr": "Qwen3_VL_VTR",` to `AVAILABLE_SIMPLE_MODELS` in
`./lmms-eval/lmms_eval/models/__init__.py`.

## 🚀 Usage

Run from `lmms-eval/`. `Qwen/Qwen3-VL-8B-Instruct` downloads from HuggingFace on first use. Defaults:
`vtr_strategy=priortr`, `vtr_query_aggregation=auto` (→ `question` for priortr/priortr_2f, `last` for
others). Typical keep ratios: `0.1111` (1/9), `0.2222` (2/9), `0.3333` (1/3).

```bash
cd lmms-eval
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
LAUNCH="accelerate launch --num_processes=5 --main_process_port=29500 -m lmms_eval --model qwen3_vl_vtr"
M=Qwen/Qwen3-VL-8B-Instruct

# Baseline (no pruning)
$LAUNCH --model_args "pretrained=$M,vtr_enabled=False,attn_implementation=sdpa" \
    --tasks scienceqa_img --batch_size 1 --output_path ../eval_results/baseline_sqa

# PriorTR (single-forward)
$LAUNCH --model_args "pretrained=$M,attn_implementation=sdpa,vtr_keep_ratio=0.2222,vtr_prune_layer=3" \
    --tasks mme,mmbench_en_dev,mmbench_cn_dev --batch_size 1 --output_path ../eval_results/priortr_0.2222

# Other strategies — swap vtr_strategy: fastv | priortr_2f | sparsevlm | vispruner
$LAUNCH --model_args "pretrained=$M,attn_implementation=sdpa,vtr_strategy=priortr_2f,vtr_keep_ratio=0.2222,vtr_prune_layer=3" \
    --tasks mme --batch_size 1 --output_path ../eval_results/priortr_2f_0.2222
# SparseVLM also takes vtr_token_merge=True; VisPruner always prunes pre-LLM at layer 1.
```

> **PriorTR-2F** is the **two-forward variant of PriorTR**: identical task attention `P` and score
> `S = P · log(P / Q)`, but the prior `Q` comes from an explicit question-free second forward instead
> of the single-forward causal-mask shortcut.

## 🎛️ VTR Parameters

Passed via `--model_args` as comma-separated `key=value` pairs. List-valued params use `;`
(e.g. `vtr_prune_layer=3;7;16`).

**Core**

| Parameter | Type | Default | Description |
|---|:---:|:---:|---|
| `vtr_enabled` | bool | `True` | Enable/disable visual token pruning |
| `vtr_strategy` | str | `priortr` | `priortr`, `fastv`, `priortr_2f`, `sparsevlm`, `vispruner` |
| `vtr_prune_layer` | int \| list | `3` | Layer(s) to prune (ignored by VisPruner — prunes pre-LLM at layer 1) |
| `vtr_keep_tokens` | int \| list | `None` | Exact tokens to keep (overrides `vtr_keep_ratio`) |
| `vtr_keep_ratio` | float \| list | `0.1111` | Fraction to keep (used when `vtr_keep_tokens` is unset) |

**Strategy-specific**

| Parameter | Applies to | Default | Description |
|---|:---:|:---:|---|
| `vtr_query_aggregation` | priortr / fastv | `auto` | `auto`, `last`, or `question`. Auto → `question` for priortr/priortr_2f, `last` for others |
| `vtr_head_aggregation` | priortr / fastv | `mean` | Aggregate across heads: `mean` or `max` |
| `vtr_token_merge` | SparseVLM | `False` | Merge pruned tokens into representatives instead of dropping |
| `vtr_important_ratio` | VisPruner | `0.5` | Fraction of kept tokens chosen by importance; rest by diversity |

## 📄 License

Built on [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL); released under the Apache 2.0 License
(root [LICENSE](../../LICENSE)).
