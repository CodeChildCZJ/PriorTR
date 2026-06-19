# PriorTR on Qwen3-VL

Visual token pruning for Qwen3-VL using **PriorTR** (Prior Token Reduction), a single-forward V-Information method. PriorTR exploits causal attention to extract both a prior distribution (from the newline token after the image) and a task distribution (from the last query token) in a single forward pass, then computes V-Information scores `S = P * log(P / Q)` to identify and retain only the most task-relevant visual tokens.

This repository also includes **FastV**, **InfoVTR**, **SparseVLM**, and **VisPruner** baselines under a unified VTR (Visual Token Reduction) framework.

## Environment Setup

> **Proxy**: If your network requires a proxy to access external resources (HuggingFace, PyPI, GitHub), configure `http_proxy` and `https_proxy` before running the commands below.

### Standard GPU (CUDA 12.1 or earlier)

```bash
conda create -n PriorTRqwen3vl python=3.10 -y
conda activate PriorTRqwen3vl

# 1. Install PyTorch with cu121
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Install transformers dev version (pinned commit — see note below)
pip install git+https://github.com/huggingface/transformers.git@f8f2834e1a

# 3. Install the project (registers visual_token_pruning + creates qwen3 symlink)
python setup.py develop
```

### Newer GPU (SM_120+, CUDA 12.8)

For GPUs that require CUDA 12.8 (e.g., Blackwell / RTX PRO series), use `cu128` for PyTorch:

```bash
conda create -n PriorTRqwen3vl python=3.10 -y
conda activate PriorTRqwen3vl

# 1. Install PyTorch with cu128
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2. Install transformers dev version (pinned commit — see note below)
pip install git+https://github.com/huggingface/transformers.git@f8f2834e1a

# 3. Install the project
python setup.py develop
```

> **Why a pinned transformers commit?** The project's custom `qwen3/` model code depends on internal APIs (e.g., `check_model_inputs` in `transformers.utils.generic`) that are present in transformers `5.2.0.dev0` (commit `f8f2834e1a`) but were removed or renamed in later dev versions. Using the wrong version will cause import errors at runtime. Always install from this exact commit.

> **What does `python setup.py develop` do?** It registers `visual_token_pruning` as an importable Python package and automatically creates a symlink from `transformers/models/qwen3_vl` to the project's `qwen3/` directory, so that the custom model implementation (with VTR hooks) replaces the upstream version. Note: We use `setup.py develop` instead of `pip install -e .` because modern pip (PEP 660) does not trigger the post-install hook needed to create the symlink.

### Verify Installation

```python
python -c "
import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
import transformers; print(f'Transformers: {transformers.__version__}')
from transformers.utils.generic import check_model_inputs; print('check_model_inputs OK')
import transformers.models.qwen3_vl.modeling_qwen3_vl as m
import os; print(f'Qwen3VL path: {os.path.realpath(m.__file__)}')
from visual_token_pruning import VTRConfig; print('VTRConfig OK')
"
```

Expected output:
- `Transformers: 5.2.0.dev0`
- `check_model_inputs OK`
- `Qwen3VL path` should point to your project's `qwen3/` directory, NOT the backup
- `VTRConfig OK`

If `check_model_inputs` import fails, you have the wrong transformers version. Make sure to install from the pinned commit above.

## lmms-eval Setup

[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) is used for benchmark evaluation.

```bash
# Clone into project directory
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git

# Install in editable mode (no-deps to avoid conflicts)
cd lmms-eval && pip install -e . --no-deps && cd ..

# Install lmms-eval runtime dependencies
pip install jiwer rouge_score nltk sacrebleu evaluate datasets loguru tenacity spacy
pip install pycocoevalcap texttable protobuf sqlitedict openai pytablewriter
```

### Register the qwen3_vl_vtr Model

```bash
# Copy the model wrapper
cp ./lmms_eval_model/qwen3_vl_vtr.py ./lmms-eval/lmms_eval/models/simple/qwen3_vl_vtr.py
```

Then add the following entry to `./lmms-eval/lmms_eval/models/__init__.py` in the `AVAILABLE_SIMPLE_MODELS` dict:

```python
"qwen3_vl_vtr": "Qwen3_VL_VTR",
```

Verify:

```python
python -c "import lmms_eval; print('lmms_eval OK')"
```

## Evaluation Examples

All evaluations use `lmms-eval` with the `qwen3_vl_vtr` model. The model `Qwen/Qwen3-VL-8B-Instruct` is automatically downloaded from HuggingFace.

### Baseline (No Pruning)

```bash
cd lmms-eval
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

accelerate launch --num_processes=5 --main_process_port=29500 \
    -m lmms_eval --model qwen3_vl_vtr \
    --model_args "pretrained=Qwen/Qwen3-VL-8B-Instruct,vtr_enabled=False,attn_implementation=sdpa" \
    --tasks scienceqa_img --batch_size 1 \
    --output_path ../eval_results/baseline_sqa
```

### PriorTR (V-Information Pruning)

PriorTR with default settings (`strategy=priortr` and `query_aggregation=question` are defaults, no need to specify):

```bash
accelerate launch --num_processes=5 --main_process_port=29500 \
    -m lmms_eval --model qwen3_vl_vtr \
    --model_args "pretrained=Qwen/Qwen3-VL-8B-Instruct,attn_implementation=sdpa,vtr_keep_ratio=0.2222,vtr_prune_layer=3" \
    --tasks mme,mmbench_en_dev,mmbench_cn_dev --batch_size 1 \
    --output_path ../eval_results/priortr_0.2222_mme_mmbench
```

PriorTR with `last` query aggregation (override default):

```bash
accelerate launch --num_processes=5 --main_process_port=29500 \
    -m lmms_eval --model qwen3_vl_vtr \
    --model_args "pretrained=Qwen/Qwen3-VL-8B-Instruct,attn_implementation=sdpa,vtr_keep_ratio=0.2222,vtr_prune_layer=3,vtr_query_aggregation=last" \
    --tasks mme,mmbench_en_dev,mmbench_cn_dev --batch_size 1 \
    --output_path ../eval_results/priortr_last_0.2222_mme_mmbench
```

### FastV Baseline

FastV defaults to `query_aggregation=last` via `auto`:

```bash
accelerate launch --num_processes=5 --main_process_port=29500 \
    -m lmms_eval --model qwen3_vl_vtr \
    --model_args "pretrained=Qwen/Qwen3-VL-8B-Instruct,attn_implementation=sdpa,vtr_strategy=fastv,vtr_keep_ratio=0.2222,vtr_prune_layer=3" \
    --tasks scienceqa_img --batch_size 1 \
    --output_path ../eval_results/fastv_0.2222_sqa
```

### InfoVTR Baseline

InfoVTR defaults to `query_aggregation=question` via `auto`:

```bash
accelerate launch --num_processes=5 --main_process_port=29500 \
    -m lmms_eval --model qwen3_vl_vtr \
    --model_args "pretrained=Qwen/Qwen3-VL-8B-Instruct,attn_implementation=sdpa,vtr_strategy=infovtr,vtr_keep_ratio=0.2222,vtr_prune_layer=3" \
    --tasks mme --batch_size 1 \
    --output_path ../eval_results/mme_infovtr_0.2222
```

### SparseVLM Baseline

```bash
accelerate launch --num_processes=5 --main_process_port=29500 \
    -m lmms_eval --model qwen3_vl_vtr \
    --model_args "pretrained=Qwen/Qwen3-VL-8B-Instruct,vtr_enabled=True,vtr_strategy=sparsevlm,vtr_prune_layer=3,vtr_token_merge=True,vtr_keep_ratio=0.2222,attn_implementation=sdpa" \
    --tasks mme,mmbench_en_dev,mmbench_cn_dev --batch_size 1 \
    --output_path ../eval_results/sparsevlm_0.2222_mme_mmbench
```

### Sweep Multiple Ratios

Typical keep ratios are `0.1111` (1/9), `0.2222` (2/9), and `0.3333` (1/3). To sweep, run multiple evaluations changing the `vtr_keep_ratio` parameter.

## VTR Parameters

All VTR parameters are passed via `--model_args` as comma-separated key=value pairs. List-valued parameters use `;` as separator (e.g., `vtr_prune_layer=3;7;16`).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `vtr_enabled` | bool | `True` | Enable/disable visual token pruning |
| `vtr_strategy` | str | `priortr` | Pruning strategy: `priortr`, `fastv`, `infovtr`, `sparsevlm`, `vispruner` |
| `vtr_prune_layer` | int or list | `3` | Layer(s) at which to prune visual tokens (ignored by VisPruner, which always prunes pre-LLM at layer 1) |
| `vtr_keep_tokens` | int or list | `None` | Exact number of visual tokens to keep (overrides `vtr_keep_ratio`) |
| `vtr_keep_ratio` | float or list | `0.1111` | Fraction of visual tokens to keep (used when `vtr_keep_tokens` is not set) |
| `vtr_query_aggregation` | str | `auto` | (priortr/fastv) How to aggregate query attention: `auto` (per-strategy default), `last` (last token), or `question` (all question tokens). Auto resolves to `question` for priortr/infovtr, `last` for others |
| `vtr_head_aggregation` | str | `mean` | (priortr/fastv) How to aggregate across attention heads: `mean` or `max` |
| `vtr_token_merge` | bool | `False` | (SparseVLM) Merge pruned tokens into a few representative tokens instead of dropping them; the cluster count is derived automatically |
| `vtr_important_ratio` | float | `0.5` | (VisPruner) Fraction of kept tokens chosen by importance; the rest are chosen by diversity |

## Project Structure

```
.
├── qwen3/                              # Custom Qwen3-VL model (symlinked into transformers)
│   ├── modeling_qwen3_vl.py            # Model implementation with VTR hooks
│   ├── configuration_qwen3_vl.py       # Model configuration
│   ├── processing_qwen3_vl.py          # Processor
│   ├── modular_qwen3_vl.py             # Modular model components
│   ├── video_processing_qwen3_vl.py    # Video processing utilities
│   └── __init__.py
├── visual_token_pruning/               # VTR framework
│   ├── config.py                       # VTRConfig dataclass
│   ├── model/
│   │   ├── vtr_qwen3_vl.py            # VTRQwen3VLForConditionalGeneration
│   │   ├── prunable_qwen3_vl.py       # Qwen3-VL with prunable attention layers
│   │   ├── prior_utils.py             # Prior distribution utilities
│   │   ├── token_merge.py             # Token merging implementation
│   │   ├── deepstack_handler.py       # Deep-stack pruning handler
│   │   └── __init__.py
│   ├── strategy/
│   │   ├── priortr.py                 # PriorTR: single-forward V-Information pruning
│   │   ├── fastv.py                   # FastV: attention-based pruning
│   │   ├── infovtr.py                 # InfoVTR: two-forward V-Information pruning
│   │   ├── sparsevlm.py              # SparseVLM: sparse attention pruning
│   │   ├── vispruner.py              # VisPruner baseline
│   │   ├── base.py                   # PruningStrategy abstract base class
│   │   └── __init__.py
│   └── __init__.py
├── lmms_eval_model/
│   └── qwen3_vl_vtr.py               # lmms-eval model wrapper (Qwen3_VL_VTR)
├── lmms-eval/                         # Evaluation framework (cloned from GitHub)
├── setup.py                           # Installation with auto-symlink (python setup.py develop)
├── requirements.txt                   # All dependencies in one file
├── LICENSE                            # Apache 2.0
└── README.md
```

## License

This project is built on [Qwen3-VL](https://github.com/QwenLM/Qwen2.5-VL) and is released under the Apache 2.0 License.
