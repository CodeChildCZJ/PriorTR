# PriorTR on LLaVA

Visual token pruning for **LLaVA-1.5 / LLaVA-1.6** using **PriorTR**. PriorTR exploits the causal
attention structure to estimate the model's inherent prior in a **single forward pass**, scoring
visual tokens by V-Information `S = P · log(P / Q)` — no extra prior forward needed.

> Part of [**PriorTR**](../../README.md) — see the [unified runner](../../docs/RUNNER.md) to launch any model × method with one CLI.

## Environment Setup

### Standard GPU (CUDA 12.4 or earlier)

```bash
conda create -n PriorTRllava python=3.10 -y
conda activate PriorTRllava
pip install -e .          # installs deps from pyproject.toml, incl. torch==2.1.2 (cu121)
```

### Newer GPU (SM_120+, CUDA 12.8)

For Blackwell / RTX PRO series, install PyTorch separately, then LLaVA without its pinned torch:

```bash
conda create -n PriorTRllava python=3.10 -y
conda activate PriorTRllava

# 1. PyTorch with cu128
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2. LLaVA dependencies
pip install transformers==4.37.2 tokenizers==0.15.1 sentencepiece==0.1.99 shortuuid
pip install "accelerate>=0.21.0" "peft>=0.4.0,<0.10.0" bitsandbytes
pip install pydantic "markdown2[all]" numpy scikit-learn
pip install "httpx==0.24.0" uvicorn fastapi requests
pip install "einops==0.6.1" "einops-exts==0.0.4" "timm==0.6.13"

# 3. Install LLaVA (skip deps to avoid the torch pin)
pip install -e . --no-deps
```

> **Note (cu128):** do NOT pin `scikit-learn==1.2.2` — it has a binary incompatibility with the
> newer numpy that ships with cu128 wheels. Use unpinned `scikit-learn` and `numpy` (as above).

**Verify:**

```bash
python -c "import torch, transformers; print(torch.__version__, transformers.__version__); from llava.vtr.config import VTRConfig; print('LLaVA VTR OK')"
```

## lmms-eval Setup

[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) drives benchmark evaluation:

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
cd lmms-eval && pip install -e . --no-deps && cd ..    # --no-deps keeps the pinned transformers

# runtime deps
pip install jiwer rouge_score nltk sacrebleu evaluate datasets loguru tenacity spacy
pip install pycocoevalcap texttable protobuf sqlitedict openai pytablewriter

# register the wrapper
cp ./lmms_eval_model/llava_vtr.py ./lmms-eval/lmms_eval/models/simple/llava_vtr.py
```

Then add `"llava_vtr": "LlavaVTR",` to `AVAILABLE_SIMPLE_MODELS` in
`./lmms-eval/lmms_eval/models/__init__.py`.

## Usage

Run from the `lmms-eval/` directory. VTR options are passed via `--model_args` (comma-separated
`key=value`). `setup_vtr()` **auto-detects the LLaVA version** and fills optimal defaults — you only
need `strategy=priortr`:

| | LLaVA-1.5 | LLaVA-1.6 |
|---|---|---|
| `keep_tokens` | 192 | 320 |
| `prune_layer` | 3 | 3 |

```bash
cd lmms-eval
TASKS=gqa,mme,pope,textvqa_val,seedbench

# Baseline (no pruning)
python -m lmms_eval --model llava_vtr \
    --model_args pretrained=liuhaotian/llava-v1.5-7b,enabled=False \
    --tasks $TASKS --batch_size 1 --output_path ./results/baseline

# PriorTR (defaults auto-filled; works the same for llava-v1.6-vicuna-7b)
python -m lmms_eval --model llava_vtr \
    --model_args pretrained=liuhaotian/llava-v1.5-7b,strategy=priortr \
    --tasks $TASKS --batch_size 1 --output_path ./results/priortr

# Override a default (e.g. keep_tokens) — explicit args always win
python -m lmms_eval --model llava_vtr \
    --model_args pretrained=liuhaotian/llava-v1.5-7b,strategy=priortr,keep_tokens=128 \
    --tasks $TASKS --batch_size 1 --output_path ./results/priortr_k128
```

**Multi-GPU** (data-parallel via `accelerate`):

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch --num_processes=4 -m lmms_eval --model llava_vtr \
    --model_args pretrained=liuhaotian/llava-v1.5-13b,device_map=auto,strategy=priortr \
    --tasks $TASKS --batch_size 1 --output_path ./results/priortr_13b
```

## VTR Parameters

Passed via `--model_args` as comma-separated `key=value` pairs.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `strategy` | str | `priortr` | Pruning strategy |
| `enabled` | bool | `True` | Enable/disable visual token pruning |
| `prune_layer` | int or list | `3` | Layer(s) at which to prune (1-indexed) |
| `keep_tokens` | int | auto | Exact tokens to keep (auto: 192 for 1.5, 320 for 1.6) |
| `keep_ratio` | float | `0.25` | Fraction to keep (used when `keep_tokens` is unset) |

## License

Built on [LLaVA](https://github.com/haotian-liu/LLaVA); released under the Apache 2.0 License
(see the root [LICENSE](../../LICENSE)).
