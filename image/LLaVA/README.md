# PriorTR on LLaVA

Visual token pruning for LLaVA-1.5 / LLaVA-1.6 using **PriorTR**, which addresses the intrinsic prior problem in visual token pruning. PriorTR exploits causal attention to extract the model's inherent prior in a single forward pass, eliminating the need for an additional prior forward.

## Quick Start

```python
from llava.vtr.config import VTRConfig
from llava.vtr.model import load_vtr_model

# 1. Load model with PriorTR
tokenizer, model, image_processor, context_len = load_vtr_model(
    model_path="liuhaotian/llava-v1.5-7b",
    model_type="priortr",
)

# 2. Configure and inject PriorTR
config = VTRConfig(enabled=True)
model.setup_vtr(config)
# setup_vtr auto-detects model version and fills defaults:
#   LLaVA-1.5 → keep_tokens=192
#   LLaVA-1.6 → keep_tokens=320

# 3. Confirm configuration
print(config)
# VTRConfig(enabled=True, strategy='priortr', prune_layer=3,
#           keep_ratio=0.25, keep_tokens=192, ...)
```

You can also override any auto-detected default:

```python
config = VTRConfig(enabled=True, keep_tokens=128)  # override keep_tokens, auto-detect the rest
model.setup_vtr(config)
```

The model now automatically prunes visual tokens during `model.generate()` — no other code changes needed.

## Environment Setup

### Standard GPU (CUDA 12.4 or earlier)

```bash
conda create -n PriorTRllava python=3.10 -y
conda activate PriorTRllava
pip install -e .
```

This installs all dependencies from `pyproject.toml`, including `torch==2.1.2` with cu121.

### Newer GPU (SM_120+, CUDA 12.8)

For GPUs that require CUDA 12.8 (e.g., Blackwell / RTX PRO series), install PyTorch separately and then install LLaVA without its pinned torch dependency:

```bash
conda create -n PriorTRllava python=3.10 -y
conda activate PriorTRllava

# 1. Install PyTorch with cu128
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2. Install LLaVA dependencies manually
pip install transformers==4.37.2 tokenizers==0.15.1 sentencepiece==0.1.99 shortuuid
pip install "accelerate>=0.21.0" "peft>=0.4.0,<0.10.0" bitsandbytes
pip install pydantic "markdown2[all]" numpy scikit-learn
pip install "httpx==0.24.0" uvicorn fastapi requests
pip install "einops==0.6.1" "einops-exts==0.0.4" "timm==0.6.13"

# 3. Install LLaVA package (skip deps to avoid torch version conflict)
pip install -e . --no-deps
```

### Verify Installation

```python
python -c "
import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
import transformers; print(f'Transformers: {transformers.__version__}')
from llava.vtr.config import VTRConfig; print('LLaVA VTR OK')
"
```

**Note**: Do NOT pin `scikit-learn==1.2.2` with newer PyTorch (cu128). The pinned version has binary incompatibility with newer numpy. Use unpinned `scikit-learn` and `numpy` instead (as shown above).

## lmms-eval Setup

[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) is used for benchmark evaluation.

```bash
# Clone into project directory
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
cd lmms-eval
pip install -e . --no-deps

# Install lmms-eval runtime dependencies
pip install jiwer rouge_score nltk sacrebleu evaluate datasets loguru tenacity spacy
pip install pycocoevalcap texttable protobuf sqlitedict openai pytablewriter
```

### Register the llava_vtr Model

```bash
# Copy the model wrapper
cp ./lmms_eval_model/llava_vtr.py ./lmms-eval/lmms_eval/models/simple/llava_vtr.py
```

Then add the following entry to `./lmms-eval/lmms_eval/models/__init__.py` in the `AVAILABLE_SIMPLE_MODELS` dict:

```python
"llava_vtr": "LlavaVTR",
```

Verify:

```python
python -c "import lmms_eval; print('lmms_eval OK')"
```

## Evaluation Examples

All examples use `lmms-eval` from the `./lmms-eval` directory. The model is automatically downloaded from HuggingFace.

> **Version-aware defaults**: `setup_vtr()` auto-detects the LLaVA version and applies optimal defaults. You only need to pass `strategy=priortr`.
>
> | | LLaVA-1.5 | LLaVA-1.6 |
> |---|---|---|
> | `keep_tokens` | 192 | 320 |
> | `prune_layer` | 3 | 3 |

### Baseline (No Pruning)

```bash
cd lmms-eval

# LLaVA-1.5
python -m lmms_eval \
    --model llava_vtr \
    --model_args pretrained=liuhaotian/llava-v1.5-7b,enabled=False \
    --tasks gqa,mme,pope,textvqa_val,seedbench \
    --batch_size 1 \
    --output_path ./results/baseline_1.5

# LLaVA-1.6
python -m lmms_eval \
    --model llava_vtr \
    --model_args pretrained=liuhaotian/llava-v1.6-vicuna-7b,enabled=False \
    --tasks gqa,mme,pope,textvqa_val,seedbench \
    --batch_size 1 \
    --output_path ./results/baseline_1.6
```

### PriorTR (Default Settings)

```bash
# LLaVA-1.5 — auto: keep_tokens=192, query_aggregation=question, prune_layer=3
python -m lmms_eval \
    --model llava_vtr \
    --model_args pretrained=liuhaotian/llava-v1.5-7b,strategy=priortr \
    --tasks gqa,mme,pope,textvqa_val,seedbench \
    --batch_size 1 \
    --output_path ./results/priortr_1.5

# LLaVA-1.6 — auto: keep_tokens=320, query_aggregation=last, prune_layer=3
python -m lmms_eval \
    --model llava_vtr \
    --model_args pretrained=liuhaotian/llava-v1.6-vicuna-7b,strategy=priortr \
    --tasks gqa,mme,pope,textvqa_val,seedbench \
    --batch_size 1 \
    --output_path ./results/priortr_1.6
```

### PriorTR (Custom Settings)

Explicit parameters always override auto-detected defaults:

```bash
# Override keep_tokens for LLaVA-1.5
python -m lmms_eval \
    --model llava_vtr \
    --model_args pretrained=liuhaotian/llava-v1.5-7b,strategy=priortr,keep_tokens=128 \
    --tasks gqa,mme,pope,textvqa_val,seedbench \
    --batch_size 1 \
    --output_path ./results/priortr_1.5_k128
```

Sweep different `keep_tokens` values:

```bash
for K in 64 128 192; do
    python -m lmms_eval \
        --model llava_vtr \
        --model_args pretrained=liuhaotian/llava-v1.5-7b,strategy=priortr,keep_tokens=${K} \
        --tasks gqa,mme,pope,textvqa_val,seedbench \
        --batch_size 1 \
        --output_path ./results/priortr_1.5_k${K}
done
```

### Multi-GPU Evaluation

Use `accelerate` for data-parallel evaluation across multiple GPUs:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch --num_processes=4 -m lmms_eval \
    --model llava_vtr \
    --model_args pretrained=liuhaotian/llava-v1.5-13b,device_map=auto,strategy=priortr \
    --tasks gqa,mme,pope,textvqa_val,seedbench \
    --batch_size 1 \
    --output_path ./results/priortr_13b
```

## VTR Parameters

All VTR parameters are passed via `--model_args` as comma-separated key=value pairs.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `strategy` | str | `priortr` | Pruning strategy |
| `enabled` | bool | `True` | Enable/disable visual token pruning |
| `prune_layer` | int or list | `3` | Layer(s) at which to prune visual tokens (1-indexed) |
| `keep_tokens` | int | auto | Exact number of visual tokens to keep (auto: 192 for 1.5, 320 for 1.6) |
| `keep_ratio` | float | `0.25` | Fraction of visual tokens to keep (used when `keep_tokens` is not set) |

## Project Structure

```
.
├── llava/
│   ├── model/                     # Base LLaVA model (LLaMA backbone, CLIP encoder, projector)
│   ├── eval/                      # Original LLaVA evaluation scripts
│   ├── train/                     # Original LLaVA training scripts
│   ├── serve/                     # Original LLaVA serving scripts
│   ├── vtr/                       # Visual Token Reduction framework
│   │   ├── config.py              # VTRConfig dataclass
│   │   ├── model/
│   │   │   ├── builder.py         # load_vtr_model() entry point
│   │   │   ├── priortr_llava.py   # PriorTR-enabled LLaVA model
│   │   │   ├── vtr_llava.py       # Base VTR LLaVA model
│   │   │   ├── prunable_llama.py  # LLaMA with prunable attention layers
│   │   │   └── rope_utils.py      # Unbounded RoPE for sparse position IDs
│   │   └── strategy/
│   │       ├── registry.py        # Strategy registration system
│   │       ├── base.py            # PruningStrategy abstract base class
│   │       └── priortr.py         # PriorTR: single-forward V-Information pruning
│   ├── constants.py
│   ├── conversation.py
│   └── mm_utils.py
├── scripts/                       # Original LLaVA scripts (training, evaluation, conversion)
├── lmms_eval_model/
│   └── llava_vtr.py               # lmms-eval model wrapper for VTR evaluation
├── pyproject.toml
└── README.md
```

## License

This project is built on [LLaVA](https://github.com/haotian-liu/LLaVA) and is released under the Apache 2.0 License.
