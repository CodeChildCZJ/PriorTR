# PriorTR on InternVL2.5

Visual token pruning for InternVL2.5 using **PriorTR** (Prior Token Reduction), which addresses the intrinsic prior problem in visual token pruning. PriorTR exploits causal attention to extract the model's inherent prior in a single forward pass, eliminating the need for an additional prior forward.

This repository also includes a **FastV** baseline under the same VTR (Visual Token Reduction) framework.

## Quick Start

```python
from internvl_vtr.config import VTRConfig
from internvl_vtr.model.vtr_internvl import setup_vtr_model
from transformers import AutoModel, AutoTokenizer

# 1. Load InternVL2.5-8B
model = AutoModel.from_pretrained(
    "OpenGVLab/InternVL2_5-8B",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(
    "OpenGVLab/InternVL2_5-8B",
    trust_remote_code=True,
)

# 2. Configure and inject PriorTR
config = VTRConfig(
    enabled=True,
    strategy="priortr",
    keep_tokens=192,
    prune_layer=2,
)
setup_vtr_model(model, config, tokenizer)

# 3. Confirm configuration
print(config)
# VTRConfig(enabled=True, strategy='priortr', prune_layer=2,
#           keep_ratio=0.25, keep_tokens=192, ...)
```

The model now automatically prunes visual tokens during `model.generate()` — no other code changes needed.

## Environment Setup

### Standard GPU (CUDA 12.1 or earlier)

```bash
conda create -n PriorTRinternvl python=3.10 -y
conda activate PriorTRinternvl

# 1. Install PyTorch with cu121
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Install dependencies
pip install -r requirements.txt
```

### Newer GPU (SM_120+, CUDA 12.8)

For GPUs that require CUDA 12.8 (e.g., Blackwell / RTX PRO series):

```bash
conda create -n PriorTRinternvl python=3.10 -y
conda activate PriorTRinternvl

# 1. Install PyTorch with cu128
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2. Install dependencies
pip install -r requirements.txt
```

> **Why transformers <= 4.49?** InternVL2.5 uses `trust_remote_code=True` to load custom model code (InternLM2 backbone). Starting from transformers 4.50+, `GenerationMixin` was refactored out of `PreTrainedModel`, causing `InternLM2ForCausalLM.generate()` to disappear. Always use transformers <= 4.49.0.

### Verify Installation

```python
python -c "
import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
import transformers; print(f'Transformers: {transformers.__version__}')
from internvl_vtr.config import VTRConfig; print('InternVL VTR OK')
"
```

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

### Register the internvl_vtr Model

```bash
# Copy the model wrapper
cp ./lmms_eval_model/internvl_vtr.py ./lmms-eval/lmms_eval/models/simple/internvl_vtr.py
```

Then add the following entry to `./lmms-eval/lmms_eval/models/__init__.py` in the `AVAILABLE_SIMPLE_MODELS` dict:

```python
"internvl_vtr": "InternVLVTR",
```

Verify:

```python
python -c "import lmms_eval; print('lmms_eval OK')"
```

## Evaluation Examples

All examples use `lmms-eval` from the `./lmms-eval` directory. The model is automatically downloaded from HuggingFace.

### Baseline (No Pruning)

```bash
cd lmms-eval
export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 lmms-eval \
    --model internvl_vtr \
    --model_args "pretrained=OpenGVLab/InternVL2_5-8B,strategy=baseline" \
    --tasks mme --batch_size 1 \
    --output_path ../eval_results/baseline_mme
```

### PriorTR (V-Information Pruning)

```bash
CUDA_VISIBLE_DEVICES=0 lmms-eval \
    --model internvl_vtr \
    --model_args "pretrained=OpenGVLab/InternVL2_5-8B,strategy=priortr,keep_tokens=192,prune_layer=2" \
    --tasks mme --batch_size 1 \
    --output_path ../eval_results/priortr_192_l2_mme
```

### FastV Baseline

```bash
CUDA_VISIBLE_DEVICES=0 lmms-eval \
    --model internvl_vtr \
    --model_args "pretrained=OpenGVLab/InternVL2_5-8B,strategy=fastv,keep_tokens=192,prune_layer=2" \
    --tasks mme --batch_size 1 \
    --output_path ../eval_results/fastv_192_l2_mme
```

### PriorTR (Sweep Configurations)

```bash
for K in 64 128 192; do
    lmms-eval \
        --model internvl_vtr \
        --model_args "pretrained=OpenGVLab/InternVL2_5-8B,strategy=priortr,keep_tokens=${K},prune_layer=2" \
        --tasks mme --batch_size 1 \
        --output_path ../eval_results/priortr_${K}_l2_mme
done
```

### Multi-GPU Evaluation

Use `accelerate` for data-parallel evaluation across multiple GPUs:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH

accelerate launch --num_processes=5 --main_process_port=29500 \
    -m lmms_eval --model internvl_vtr \
    --model_args "pretrained=OpenGVLab/InternVL2_5-8B,strategy=priortr,keep_tokens=128,prune_layer=2" \
    --tasks mme --batch_size 1 \
    --output_path ../eval_results/priortr_128_l2_mme
```

## VTR Parameters

All VTR parameters are passed via `--model_args` as comma-separated key=value pairs.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `strategy` | str | `baseline` | Pruning strategy: `priortr`, `fastv`, or `baseline` (no pruning) |
| `keep_tokens` | int | — | Exact number of visual tokens to keep (overrides `keep_ratio`) |
| `keep_ratio` | float | `0.25` | Fraction of visual tokens to keep (used when `keep_tokens` is not set) |
| `prune_layer` | int | `2` | Layer at which to prune visual tokens (1-indexed) |
| `max_num` | int | `6` | Maximum number of image tiles for dynamic resolution |

## Project Structure

```
.
├── internvl_vtr/                      # VTR framework
│   ├── config.py                      # VTRConfig
│   ├── model/
│   │   ├── prunable_internlm2.py      # InternLM2Model with token pruning hooks
│   │   └── vtr_internvl.py            # setup_vtr_model() — main entry point
│   └── strategy/
│       ├── registry.py                # Strategy registration system
│       ├── base.py                    # PruningStrategy abstract base class
│       ├── fastv.py                   # FastV: attention-based pruning
│       └── priortr.py                 # PriorTR: single-forward V-Information pruning
├── lmms_eval_model/
│   └── internvl_vtr.py               # lmms-eval model wrapper for VTR evaluation
├── requirements.txt
└── README.md
```

## License

This project is built on [InternVL2.5](https://github.com/OpenGVLab/InternVL) and is released under the MIT License.
