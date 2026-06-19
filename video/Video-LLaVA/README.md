# PriorTR-2F on Video-LLaVA

Visual token pruning for video understanding using V-Information theory.
**PriorTR-2F is the two-forward variant of PriorTR**: same task attention `P` and same
score `S = P * log(P / Q)`; the prior `Q` comes from an explicit prior forward instead of
PriorTR's single-forward causal-mask shortcut. Video models lack that shortcut available
to image-only models, so here PriorTR-2F uses the two-forward approach: a **prior forward**
(with an empty/generic prompt) followed by a **task forward** (with the real question). The
V-Information score `S = P * log(P / Q)` identifies visual tokens that carry task-specific
information beyond what the prior already captures, then prunes the rest.

## Environment Setup

### Standard GPU (CUDA 12.4 or earlier)

```bash
conda create -n PriorTRvideollava python=3.10 -y
conda activate PriorTRvideollava
pip install -e .
```

This installs all dependencies from `pyproject.toml`, including `torch>=2.0.1` with cu121.

### Newer GPU (SM_120+, CUDA 12.8)

On GPUs that require CUDA 12.8 (e.g., Blackwell / RTX PRO series), install
dependencies manually to avoid the `torch>=2.0.1` pin pulling an incompatible build.

```bash
conda create -n PriorTRvideollava python=3.10 -y
conda activate PriorTRvideollava

# 1. PyTorch with cu128
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2. Core dependencies
pip install transformers==4.37.2 "tokenizers>=0.15" sentencepiece==0.1.99 shortuuid
pip install "accelerate>=0.21.0" "peft>=0.4.0,<0.10.0" bitsandbytes
pip install pydantic "markdown2[all]" numpy scikit-learn
pip install "httpx>=0.24.0" uvicorn fastapi requests
pip install "einops==0.6.1" "einops-exts==0.0.4" "timm==0.6.13"
pip install decord pytorchvideo opencv-python-headless
pip install "openai==0.28"
pip install "tensorboardX==2.6.2.2"

# 3. Install the package (no-deps to skip pyproject.toml torch pin)
pip install -e . --no-deps
```

**Note on numpy / scikit-learn:** The cu128 torch wheels ship with numpy 2.x.
Do not pin `scikit-learn==1.2.2` -- its pre-built binaries are compiled against
numpy 1.x and will segfault. Use an unpinned `scikit-learn` instead.

**Note on pytorchvideo:** Newer torchvision removes the
`torchvision.transforms.functional_tensor` module. If you see an import error,
patch `pytorchvideo/transforms/augmentations.py` line 9:

```python
# replace:
import torchvision.transforms.functional_tensor as F_t
# with:
try:
    import torchvision.transforms.functional_tensor as F_t
except ModuleNotFoundError:
    import torchvision.transforms._functional_tensor as F_t
```

## Inference

### Baseline (no pruning)

```bash
python videollava/eval/video/run_inference_video_qa.py \
    --model_path LanguageBind/Video-LLaVA-7B \
    --cache_dir ./cache \
    --video_dir /path/to/MSVD/videos \
    --gt_file_question /path/to/MSVD/test_q.json \
    --gt_file_answers /path/to/MSVD/test_a.json \
    --output_dir output/msvd_baseline \
    --output_name pred \
    --num_samples 500
```

### PriorTR-2F (V-Information pruning)

```bash
python videollava/eval/video/run_inference_video_qa.py \
    --model_path LanguageBind/Video-LLaVA-7B \
    --cache_dir ./cache \
    --video_dir /path/to/MSVD/videos \
    --gt_file_question /path/to/MSVD/test_q.json \
    --gt_file_answers /path/to/MSVD/test_a.json \
    --output_dir output/msvd_priortr_2f_k64 \
    --output_name pred \
    --vtr_enabled \
    --vtr_strategy priortr_2f \
    --vtr_prune_layer 3 \
    --vtr_keep_tokens 64 \
    --vtr_query_aggregation question \
    --vtr_head_aggregation mean
```

### FastV baseline

```bash
python videollava/eval/video/run_inference_video_qa.py \
    --model_path LanguageBind/Video-LLaVA-7B \
    --cache_dir ./cache \
    --video_dir /path/to/MSVD/videos \
    --gt_file_question /path/to/MSVD/test_q.json \
    --gt_file_answers /path/to/MSVD/test_a.json \
    --output_dir output/msvd_fastv_k64 \
    --output_name pred \
    --vtr_enabled \
    --vtr_strategy fastv \
    --vtr_prune_layer 3 \
    --vtr_keep_tokens 64
```

Replace paths with your dataset locations. The same script works for MSVD, MSRVTT,
TGIF, and ActivityNet -- just point `--video_dir`, `--gt_file_question`, and
`--gt_file_answers` to the appropriate dataset files.

## GPT Evaluation

After inference, evaluate predictions using the GPT-based scorer:

```bash
python videollava/eval/video/eval_video_qa.py \
    --pred_path output/msvd_priortr_2f_k64/pred.json \
    --output_dir output/msvd_priortr_2f_k64/gpt_eval \
    --output_json output/msvd_priortr_2f_k64/results.json \
    --api_key YOUR_OPENAI_API_KEY \
    --api_base https://api.openai.com/v1 \
    --model gpt-3.5-turbo \
    --num_tasks 4
```

This scores each prediction against the ground truth and reports accuracy and
average score (0-5).

## VTR Parameters

| Parameter | CLI Flag | Default | Description |
|---|---|---|---|
| enabled | `--vtr_enabled` | `False` | Enable visual token reduction |
| strategy | `--vtr_strategy` | `priortr_2f` | Pruning strategy: `priortr_2f`, `fastv` |
| prune_layer | `--vtr_prune_layer` | `3` | LLM layer index at which to prune |
| keep_tokens | `--vtr_keep_tokens` | `194` | Number of visual tokens to keep after pruning |
| keep_ratio | (config only) | `0.25` | Fraction of tokens to keep (ignored when keep_tokens is set) |
| query_aggregation | `--vtr_query_aggregation` | `question` | Which query positions to aggregate: `question` (all question tokens) or `last` (last token only) |
| head_aggregation | `--vtr_head_aggregation` | `mean` | How to aggregate across attention heads: `mean` or `max` |
| prior_prompt | (config only) | `""` | Prompt used during the prior forward pass (PriorTR-2F only) |
| score_threshold | (config only) | `None` | Keep tokens with V-Info score above this threshold (PriorTR-2F only) |
| adaptive_layer | (config only) | `False` | Enable adaptive layer selection across candidate layers (PriorTR-2F only) |

### Verify Installation

```python
python -c "
import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
import transformers; print(f'Transformers: {transformers.__version__}')
from videollava.vtr.model import load_vtr_model
from videollava.vtr.config import VTRConfig, PriorTR2FConfig
from videollava.vtr.strategy import PriorTR2FStrategy, FastVStrategy
print('VTR OK')
"
```

## Project Structure

```
Video-LLaVA/
├── videollava/
│   ├── __init__.py
│   ├── constants.py
│   ├── conversation.py
│   ├── mm_utils.py
│   ├── model/                        # Original Video-LLaVA model
│   │   ├── language_model/
│   │   │   └── llava_llama.py
│   │   ├── multimodal_encoder/
│   │   │   └── languagebind/         # LanguageBind video/image encoders
│   │   ├── llava_arch.py
│   │   └── builder.py
│   ├── train/                        # Training utilities (used by inference scripts)
│   │   ├── train.py
│   │   └── llava_trainer.py
│   ├── eval/
│   │   ├── image/
│   │   │   └── run_inference_image_vtr.py # Image VTR inference script
│   │   └── video/
│   │       ├── run_inference_video_qa.py  # Main VTR-enabled inference script
│   │       └── eval_video_qa.py           # GPT-based evaluation
│   └── vtr/                           # Visual Token Reduction framework
│       ├── __init__.py
│       ├── config.py                  # VTRConfig, PriorTR2FConfig
│       ├── strategy/
│       │   ├── base.py               # PruningStrategy base class
│       │   ├── registry.py           # Strategy registry
│       │   ├── priortr_2f.py            # PriorTR-2F (V-Information)
│       │   └── fastv.py              # FastV (attention-based)
│       ├── model/
│       │   ├── builder.py            # load_vtr_model()
│       │   ├── prunable_llama.py     # LlamaModel with token pruning hooks
│       │   ├── rope_utils.py         # Unbounded RoPE for sparse position IDs
│       │   ├── vtr_llava.py          # Base VTR-enabled LLaVA
│       │   ├── fastv_llava.py        # FastV variant
│       │   └── priortr_2f_llava.py      # PriorTR-2F variants (fixed / adaptive layer)
│       └── utlis/
│           └── modeling_attn_mask_utils.py  # Custom attention mask utilities
├── pyproject.toml
└── README.md
```
