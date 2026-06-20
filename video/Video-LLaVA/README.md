# PriorTR-2F on Video-LLaVA

Visual token pruning for **video** understanding. **PriorTR-2F is the two-forward variant of PriorTR**:
identical task attention `P` and score `S = P · log(P / Q)`, but the prior `Q` comes from an explicit
**prior forward** (empty/generic prompt) followed by a **task forward** (the real question), instead of
PriorTR's single-forward causal-mask shortcut — video models lack that shortcut. The score identifies
visual tokens carrying task-specific information beyond the prior, and prunes the rest.

> Part of [**PriorTR**](../../README.md) — see the [unified runner](../../docs/RUNNER.md) to launch any model × method with one CLI.

## Environment Setup

### Standard GPU (CUDA 12.4 or earlier)

```bash
conda create -n PriorTRvideollava python=3.10 -y
conda activate PriorTRvideollava
pip install -e .          # installs deps from pyproject.toml, incl. torch>=2.0.1 (cu121)
```

### Newer GPU (SM_120+, CUDA 12.8)

For Blackwell / RTX PRO series, install dependencies manually so the `torch>=2.0.1` pin does not pull
an incompatible build:

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
pip install "openai==0.28" "tensorboardX==2.6.2.2"

# 3. Install the package (no-deps to skip the pyproject torch pin)
pip install -e . --no-deps
```

> **Note (numpy / scikit-learn):** cu128 torch wheels ship numpy 2.x. Do not pin
> `scikit-learn==1.2.2` (its binaries are built against numpy 1.x and will segfault) — use unpinned
> `scikit-learn`.

> **Note (pytorchvideo patch, required for cu128 / recent torchvision):** recent torchvision removed
> `torchvision.transforms.functional_tensor`, which `pytorchvideo/transforms/augmentations.py` imports,
> so inference fails at import time. Apply this one-time patch (it locates the file via the **top-level**
> `pytorchvideo` package — importing the `augmentations` submodule directly is the very import that
> breaks):

```bash
python - <<'PY'
import os, pytorchvideo
f = os.path.join(os.path.dirname(pytorchvideo.__file__), "transforms", "augmentations.py")
s = open(f).read()
old = "import torchvision.transforms.functional_tensor as F_t"
new = ("try:\n    import torchvision.transforms.functional_tensor as F_t\n"
       "except ModuleNotFoundError:\n    import torchvision.transforms._functional_tensor as F_t")
if "except ModuleNotFoundError" not in s and old in s:
    open(f, "w").write(s.replace(old, new)); print("patched:", f)
else:
    print("already patched / nothing to do")
PY
```

**Verify:**

```bash
python -c "
import torch, transformers; print(torch.__version__, transformers.__version__)
from videollava.vtr.config import VTRConfig, PriorTR2FConfig
from videollava.vtr.strategy import PriorTR2FStrategy, FastVStrategy
print('VTR OK')
"
```

## Inference

Video-LLaVA ships its own inference pipeline (it does **not** use lmms-eval). Point `--video_dir`,
`--gt_file_question`, and `--gt_file_answers` at your dataset (same script works for MSVD, MSRVTT, TGIF,
ActivityNet).

```bash
S=videollava/eval/video/run_inference_video_qa.py
M=LanguageBind/Video-LLaVA-7B
DATA=/path/to/MSVD            # expects $DATA/videos, $DATA/test_q.json, $DATA/test_a.json

# Baseline (no pruning) — omit the --vtr_* flags
python $S --model_path $M --cache_dir ./cache \
    --video_dir $DATA/videos --gt_file_question $DATA/test_q.json --gt_file_answers $DATA/test_a.json \
    --output_dir output/msvd_baseline --output_name pred --num_samples 500

# PriorTR-2F (V-Information pruning, keep 64 tokens)
python $S --model_path $M --cache_dir ./cache \
    --video_dir $DATA/videos --gt_file_question $DATA/test_q.json --gt_file_answers $DATA/test_a.json \
    --output_dir output/msvd_priortr_2f_k64 --output_name pred \
    --vtr_enabled --vtr_strategy priortr_2f --vtr_prune_layer 3 --vtr_keep_tokens 64 \
    --vtr_query_aggregation question --vtr_head_aggregation mean

# FastV baseline — swap --vtr_strategy fastv
python $S --model_path $M --cache_dir ./cache \
    --video_dir $DATA/videos --gt_file_question $DATA/test_q.json --gt_file_answers $DATA/test_a.json \
    --output_dir output/msvd_fastv_k64 --output_name pred \
    --vtr_enabled --vtr_strategy fastv --vtr_prune_layer 3 --vtr_keep_tokens 64
```

## GPT Evaluation

After inference, score predictions with the GPT-based evaluator (requires an OpenAI key):

```bash
python videollava/eval/video/eval_video_qa.py \
    --pred_path output/msvd_priortr_2f_k64/pred.json \
    --output_dir output/msvd_priortr_2f_k64/gpt_eval \
    --output_json output/msvd_priortr_2f_k64/results.json \
    --api_key YOUR_OPENAI_API_KEY --api_base https://api.openai.com/v1 \
    --model gpt-3.5-turbo --num_tasks 4
```

This reports accuracy and average score (0–5) against the ground truth.

## VTR Parameters

| Parameter | CLI Flag | Default | Description |
|---|---|---|---|
| enabled | `--vtr_enabled` | `False` | Enable visual token reduction |
| strategy | `--vtr_strategy` | `priortr_2f` | `priortr_2f` or `fastv` |
| prune_layer | `--vtr_prune_layer` | `3` | LLM layer index at which to prune |
| keep_tokens | `--vtr_keep_tokens` | `194` | Number of visual tokens to keep |
| keep_ratio | (config only) | `0.25` | Fraction to keep (ignored when keep_tokens is set) |
| query_aggregation | `--vtr_query_aggregation` | `question` | `question` (all question tokens) or `last` (last token) |
| head_aggregation | `--vtr_head_aggregation` | `mean` | aggregate across heads: `mean` or `max` |
| prior_prompt | (config only) | `""` | prompt for the prior forward (PriorTR-2F only) |
| score_threshold | (config only) | `None` | keep tokens above this V-Info score (PriorTR-2F only) |
| adaptive_layer | (config only) | `False` | adaptive layer selection across candidates (PriorTR-2F only) |

## License

Built on [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA); released under the Apache 2.0
License (see the root [LICENSE](../../LICENSE)).
