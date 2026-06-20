<div align="center">
<h2>PriorTR on InternVL2.5</h2>
<p><b>Single-forward V-Information visual token pruning.</b> Estimate the model's prior from causal attention and rank visual tokens by <code>S = P · log(P / Q)</code> — no extra prior forward. A <b>FastV</b> baseline is included under the same framework.</p>
<p>
  <img src="https://img.shields.io/badge/conda-PriorTRinternvl-44A833?logo=anaconda&logoColor=white" alt="env">
  <img src="https://img.shields.io/badge/transformers-%E2%89%A44.49.0-FFD21E?logo=huggingface&logoColor=black" alt="transformers">
  <img src="https://img.shields.io/badge/methods-PriorTR%20%7C%20FastV-3776AB" alt="methods">
</p>
</div>

> 🧩 Part of [**PriorTR**](../../README.md) · [unified runner](../../docs/RUNNER.md) · [add a method](../../docs/adding-a-method.md)

## ⚙️ Environment Setup

```bash
conda create -n PriorTRinternvl python=3.10 -y
conda activate PriorTRinternvl

# PyTorch — cu121 for standard GPUs, cu128 for Blackwell / RTX PRO (SM_120+)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#   or: --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

> ⚠️ **Why `transformers <= 4.49`?** InternVL2.5 loads custom code (InternLM2 backbone) via
> `trust_remote_code=True`. From transformers 4.50+, `GenerationMixin` was refactored out of
> `PreTrainedModel`, so `InternLM2ForCausalLM.generate()` disappears. Always use `transformers <= 4.49.0`.

**Verify**

```bash
python -c "import torch, transformers; print(torch.__version__, transformers.__version__); from internvl_vtr.config import VTRConfig; print('InternVL VTR OK')"
```

## 📦 lmms-eval Setup

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
cd lmms-eval && pip install -e . --no-deps && cd ..    # --no-deps keeps the pinned transformers

# runtime deps
pip install jiwer rouge_score nltk sacrebleu evaluate datasets loguru tenacity spacy
pip install pycocoevalcap texttable protobuf sqlitedict openai pytablewriter

# register the wrapper
cp ./lmms_eval_model/internvl_vtr.py ./lmms-eval/lmms_eval/models/simple/internvl_vtr.py
```

Then add `"internvl_vtr": "InternVLVTR",` to `AVAILABLE_SIMPLE_MODELS` in
`./lmms-eval/lmms_eval/models/__init__.py`.

## 🚀 Usage

Run from `lmms-eval/` (export `PYTHONPATH` so the wrapper can import `internvl_vtr`):

```bash
cd lmms-eval
export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH
M=OpenGVLab/InternVL2_5-8B

# Baseline (no pruning)
CUDA_VISIBLE_DEVICES=0 lmms-eval --model internvl_vtr \
    --model_args "pretrained=$M,strategy=baseline" \
    --tasks mme --batch_size 1 --output_path ../eval_results/baseline_mme

# PriorTR
CUDA_VISIBLE_DEVICES=0 lmms-eval --model internvl_vtr \
    --model_args "pretrained=$M,strategy=priortr,keep_tokens=192,prune_layer=2" \
    --tasks mme --batch_size 1 --output_path ../eval_results/priortr_192_l2

# FastV baseline (swap strategy=fastv)
CUDA_VISIBLE_DEVICES=0 lmms-eval --model internvl_vtr \
    --model_args "pretrained=$M,strategy=fastv,keep_tokens=192,prune_layer=2" \
    --tasks mme --batch_size 1 --output_path ../eval_results/fastv_192_l2
```

**Multi-GPU** (data-parallel via `accelerate`):

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH
accelerate launch --num_processes=5 --main_process_port=29500 -m lmms_eval --model internvl_vtr \
    --model_args "pretrained=$M,strategy=priortr,keep_tokens=128,prune_layer=2" \
    --tasks mme --batch_size 1 --output_path ../eval_results/priortr_128_l2
```

## 🎛️ VTR Parameters

Passed via `--model_args` as comma-separated `key=value` pairs.

| Parameter | Type | Default | Description |
|---|:---:|:---:|---|
| `strategy` | str | `baseline` | `priortr`, `fastv`, or `baseline` (no pruning) |
| `keep_tokens` | int | — | Exact tokens to keep (overrides `keep_ratio`) |
| `keep_ratio` | float | `0.25` | Fraction to keep (used when `keep_tokens` is unset) |
| `prune_layer` | int | `2` | Layer at which to prune (1-indexed) |
| `max_num` | int | `6` | Max image tiles for dynamic resolution |

## 📄 License

Built on [InternVL2.5](https://github.com/OpenGVLab/InternVL) (MIT); the PriorTR-specific code follows
the root [LICENSE](../../LICENSE).
