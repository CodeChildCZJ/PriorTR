# PriorTR: Prior-Corrected Visual Token Reduction

Official code for **"Accelerating Multimodal Large Language Models with Prior-Corrected Token Reduction"** (ECCV 2026).

> **TL;DR**: Attention-based visual token pruning is dominated by a model-induced prior. PriorTR corrects this by contrasting task-conditioned attention (P) with an instruction-agnostic prior (Q) estimated from a null token within a single forward pass, scoring tokens via V-Information: `S = P * log(P / Q)`.

## Overview

Visual token reduction accelerates MLLMs by pruning redundant image tokens at an early decoder layer. Existing methods rank tokens by raw attention magnitude, but we show this ranking is confounded by a **model-induced prior** — the model attends to certain regions even without any instruction. PriorTR explicitly disentangles this prior from instruction-conditioned evidence using the causal attention structure: the null token (e.g., `\n`) after the image cannot see instruction tokens under the causal mask, making its attention a natural prior estimate. The top-K tokens by V-Information score are physically retained, reducing computation for all subsequent layers.

## Supported Models

| Model | Path | Conda Env | `transformers` | Strategies |
|---|---|---|---|---|
| LLaVA-1.5 | [`image/LLaVA/`](image/LLaVA/) | `PriorTRllava` | `4.37.2` | PriorTR |
| InternVL2.5 | [`image/InternVL/`](image/InternVL/) | `PriorTRinternvl` | `≤4.49.0` | PriorTR, FastV |
| Qwen3-VL | [`image/Qwen3-VL/`](image/Qwen3-VL/) | `PriorTRqwen3vl` | `5.2.0.dev0` (pinned commit) | PriorTR, PriorTR-2F, FastV, SparseVLM, VisPruner |
| Video-LLaVA | [`video/Video-LLaVA/`](video/Video-LLaVA/) | `PriorTRvideollava` | `4.37.2` | PriorTR-2F, FastV |

Each subproject pins a **mutually-incompatible** `transformers` version, so every model lives in
its **own conda env** — they cannot coexist in one Python process. Each subproject has its own
README with environment setup, usage, and evaluation commands. `PriorTR-2F` is the two-forward
variant of PriorTR (explicit prior forward instead of the single-forward causal-mask shortcut);
Video-LLaVA has no single-forward PriorTR because video lacks that shortcut.

## Repository Structure

```
.
├── vtr_run.py                # Unified runner: one CLI for any model × method (see RUNNER.md)
├── RUNNER.md                 # Unified-runner docs (capability matrix, flags, per-method params)
├── image/
│   ├── LLaVA/                # LLaVA-1.5-7B / 13B
│   ├── InternVL/             # InternVL2.5-8B
│   └── Qwen3-VL/             # Qwen3-VL-8B-Instruct
├── video/
│   └── Video-LLaVA/          # Video-LLaVA-7B (its own inference pipeline, not lmms-eval)
├── .gitignore
├── LICENSE
└── README.md
```

Each subproject follows the same shape: a `<vtr>/` package (`config.py`, `strategy/`, `model/`),
a model wrapper (`lmms_eval_model/` for the image models; native `run_inference_*` scripts for
Video-LLaVA), a per-model `README.md`, and a locked `environment.yml` (image models).

## Environment Setup (from a fresh clone)

There is **no single environment** — you build one conda env per model you want to run, because
the `transformers` pins are mutually incompatible. You only set up the model(s) you need.

**1. Create the per-model env** (follow that subproject's README for the exact commands):

```bash
conda create -n PriorTR<model> python=3.10 -y          # name must match the table above
conda activate PriorTR<model>
pip install torch torchvision --index-url .../cu128     # cu128 for Blackwell/SM_120, else cu121
pip install <pinned transformers>                       # per model: see the table / subproject README
pip install -e .                                        # Qwen3-VL uses `python setup.py develop` (creates a symlink)
```

**2. Image models also need lmms-eval** (Video-LLaVA does not — it ships its own scripts):

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
cd lmms-eval && pip install -e . --no-deps && cd ..     # --no-deps keeps the pinned transformers
cp ./lmms_eval_model/<model>_vtr.py ./lmms-eval/lmms_eval/models/simple/
#   then register the wrapper in lmms-eval/.../models/__init__.py (AVAILABLE_SIMPLE_MODELS)
```

Model weights and benchmark datasets download from HuggingFace on first run. Video-LLaVA also pulls
the `LanguageBind` vision encoders; for video QA you point the runner at your own video dataset.

> **Reproducibility:** each image subproject ships a locked `environment.yml` (`conda env export`).
> It pins every transitive version but is a **record, not a one-command rebuild** — `torch` (cu128
> index), `transformers` (git/pinned), and the editable `lmms-eval` are not on default channels, so
> install those per the README first; the `.yml` then pins the rest.

## Unified Runner

Once the env(s) exist, [`vtr_run.py`](vtr_run.py) is a single CLI for **any model × method**. It does
not load models itself — it builds the right command and dispatches it into the matching conda env
(`conda run -n <env>`). See [RUNNER.md](RUNNER.md) for the full capability matrix, per-method
hyperparameters, and flags.

```bash
python vtr_run.py --list                       # capability matrix; marks each env ✓ present / ✗ missing

# PriorTR on Qwen3-VL (image), 2/9 keep ratio
python vtr_run.py --model qwen3vl --method priortr --tasks mme --keep-ratio 0.2222 --gpus 0

# PriorTR-2F on Video-LLaVA (video: a dataset instead of --tasks)
python vtr_run.py --model video-llava --method priortr_2f \
    --video-dir /data/MSVD/videos --gt-question /data/MSVD/test_q.json \
    --gt-answers /data/MSVD/test_a.json --keep-tokens 64 --num-samples 500 --gpus 0
```

The runner translates unified flags into each subproject's own argument names. If your conda envs
are named differently, point it at them with `--env <name>` or an `envs.json` map — no code edits.
Prefer the per-subproject README commands directly? Those still work; the runner is just a uniform
front-end over them.

## Citation

```bibtex
@inproceedings{priortr2026,
    title     = {Accelerating Multimodal Large Language Models with Prior-Corrected Token Reduction},
    author    = {Zengjie Chen and Yuxiang Cai and Jingcai Guo and Taotao Cai and Jianwei Yin and Zhi Chen},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year      = {2026}
}
```

## License

This repository contains code built on multiple open-source projects. Each subproject retains the license of its base model:

| Subproject | Base Model License |
|---|---|
| LLaVA | Apache 2.0 |
| InternVL | MIT |
| Qwen3-VL | Apache 2.0 |
| Video-LLaVA | Apache 2.0 |

The PriorTR-specific code (VTR framework, strategies, model wrappers) in this repository is released under the [Apache 2.0 License](LICENSE).
