# PriorTR: Prior-Corrected Visual Token Reduction

Official code for **"Accelerating Multimodal Large Language Models with Prior-Corrected Token Reduction"** (ECCV 2026).

> **TL;DR**: Attention-based visual token pruning is dominated by a model-induced prior. PriorTR corrects this by contrasting task-conditioned attention (P) with an instruction-agnostic prior (Q) estimated from a null token within a single forward pass, scoring tokens via V-Information: `S = P * log(P / Q)`.

## Overview

Visual token reduction accelerates MLLMs by pruning redundant image tokens at an early decoder layer. Existing methods rank tokens by raw attention magnitude, but we show this ranking is confounded by a **model-induced prior** — the model attends to certain regions even without any instruction. PriorTR explicitly disentangles this prior from instruction-conditioned evidence using the causal attention structure: the null token (e.g., `\n`) after the image cannot see instruction tokens under the causal mask, making its attention a natural prior estimate. The top-K tokens by V-Information score are physically retained, reducing computation for all subsequent layers.

## Supported Models

| Model | Path | Conda Env | Strategies |
|---|---|---|---|
| LLaVA-1.5 | [`image/LLaVA/`](image/LLaVA/) | `PriorTRllava` | PriorTR |
| InternVL2.5 | [`image/InternVL/`](image/InternVL/) | `PriorTRinternvl` | PriorTR, FastV |

Each subproject has its own README with environment setup, usage, and evaluation commands.

## Repository Structure

```
.
├── image/
│   ├── LLaVA/                 # LLaVA-1.5-7B / 13B
│   └── InternVL/              # InternVL2.5-8B
├── .gitignore
├── LICENSE
└── README.md
```

## Quick Start

1. **Pick a model** from the table above and `cd` into its directory.
2. **Create the conda environment** following the subproject README.
3. **Set up lmms-eval.** Benchmark evaluation uses [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), which is **not bundled in this repository**. Each subproject README has an "lmms-eval Setup" section that clones it into the subproject directory and registers the model wrapper.
4. **Run evaluation** with the provided commands.

For example, to evaluate PriorTR on InternVL2.5 with MME (after completing the lmms-eval setup under `image/InternVL/`):

```bash
cd image/InternVL/lmms-eval
export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 lmms-eval \
    --model internvl_vtr \
    --model_args "pretrained=OpenGVLab/InternVL2_5-8B,strategy=priortr,keep_tokens=192,prune_layer=2" \
    --tasks mme --batch_size 1 \
    --output_path ../eval_results/priortr_192_l2_mme
```

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

The PriorTR-specific code (VTR framework, strategies, model wrappers) in this repository is released under the [Apache 2.0 License](LICENSE).
