<div align="center">
<h2>CLSE on PriorTR — Cross-Layer Spectral Evolution token pruning</h2>
<p><b>A training-free, progressive visual-token pruning method, integrated into PriorTR as a drop-in strategy (<code>strategy=clse</code>).</b> CLSE scores image tokens by how their spectral (frequency-domain) content evolves across decoder layers, combined with text→image attention, and prunes in three progressive stages.</p>
<p>
  <img src="https://img.shields.io/badge/method-CLSE-3776AB" alt="method">
  <img src="https://img.shields.io/badge/models-LLaVA%C2%B7Qwen2--VL%C2%B7Qwen3--VL-44A833" alt="models">
</p>
</div>

> 🧩 Part of [**PriorTR**](../README.md) · [unified runner](RUNNER.md) · reference: [CLSE](https://github.com/zjubinchen/CLSE)

| Backbone | wrapper (`--model`) | budget knob | conda env | transformers |
|---|---|---|---|---|
| **LLaVA-1.5/1.6** | `llava_vtr` | `keep_tokens` = 192 / 128 / 64 | [LLaVA README](../image/LLaVA/README.md) | 4.37.2 |
| **Qwen2-VL-7B** | `qwen2_vl_vtr` | `vtr_retain_ratio` = 0.334 / 0.223 / 0.112 | *(below)* | 4.57.x |
| **Qwen3-VL-8B** | `qwen3_vl_vtr` | `vtr_retain_ratio` = 0.334 / 0.223 / 0.112 | [Qwen3-VL README](../image/Qwen3-VL/README.md) | 5.2.0.dev0 |

---

## ⚙️ Environment Setup

CLSE adds **no extra packages** — build each backbone's standard PriorTR env and `strategy=clse`
is available. LLaVA / Qwen3-VL: see their READMEs. Qwen2-VL (no separate README) is inline:

```bash
conda create -n priortr-qwen2vl python=3.10 -y -c conda-forge --override-channels
conda activate priortr-qwen2vl
# PyTorch — cu121 for standard GPUs, cu128 for Blackwell / RTX PRO (SM_120+)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# Qwen2-VL stack (DynamicCache `.layers` API needs transformers >= 4.57)
pip install "transformers==4.57.*" accelerate qwen-vl-utils pillow decord
```

> ✅ **No transformers patch needed** — `VTRQwen2VLForConditionalGeneration` sets the visual mask /
> image grid on its text model itself, so stock `pip install transformers` works as-is.

**lmms-eval** (image models are evaluated through the `*_vtr` wrappers):

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
cd lmms-eval && pip install -e . --no-deps && cd ..      # --no-deps keeps the pinned transformers
pip install jiwer rouge_score nltk sacrebleu evaluate datasets loguru tenacity spacy \
            pycocoevalcap texttable protobuf sqlitedict openai pytablewriter

# register the wrapper(s) you need
cp image/Qwen2-VL/lmms_eval_model/qwen2_vl_vtr.py lmms-eval/lmms_eval/models/simple/
# then add to lmms-eval/lmms_eval/models/__init__.py (AVAILABLE_SIMPLE_MODELS):
#   "llava_vtr": "LlavaVTR", "qwen2_vl_vtr": "Qwen2_VL_VTR", "qwen3_vl_vtr": "Qwen3_VL_VTR"
```

> **Qwen2-VL only:** run lmms-eval with `PYTHONPATH=<repo>/image/Qwen2-VL` so the wrapper finds
> `visual_token_pruning`.

## 🚀 Running CLSE

CLSE needs **just two things: `strategy=clse` and one budget knob.** The 3 prune stages and the
spectral reference layer auto-resolve from the model's decoder depth — you don't pass them.

```bash
cd lmms-eval

# LLaVA-1.5      (keep_tokens = 192 | 128 | 64)
python -m lmms_eval --model llava_vtr \
    --model_args "pretrained=liuhaotian/llava-v1.5-7b,strategy=clse,keep_tokens=192" \
    --tasks gqa,mme,pope,textvqa_val --batch_size 1 --output_path ./results/llava_clse

# Qwen2-VL       (vtr_retain_ratio = 0.334 | 0.223 | 0.112)
PYTHONPATH=<repo>/image/Qwen2-VL python -m lmms_eval --model qwen2_vl_vtr \
    --model_args "pretrained=Qwen/Qwen2-VL-7B-Instruct,vtr_strategy=clse,vtr_retain_ratio=0.334" \
    --tasks mme,gqa --batch_size 1 --output_path ./results/qwen2_clse

# Qwen3-VL       (same presets)
python -m lmms_eval --model qwen3_vl_vtr \
    --model_args "pretrained=Qwen/Qwen3-VL-8B-Instruct,attn_implementation=sdpa,vtr_strategy=clse,vtr_retain_ratio=0.334" \
    --tasks mme,gqa --batch_size 1 --output_path ./results/qwen3_clse
```

Baseline (no pruning): pass `enabled=False` (LLaVA) / `vtr_enabled=False` (Qwen) instead.

> **Overriding the auto schedule** is optional — only needed for a different model size or to
> experiment. Add `prune_layer=[1,11,21],ref_layers=[0]` (LLaVA, bracket lists) or
> `vtr_prune_layer=1;10;19` (Qwen, `;`-separated). An explicit value always wins.

## 🎛️ Knobs & hard-coding

**Budget presets.** One knob expands into a hard-coded 3-stage keep schedule (the original CLSE
budgets). `keep_tokens` and `vtr_retain_ratio` are interchangeable on every backbone:

| budget | `keep_tokens` | `vtr_retain_ratio` | LLaVA per-stage (of 576) | Qwen per-stage (of orig. len) |
|:---:|:---:|:---:|:---:|:---:|
| high | 192 | 0.334 | 330 / 210 / 62 | 0.57 / 0.36 / 0.098 |
| mid  | 128 | 0.223 | 220 / 140 / 41 | 0.38 / 0.24 / 0.066 |
| low  | 64  | 0.112 | 110 / 70 / 20  | 0.19 / 0.12 / 0.034 |

Keep count = `int(len · schedule[stage])` (floor). The headline number is nominal (≈ 1/3, 2/9, 1/9 of
576); the final-stage survivor fraction is the third entry. An off-table value falls back to the same
3-stage shape scaled to it; unset → the framework's generic `keep_ratio`.

**Prune layers (auto).** The 3 stages sit at ~0.36 / 0.67 of decoder depth, resolved from the model's
layer count — so the defaults differ by backbone and you normally pass nothing:

| LLaVA-1.5 (32 layers) | Qwen2-VL-7B (28) | Qwen3-VL-8B (36) |
|:---:|:---:|:---:|
| `[1, 11, 21]` | `[1, 10, 19]` | `[1, 13, 24]` |

**Spectral hyper-params (optional).** `clse_cutoff_ratio` (`0.1`, 2D-FFT high-pass cutoff) and
`clse_temp` (`0.1`, evolution sigmoid temperature) match the published method; tune via config (or
`vtr_clse_*` on the Qwen wrappers). Structural choices — score = evolution × attention, spectral term
only at stage 0 (the grid is intact there) — are fixed in `strategy/clse.py`.

## ✅ Results (full MME)

| Backbone | budget | vanilla | CLSE | retention |
|---|---|:---:|:---:|:---:|
| Qwen2-VL-7B | `0.334` (~10% tokens) | 2313.67 | **2305.58** | **99.65%** |
| Qwen3-VL-8B | `0.334`, layers `1;13;24` | 2389.01 | **2250.12** | **94.2%** |
| LLaVA-1.5-7B | `192 / 128 / 64` | — | within **±0.4%** of original | — |

**In short:** keeping ~10% of tokens, CLSE holds **99.7%** of MME on Qwen2-VL and **94%** on Qwen3-VL,
and stays within **±0.4%** on LLaVA; GQA matches the original within **+0.1%**.

- **Qwen2-VL** numbers are on **stock** `pip install transformers`. The original CLSE-Qwen2-VL instead
  patches `Qwen2VLTextModel` in-place and reports **2284.81**. Running this framework's *identical*
  code in that patched env reproduces **2284.81 bit-for-bit**, so the ~21-pt gap is **the transformers
  environment, not the pruning code** (it shifts even the un-pruned baseline). The framework is thus
  both faithful (exact in the original's env) and cleaner (higher retention on stock).
- **Qwen3-VL** is a cross-model port (CLSE is native to Qwen2-VL); depth-aligned `1;13;24` beats the
  naive `1;10;19` (2180.32 / 91.3%) by **+2.9%**.

## 📄 Credit & License

CLSE method © its authors — [zjubinchen/CLSE](https://github.com/zjubinchen/CLSE) (ECCV 2026). This is
an integration into PriorTR, released under the Apache 2.0 License (root [LICENSE](../LICENSE)).
