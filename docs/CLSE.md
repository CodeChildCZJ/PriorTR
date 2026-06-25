<div align="center">
<h2>CLSE on PriorTR — Cross-Layer Spectral Evolution token pruning</h2>
<p><b>A training-free, progressive visual-token pruning method integrated into the PriorTR framework as a drop-in strategy (<code>strategy=clse</code>).</b> CLSE scores image tokens by how their spectral (frequency-domain) content evolves across decoder layers, combined with text→image attention, and prunes in three progressive stages.</p>
<p>
  <img src="https://img.shields.io/badge/method-CLSE-3776AB" alt="method">
  <img src="https://img.shields.io/badge/models-LLaVA%C2%B7Qwen2--VL%C2%B7Qwen3--VL-44A833" alt="models">
  <img src="https://img.shields.io/badge/reference-CLSE%20(ECCV%202026)-FFD21E" alt="ref">
</p>
</div>

> 🧩 Part of [**PriorTR**](../README.md) · [unified runner](RUNNER.md) · reference: [CLSE](https://github.com/zjubinchen/CLSE)

CLSE is integrated into three PriorTR backbones as one more pluggable strategy. The pruning
*mechanics* (where to prune, how to score, how to physically drop tokens) live in the shared VTR
framework; CLSE only supplies the spectral-evolution score and a hard-coded **3-stage keep schedule**.

| Backbone | wrapper (`--model`) | schedule knob | conda env | transformers |
|---|---|---|---|---|
| **LLaVA-1.5/1.6** | `llava_vtr` | `keep_tokens` (192 / 128 / 64) | see [LLaVA README](../image/LLaVA/README.md) | 4.37.2 |
| **Qwen2-VL-7B** | `qwen2_vl_vtr` | `vtr_retain_ratio` (0.334 / 0.223 / 0.112) | *(setup below)* | 4.57.x |
| **Qwen3-VL-8B** | `qwen3_vl_vtr` | `vtr_retain_ratio` (0.334 / 0.223 / 0.112) | see [Qwen3-VL README](../image/Qwen3-VL/README.md) | 5.2.0.dev0 |

---

## ⚙️ Environment Setup

CLSE reuses each backbone's standard PriorTR environment — **no extra packages**. Build the env from
the per-model README, then `strategy=clse` is available immediately.

- **LLaVA** → [image/LLaVA/README.md](../image/LLaVA/README.md) (see *Environment Setup*)
- **Qwen3-VL** → [image/Qwen3-VL/README.md](../image/Qwen3-VL/README.md) (see *Environment Setup*)
- **Qwen2-VL** → there is no separate README yet; the full setup is inline below.

### Qwen2-VL setup (inline)

```bash
conda create -n priortr-qwen2vl python=3.10 -y -c conda-forge --override-channels
conda activate priortr-qwen2vl

# 1. PyTorch — cu121 for standard GPUs, cu128 for Blackwell / RTX PRO (SM_120+)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#   or: --index-url https://download.pytorch.org/whl/cu128

# 2. Qwen2-VL stack (DynamicCache `.layers` API requires transformers >= 4.57)
pip install "transformers==4.57.*" accelerate qwen-vl-utils pillow decord
```

> ✅ **No transformers patch needed.** `VTRQwen2VLForConditionalGeneration` sets the visual mask and
> image grid on its (prunable) text model itself inside `forward`, so the stock transformers package
> works as-is — plain `pip install` only.

**Verify**

```bash
cd <repo>/image/Qwen2-VL
PYTHONPATH=$PWD python -c "
import torch, transformers
print(torch.__version__, transformers.__version__)             # expect transformers 4.57.x
from visual_token_pruning import VTRConfig
from visual_token_pruning.model import VTRQwen2VLForConditionalGeneration
print('Qwen2-VL VTR OK')
"
```

## 📦 lmms-eval Setup

CLSE is evaluated through the same `*_vtr` lmms-eval wrappers as the rest of PriorTR. Each per-model
README has the full steps; the short version:

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
cd lmms-eval && pip install -e . --no-deps && cd ..   # --no-deps keeps the pinned transformers

# runtime deps
pip install jiwer rouge_score nltk sacrebleu evaluate datasets loguru tenacity spacy
pip install pycocoevalcap texttable protobuf sqlitedict openai pytablewriter

# register the wrapper for the backbone you want (pick one)
cp image/LLaVA/lmms_eval_model/llava_vtr.py     lmms-eval/lmms_eval/models/simple/
cp image/Qwen2-VL/lmms_eval_model/qwen2_vl_vtr.py lmms-eval/lmms_eval/models/simple/
cp image/Qwen3-VL/lmms_eval_model/qwen3_vl_vtr.py lmms-eval/lmms_eval/models/simple/
```

Then register the class in `lmms-eval/lmms_eval/models/__init__.py` (`AVAILABLE_SIMPLE_MODELS`):

```python
"llava_vtr": "LlavaVTR",
"qwen2_vl_vtr": "Qwen2_VL_VTR",
"qwen3_vl_vtr": "Qwen3_VL_VTR",
```

> **Qwen2-VL only:** the wrapper imports `visual_token_pruning` from the project, so run lmms-eval with
> `PYTHONPATH=<repo>/image/Qwen2-VL` set.

## 🚀 Running CLSE

You select a CLSE budget with **one knob** — the framework expands it into the hard-coded 3-stage
schedule (see *Hard-coded Schedule & Knobs* below). You do **not** pass per-stage ratios by hand.

### LLaVA — knob = `keep_tokens` (192 / 128 / 64)

```bash
cd lmms-eval
TASKS=gqa,mme,pope,textvqa_val

# Baseline (no pruning)
python -m lmms_eval --model llava_vtr \
    --model_args pretrained=liuhaotian/llava-v1.5-7b,enabled=False \
    --tasks $TASKS --batch_size 1 --output_path ./results/baseline

# CLSE @ 192 tokens   (swap keep_tokens=128 or 64 for the other budgets)
python -m lmms_eval --model llava_vtr \
    --model_args "pretrained=liuhaotian/llava-v1.5-7b,strategy=clse,prune_layer=[1,11,21],ref_layers=[0],keep_tokens=192" \
    --tasks $TASKS --batch_size 1 --output_path ./results/clse_192
```

### Qwen2-VL — knob = `vtr_retain_ratio` (0.334 / 0.223 / 0.112)

```bash
cd lmms-eval
M=Qwen/Qwen2-VL-7B-Instruct
export PYTHONPATH=<repo>/image/Qwen2-VL          # so the wrapper finds visual_token_pruning

# Baseline (no pruning)
python -m lmms_eval --model qwen2_vl_vtr \
    --model_args "pretrained=$M,vtr_enabled=False" \
    --tasks mme,gqa --batch_size 1 --output_path ./results/qwen2_baseline

# CLSE @ retain_ratio=0.334  (≈ the "192-token" preset; final stage keeps 9.8% of tokens)
python -m lmms_eval --model qwen2_vl_vtr \
    --model_args "pretrained=$M,vtr_strategy=clse,vtr_prune_layer=1;10;19,vtr_retain_ratio=0.334" \
    --tasks mme,gqa --batch_size 1 --output_path ./results/qwen2_clse_0.334
# other presets: vtr_retain_ratio=0.223 (128-eq) or 0.112 (64-eq, ≈ the 11.1% budget)
```

### Qwen3-VL — knob = `vtr_retain_ratio` (same presets)

```bash
cd lmms-eval
M=Qwen/Qwen3-VL-8B-Instruct

# prune_layer is depth-scaled: Qwen3-VL-8B has 36 decoder layers (vs 28 for Qwen2-VL,
# 32 for LLaVA), so the stages sit at 1;13;24 (≈ the same 0.36 / 0.67 depth fractions).
python -m lmms_eval --model qwen3_vl_vtr \
    --model_args "pretrained=$M,attn_implementation=sdpa,vtr_strategy=clse,vtr_prune_layer=1;13;24,vtr_retain_ratio=0.334" \
    --tasks mme,gqa --batch_size 1 --output_path ./results/qwen3_clse_0.334
```

> **List syntax differs by wrapper.** LLaVA uses bracket lists (`prune_layer=[1,11,21]`); the Qwen
> wrappers use `;`-separated lists (`vtr_prune_layer=1;10;19`) and the `vtr_` prefix.

## 🎛️ Hard-coded Schedule & Knobs

CLSE has three layers of hard-coding. **Layer 1 (the schedule) is fully driven by one config knob;
layers 2–3 are fixed method hyper-parameters** baked into the strategy.

### 1. The 3-stage keep schedule (the one knob)

Pruning happens in 3 progressive stages at `prune_layer`, using reference features snapshotted at
`ref_layers=[0]`. A single budget knob picks the per-stage keep counts from a hard-coded dict.

**`prune_layer` scales with the LLM's decoder depth** — the stages sit at roughly the same depth
fractions (~0.36 / ~0.67) across backbones, so the absolute layer indices differ:

| Backbone | decoder layers | `prune_layer` (stages) |
|---|:---:|:---:|
| LLaVA-1.5 | 32 | `[1, 11, 21]` |
| Qwen2-VL-7B | 28 | `[1, 10, 19]` |
| Qwen3-VL-8B | 36 | `[1, 13, 24]` |

`prune_layer` is a plain config value (`prune_layer=...` / `vtr_prune_layer=...`), not a buried
constant — adjust it if you run a backbone with a different layer count.

**LLaVA — `_TOKEN_DICT`, keyed by `keep_tokens`** (absolute counts; fixed 24×24 = 576-token grid):

| `keep_tokens` | stage 0 | stage 1 | stage 2 (final) |
|:---:|:---:|:---:|:---:|
| **192** | 330 | 210 | 62 |
| **128** | 220 | 140 | 41 |
| **64**  | 110 | 70  | 20 |

*Arbitrary `n` (not in the table) → fallback `[int(n·1.72), int(n·1.09), int(n·0.32)]`.*

**Qwen2-VL / Qwen3-VL — `_RATIO_DICT`, keyed by `vtr_retain_ratio`** (per-stage ratios **of the
original visual length**; Qwen images vary in size, so ratios not absolute counts):

```python
_RATIO_DICT = {
    0.334: [0.57, 0.36, 0.098],   # "192-equivalent" preset  → final stage keeps 9.8% of tokens
    0.223: [0.38, 0.24, 0.066],   # "128-equivalent"         → final 6.6%
    0.112: [0.19, 0.12, 0.034],   # "64-equivalent" (≈11.1%) → final 3.4%
}
```

Keep count at stage *s* = `int(original_visual_len · schedule[s])` (floor — this exactly reproduces
the original CLSE-Qwen2-VL per-stage budget). *Arbitrary `r` → fallback `[r·1.72, r·1.09, r·0.32]`.*

> **`retain_ratio` is a nominal label, not the final keep fraction.** `0.334 / 0.223 / 0.112` mirror
> LLaVA's `192 / 128 / 64` (≈ 1/3, 2/9, 1/9 of 576). The *actual* fraction of tokens surviving the
> last stage is the third schedule entry (`0.098 / 0.066 / 0.034`).

If `retain_ratio` / `keep_tokens` is **unset**, CLSE falls back to the framework's generic
`keep_ratio` (ratio-of-current) — for power users who want to pass per-stage ratios directly.

**Cross-model symmetry.** Both knobs work on every backbone and select the same preset:
`retain_ratio=0.334 ≡ keep_tokens=192`, `0.223 ≡ 128`, `0.112 ≡ 64`. Use whichever you remember —
the LLaVA-style token budget or the Qwen-style ratio — on any model. (A list-valued `keep_tokens`,
e.g. `145;92;25`, is taken literally as explicit per-stage counts, not as a preset.)

### 2. Spectral-scoring constants

**Configurable** (defaults match the published method) via `clse_cutoff_ratio` / `clse_temp`
(`vtr_clse_cutoff_ratio` / `vtr_clse_temp` on the Qwen wrappers):

| Knob | Default | Role |
|---|:---:|---|
| `clse_cutoff_ratio` | `0.1` | 2D-FFT high-pass cutoff (Gaussian `1 − exp(−d²/2σ²)`, `σ = min(cH,cW)·cutoff`) |
| `clse_temp` | `0.1` | sigmoid temperature in the evolution factor |

**Fixed in the strategy:** `epsilon = 1e-6` (divide-by-zero guards) and `clamp(evo, max=1)` (upper
bound on the per-token evolution rate).

### 3. Structural choices (fixed in the strategy)

- `SCORE_TYPE = "clse_attn"` → final score = `evolution × attention` (alternative `"clse"` = evolution
  only is implemented but unused).
- **Attention term** = mean over heads of the **last query token's** attention to image tokens.
- **Spectral term only at stage 0** (the full grid is intact); later stages are attention-only,
  because pruning breaks the 2D grid the FFT needs.
- Grid: LLaVA fixed **24×24**; Qwen derived from `image_grid_thw` (`H//2, W//2` after the 2×2 merge).

These structural choices are fixed in the strategy file (`.../strategy/clse.py`); edit there to
change them. The two numeric spectral hyper-parameters (`clse_cutoff_ratio`, `clse_temp`) are
exposed as config — see section 2.

## ✅ Reference numbers (full MME)

| Backbone | budget | vanilla | CLSE | retention |
|---|---|:---:|:---:|:---:|
| Qwen2-VL-7B | `retain_ratio=0.334` (~10% tokens) | 2313.67 | **2305.58** | **99.65%** |
| Qwen3-VL-8B | `retain_ratio=0.334`, `prune_layer=1;13;24` | 2389.01 | **2250.12** | **94.2%** |
| LLaVA-1.5-7B | `keep_tokens=192/128/64` | — | within ±0.4% of original | — |

Notes:
- **Qwen2-VL** numbers are on **stock `pip install transformers`** (this self-contained build, no
  patch). The original CLSE-Qwen2-VL implements pruning by rewriting `Qwen2VLTextModel` *in-place*
  (env-var driven) and reports **2284.81 (98.6%)**. Running this framework's **identical code** in the
  original's modified transformers reproduces **2284.81 bit-for-bit**; on stock transformers it lands
  at **2305.58 (99.65%)**. The ~21-point gap is **entirely the transformers environment**, not the
  pruning code — the modified build even shifts the *un-pruned* baseline (vanilla 2316.67 patched vs
  2313.67 stock), and that difference amplifies under pruning. So the framework is both faithful
  (reproduces the original exactly in its env) and cleaner (higher retention on stock).
  GQA `exact_match` 0.6088–0.6096.
- **Qwen3-VL** is a cross-model port (CLSE is native to Qwen2-VL). `prune_layer=1;13;24` is
  depth-aligned to its 36 decoder layers and beats the naive `1;10;19` (2180.32 / 91.3%) by +2.9%.
- **LLaVA** runs on PriorTR's prunable engine; GQA +0.08% vs the original.

## 📄 Credit & License

CLSE method © its authors — [zjubinchen/CLSE](https://github.com/zjubinchen/CLSE) (ECCV 2026). This is
an integration into the PriorTR framework, released under the Apache 2.0 License (root
[LICENSE](../LICENSE)).
