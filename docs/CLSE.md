<div align="center">
<h2>CLSE on PriorTR — Cross-Layer Spectral Evolution token pruning</h2>
<p><b>Training-free, progressive visual-token pruning, integrated into PriorTR as a drop-in strategy (<code>strategy=clse</code>).</b> Tokens are scored by how their spectral (frequency-domain) content evolves across decoder layers × text→image attention, then pruned in three progressive stages.</p>
<p>
  <img src="https://img.shields.io/badge/method-CLSE-3776AB" alt="method">
  <img src="https://img.shields.io/badge/models-LLaVA%C2%B7Qwen2--VL%C2%B7Qwen3--VL-44A833" alt="models">
</p>
</div>

> 🧩 Part of [**PriorTR**](../README.md) · [unified runner](RUNNER.md) · reference: [CLSE](https://github.com/zjubinchen/CLSE)

| Backbone | wrapper (`--model`) | budget knob | env · transformers |
|---|---|---|---|
| **LLaVA-1.5/1.6** | `llava_vtr` | `keep_tokens` = 192 / 128 / 64 | [README](../image/LLaVA/README.md) · 4.37 |
| **Qwen2-VL-7B** | `qwen2_vl_vtr` | `vtr_retain_ratio` = 0.334 / 0.223 / 0.112 | [README](../image/Qwen2-VL/README.md) · 4.57 |
| **Qwen3-VL-8B** | `qwen3_vl_vtr` | `vtr_retain_ratio` = 0.334 / 0.223 / 0.112 | [README](../image/Qwen3-VL/README.md) · 5.2 |

---

## ⚙️ Setup

CLSE needs **no extra packages**. Build each backbone's standard PriorTR env and register its
lmms-eval wrapper as described in its README — [LLaVA](../image/LLaVA/README.md) ·
[Qwen2-VL](../image/Qwen2-VL/README.md) · [Qwen3-VL](../image/Qwen3-VL/README.md) — then `strategy=clse`
is available immediately. (Qwen2-VL uses **stock** transformers with no patch; run its wrapper with
`PYTHONPATH=<repo>/image/Qwen2-VL`.)

## 🚀 Running

**Just `strategy=clse` + one budget knob.** The 3 prune stages and the spectral reference layer
auto-resolve from the model's decoder depth — you don't pass them.

```bash
# LLaVA-1.5      keep_tokens = 192 | 128 | 64
python -m lmms_eval --model llava_vtr \
  --model_args "pretrained=liuhaotian/llava-v1.5-7b,strategy=clse,keep_tokens=192" \
  --tasks gqa,mme,pope,textvqa_val --batch_size 1 --output_path ./results/llava_clse

# Qwen2-VL       vtr_retain_ratio = 0.334 | 0.223 | 0.112
PYTHONPATH=<repo>/image/Qwen2-VL python -m lmms_eval --model qwen2_vl_vtr \
  --model_args "pretrained=Qwen/Qwen2-VL-7B-Instruct,vtr_strategy=clse,vtr_retain_ratio=0.334" \
  --tasks mme,gqa --batch_size 1 --output_path ./results/qwen2_clse

# Qwen3-VL       same presets
python -m lmms_eval --model qwen3_vl_vtr \
  --model_args "pretrained=Qwen/Qwen3-VL-8B-Instruct,attn_implementation=sdpa,vtr_strategy=clse,vtr_retain_ratio=0.334" \
  --tasks mme,gqa --batch_size 1 --output_path ./results/qwen3_clse
```

Baseline (no pruning): `enabled=False` (LLaVA) / `vtr_enabled=False` (Qwen). Override the auto layers
only for a different model size: `prune_layer=[1,11,21],ref_layers=[0]` (LLaVA, brackets) or
`vtr_prune_layer=1;10;19` (Qwen, `;`) — an explicit value always wins.

## 🎛️ Knobs

`keep_tokens` and `vtr_retain_ratio` are interchangeable on every backbone; each selects one
hard-coded 3-stage keep schedule (the original CLSE budgets, floored per stage):

| budget | `keep_tokens` | `vtr_retain_ratio` | LLaVA stages / 576 | Qwen stages (of orig. len) |
|:---:|:---:|:---:|:---:|:---:|
| high | 192 | 0.334 | 330 / 210 / 62 | 0.57 / 0.36 / 0.098 |
| mid  | 128 | 0.223 | 220 / 140 / 41 | 0.38 / 0.24 / 0.066 |
| low  | 64  | 0.112 | 110 / 70 / 20  | 0.19 / 0.12 / 0.034 |

**Prune layers** (auto, from decoder depth): LLaVA-32 `[1,11,21]` · Qwen2-VL-28 `[1,10,19]` ·
Qwen3-VL-36 `[1,13,24]`. The first two are the original CLSE schedules; Qwen3-VL (no upstream CLSE)
is our depth-aligned port. Override with `prune_layer` / `vtr_prune_layer` for other sizes.

**Spectral hyper-params** `clse_cutoff_ratio` / `clse_temp` (both `0.1`) are config-tunable
(`vtr_clse_*` on Qwen); other choices (score = evolution × attention, spectral only at stage 0) are
fixed in `strategy/clse.py`.

## 📄 Credit

CLSE © [zjubinchen/CLSE](https://github.com/zjubinchen/CLSE) (ECCV 2026). Integration into PriorTR,
released under the Apache 2.0 License (root [LICENSE](../LICENSE)).
