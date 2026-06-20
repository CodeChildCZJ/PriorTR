# Adding a New Method

PriorTR's Visual Token Reduction (VTR) framework uses a **strategy pattern**: a token-reduction method
is a small class that computes an importance score per visual token. The framework handles everything
else — selecting Top-K, physically dropping tokens, and re-plumbing the attention mask, position IDs,
KV cache, and RoPE. **You do not touch the model forward, the cache, or the masking.**

Paths below are for LLaVA (`image/LLaVA/llava/vtr/`); InternVL (`internvl_vtr/`), Qwen3-VL
(`visual_token_pruning/`), and Video-LLaVA (`videollava/vtr/`) mirror the same shape.

## The 3-step recipe (attention-score methods)

**1. Create `llava/vtr/strategy/mymethod.py`** — implement the one required method, `compute_scores`:

```python
from .base import PruningStrategy
from .registry import register_strategy

@register_strategy("mymethod")                        # ← the name you pass as strategy=mymethod
class MyStrategy(PruningStrategy):
    def compute_scores(self, attention, image_token_range, config, **ctx):
        # attention:          [batch, heads, seq, seq] at the prune layer
        # image_token_range:  (img_start, img_end)
        # return:             one score per image token, shape [num_img]
        img_start, img_end = image_token_range
        return attention[:, :, -1, img_start:img_end].mean(dim=1).squeeze(0)
```

**2. Register it** — add one import to `llava/vtr/strategy/__init__.py` so the decorator runs:

```python
from .mymethod import MyStrategy
```

**3. Run it** — select your strategy by name:

```bash
python -m lmms_eval --model llava_vtr \
    --model_args pretrained=liuhaotian/llava-v1.5-7b,strategy=mymethod \
    --tasks mme --batch_size 1 --output_path ./results/mymethod
```

That's the whole loop. A FastV-style "rank by attention magnitude" method is essentially one line in
`compute_scores`.

## What you inherit (reuse, don't re-implement)

`PruningStrategy` (`strategy/base.py`) already provides:

- **`select_tokens(scores, num_tokens, config)`** — turns your scores into kept indices, honoring
  `keep_tokens` → `score_threshold` → `keep_ratio` (in that priority) and preserving original order.
  The framework calls this for you; you never write Top-K logic.
- **`_aggregate_attention(attention, image_token_range, config)`** — collapses the attention matrix to
  per-token scores using `query_aggregation` (`last` / `question`) and `head_aggregation`
  (`mean` / `max`). Call it inside `compute_scores` if you just want pre-aggregated attention:
  `scores = self._aggregate_attention(attention, image_token_range, config)`.

You can also **subclass an existing strategy** (e.g. `class MyVariant(PriorTRStrategy)`) and override a
piece of the scoring.

## Adding hyperparameters

Add a field to `VTRConfig` (`vtr/config.py`):

```python
@dataclass
class VTRConfig:
    ...
    my_temperature: float = 1.0      # your new knob
```

Read it in your strategy as `config.my_temperature`. It is automatically passable via
`--model_args ...,my_temperature=0.7` (image models) or the matching `--vtr_*` flag (Video-LLaVA).

## The contract

```
compute_scores(attention, image_token_range, config, **ctx) -> Tensor[num_img]
```

- `attention` — `[batch, heads, seq, seq]` from the **prune layer** (the framework forces this layer to
  output attention via eager/sdpa; `flash_attention_2` cannot return weights, so use `sdpa`/`eager`).
- `image_token_range` — `(img_start, img_end)`; tokens before/after the image are always kept.
- `config` — the active `VTRConfig`.
- `**ctx` — extra per-call context (see below).
- **Return** — one score per image token; higher = more important. The framework keeps the Top-K and
  physically prunes the rest, then continues from the next layer on the shortened sequence.

## Advanced: multi-layer, per-layer strategy, and cross-layer caching

### Multi-layer pruning — built in

Pass a list: `prune_layer=[3, 7, 16]`. The framework prunes at each layer in turn, re-computing the
image-token range and re-plumbing the mask / position IDs / KV cache / RoPE between layers. **Every
prune layer calls the same strategy** (there is a single global `self._vtr_strategy`).

### A different strategy per layer — a small extension

Not built in, but easy. Write a **composite/router** strategy that holds several sub-strategies and
dispatches by layer. The only missing piece is the current layer index, which `compute_scores` does not
receive today — route it through `ctx`:

```python
# in prunable_llama.py, where attention is grabbed in the layer loop, add one line:
vtr_ctx["layer_idx"] = layer_idx          # before _compute_and_apply_pruning(...)
```

```python
@register_strategy("router")
class RouterStrategy(PruningStrategy):
    def __init__(self):
        self.by_layer = {3: PriorTRStrategy(), 7: FastVStrategy()}   # layer -> strategy
    def compute_scores(self, attention, image_token_range, config, **ctx):
        sub = self.by_layer[ctx["layer_idx"]]
        return sub.compute_scores(attention, image_token_range, config, **ctx)
```

### Caching features across layers — hook points exist, route the data

The framework gives you two places to keep state across the prune layers of a single forward:

- **`vtr_ctx`** is **one shared dict** threaded through every prune layer of a forward. Put a mutable
  container in it (e.g. `vtr_ctx["cache"] = {}`) and your strategy can write at layer 3 and read at
  layer 7.
- The **strategy instance** (`self._vtr_strategy`) persists for the model's lifetime, so it can stash
  `self._cache = ...`. **Reset it at the start of each forward**, or state leaks across calls.

The one gap: `compute_scores` currently receives only the **attention matrix**, not the layer's
**hidden states / features**. Those are available at the call site
(`_compute_and_apply_pruning(hidden_states=..., ...)`) but not forwarded. If your method needs to cache
*features* (not just attention), add one line to route them in:

```python
# in _compute_and_apply_pruning, before calling compute_scores:
vtr_ctx["hidden_states"] = hidden_states
```

then read `ctx["hidden_states"]` in your strategy.

### For sophisticated methods, start from Qwen3-VL

If your method needs a second forward, pre-LLM pruning, token merging, or multi-source signals, the
**Qwen3-VL** subproject (`image/Qwen3-VL/visual_token_pruning/`) is the richest template — it already
implements a two-forward prior (`model/prior_utils.py`), VisPruner-style pre-LLM pruning, token merging
(`model/token_merge.py`), and multi-source handling (`model/deepstack_handler.py`). Copy the closest
existing strategy there rather than extending LLaVA's minimal framework.

## Note: the per-model frameworks are parallel copies

Each model ships its own VTR framework (`llava/vtr`, `internvl_vtr`, `visual_token_pruning`,
`videollava/vtr`) — they are not yet a shared core. A strategy added to LLaVA only affects LLaVA; to
support several backbones, drop an equivalent strategy file into each (the interfaces are nearly
identical). A model-agnostic `vtr-core` extraction is planned for after publication.
