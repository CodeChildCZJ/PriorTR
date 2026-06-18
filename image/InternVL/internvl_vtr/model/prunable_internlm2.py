# Prunable InternLM2Model
# Function-based monkeypatching for InternLM2Model to support multi-layer visual token pruning.
#
# Since InternLM2Model lives inside a trust_remote_code module and cannot be imported directly,
# we use a function-based approach: setup_vtr(model, config) patches the model instance in-place.
#
# Key differences from PrunableLlamaModel (LLaVA):
#   - Embedding:   self.tok_embeddings  (not self.embed_tokens)
#   - Cache:       OLD-STYLE TUPLE ((k0,v0),(k1,v1),...) NOT DynamicCache
#   - Attention:   decoder_layer.attention  (not decoder_layer.self_attn)
#   - No layer_idx on decoder_layer: use enumerate(self.layers) index
#   - RoPE:        InternLM2RotaryEmbedding.forward(x, seq_len) returns cos_cached[:seq_len]
#                  After pruning, physical kv_seq_len < original position_ids max -> IndexError
#                  Fix: monkeypatch forward() to return FULL cache (unbounded)
#   - DecoderLayer output:
#       output_attentions=False, use_cache=True  -> (hidden, present_kv)          kv@1
#       output_attentions=True,  use_cache=True  -> (hidden, attn, present_kv)    kv@2
from __future__ import annotations

import logging
import types
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
from transformers.modeling_outputs import BaseModelOutputWithPast

from ..config import VTRConfig
from ..strategy import PruningStrategy, get_strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Unbounded RoPE monkeypatch
# ---------------------------------------------------------------------------

def _make_unbounded_rope(old_rope) -> None:
    """
    Monkeypatch an InternLM2RotaryEmbedding instance so that its forward()
    returns the FULL cos/sin cache instead of slicing to [:seq_len].

    Why:
        After pruning, physical kv_seq_len (e.g. 200) can be much smaller than
        the original position_ids max (e.g. 1809).  The standard RoPE forward
        does ``cos_cached[:seq_len]``, so indexing ``cos[position_ids]`` where
        position_ids has values >= seq_len would raise an IndexError.
        Returning the full cache avoids this entirely.
    """
    if getattr(old_rope, '_is_unbounded', False):
        return  # already patched

    def _unbounded_forward(self, x, seq_len=None):
        cache_len = self.max_seq_len_cached
        if seq_len is not None and seq_len > cache_len:
            self._set_cos_sin_cache(seq_len, x.device, x.dtype)
            cache_len = self.max_seq_len_cached
        return (
            self.cos_cached[:cache_len].to(dtype=x.dtype),
            self.sin_cached[:cache_len].to(dtype=x.dtype),
        )

    old_rope.forward = types.MethodType(_unbounded_forward, old_rope)
    old_rope._is_unbounded = True


# ---------------------------------------------------------------------------
# 2. setup_vtr  --  main entry-point
# ---------------------------------------------------------------------------

def setup_vtr(model, config: VTRConfig) -> None:
    """
    Apply VTR (Visual Token Reduction) patches to an InternLM2Model instance.

    This replaces the model's ``forward`` method with a pruning-capable version
    and monkeypatches all RoPE modules to the unbounded variant.

    Args:
        model:  An InternLM2Model instance (the language backbone, NOT the
                outer InternLM2ForCausalLM wrapper).
        config: A VTRConfig object.
    """
    model._vtr_config = config
    model._vtr_strategy: Optional[PruningStrategy] = None
    model._current_image_token_range: Optional[Tuple[int, int]] = None

    # Always replace RoPE with unbounded version (safety: even when VTR is
    # disabled, sparse position_ids from external code could cause OOB).
    _replace_rope_with_unbounded(model)

    if config.enabled:
        model._vtr_strategy = get_strategy(config.strategy)
        logger.debug(
            "VTR enabled on InternLM2Model with strategy=%s, prune_layers=%s, "
            "keep_ratio=%s, keep_tokens=%s",
            config.strategy, config.prune_layers,
            config.keep_ratio, config.keep_tokens,
        )
    else:
        model._vtr_strategy = None

    # Save the original forward so the early-exit path can delegate to it.
    model._original_forward = model.forward

    # Bind the prunable forward as the new forward method.
    model.forward = types.MethodType(_prunable_forward, model)


# ---------------------------------------------------------------------------
# 3. Internal helpers
# ---------------------------------------------------------------------------

def _replace_rope_with_unbounded(model) -> None:
    """Replace RoPE on every decoder layer's attention module with unbounded version."""
    for layer in model.layers:
        _make_unbounded_rope(layer.attention.rotary_emb)


def _get_prune_layer_set(model) -> Set[int]:
    """
    Return the set of layer indices (K-1) where attention weights must be
    captured so that pruning can happen *after* that layer.

    ``prune_layers`` in VTRConfig are 1-indexed "prune after layer K" values;
    we need index K-1 (0-indexed) because that is the layer whose attention
    output we use.
    """
    cfg: Optional[VTRConfig] = getattr(model, '_vtr_config', None)
    if cfg is None:
        return set()
    return {layer - 1 for layer in cfg.prune_layers}


# ---------------------------------------------------------------------------
# 4. Prunable forward  (replaces InternLM2Model.forward)
# ---------------------------------------------------------------------------

def _prunable_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    # ----- VTR extra arguments (or injected via attributes) -----
    image_token_range: Optional[Tuple[int, int]] = None,
    vtr_ctx: Optional[dict] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    """
    Prunable replacement for ``InternLM2Model.forward``.

    When VTR is disabled (or no image_token_range is provided), this simply
    delegates to the original forward.  Otherwise it performs multi-layer
    visual-token pruning following the same logic as PrunableLlamaModel but
    adapted for the InternLM2 architecture.
    """

    # ---- Early exit if VTR is not active ----
    vtr_cfg: Optional[VTRConfig] = getattr(self, '_vtr_config', None)
    if vtr_cfg is None or not vtr_cfg.enabled:
        return self._original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    # ---- State injection support ----
    if image_token_range is None and hasattr(self, '_vtr_image_token_range'):
        image_token_range = self._vtr_image_token_range
    if vtr_ctx is None and hasattr(self, '_vtr_ctx'):
        vtr_ctx = self._vtr_ctx

    # ---- Defaults from config (mirrors InternLM2Model.forward lines 866-872) ----
    output_attentions = (
        output_attentions if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # ---- Retrieve input_ids / inputs_embeds ----
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError('You have to specify either input_ids or inputs_embeds')

    # ---- Past key values length (old-style tuple cache) ----
    seq_length_with_past = seq_length
    past_key_values_length = 0
    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length

    # ---- Position IDs ----
    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0)

    # ---- Embeddings ----
    if inputs_embeds is None:
        inputs_embeds = self.tok_embeddings(input_ids)

    # ---- Dynamic RoPE cache pre-expansion ----
    # Prevent IndexError when position_ids exceed current cache size.
    if position_ids is not None and position_ids.numel() > 0:
        current_max_pos = position_ids.max().item()
        rope_module = self.layers[0].attention.rotary_emb
        current_limit = getattr(
            rope_module, 'max_seq_len_cached', rope_module.cos_cached.shape[0]
        )
        if current_max_pos >= current_limit:
            new_limit = current_max_pos + 1024
            dtype = inputs_embeds.dtype if inputs_embeds is not None else torch.float16
            device = position_ids.device
            for layer in self.layers:
                layer.attention.rotary_emb._set_cos_sin_cache(
                    seq_len=new_limit, device=device, dtype=dtype,
                )

    # ---- Determine whether pruning should run ----
    should_prune = (
        vtr_cfg is not None
        and vtr_cfg.enabled
        and self._vtr_strategy is not None
        and image_token_range is not None
        and past_key_values_length == 0  # only prune during prefill
    )
    prune_layer_set = _get_prune_layer_set(self) if should_prune else set()
    vtr_ctx = vtr_ctx or {}
    current_image_range = image_token_range

    # ---- Attention mask ----
    # Always use eager attention (4D causal mask) so we can extract attention
    # weights on the K-1 layers.  flash_attention_2 cannot output weights.
    if self.config.attn_implementation == 'flash_attention_2':
        attention_mask = (
            attention_mask
            if (attention_mask is not None and 0 in attention_mask)
            else None
        )
    else:
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length,
        )

    # ---- Layer loop init ----
    hidden_states = inputs_embeds

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        # Decide whether we need attention weights from this layer for pruning
        need_attention_for_pruning = should_prune and idx in prune_layer_set
        layer_output_attentions = output_attentions or need_attention_for_pruning

        if need_attention_for_pruning:
            # Force eager attention on this layer: recompute 4D causal mask
            # sized to the *current* hidden_states length (may have been
            # pruned by an earlier pruning layer).
            layer_mask = self._prepare_decoder_attention_mask(
                None,
                (batch_size, hidden_states.shape[1]),
                hidden_states,
                past_key_values_length,
            )
            layer_output_attentions = True
        else:
            layer_mask = attention_mask

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=layer_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=layer_output_attentions,
            use_cache=use_cache,
        )

        hidden_states = layer_outputs[0]

        if use_cache:
            kv_idx = 2 if layer_output_attentions else 1
            next_decoder_cache += (layer_outputs[kv_idx],)

        # ---- Prune after K-1 layer ----
        if need_attention_for_pruning and current_image_range is not None:
            attention_weights = layer_outputs[1]
            if attention_weights is None:
                if self.config.attn_implementation == 'flash_attention_2':
                    raise ValueError(
                        "VTR Pruning error: flash_attention_2 does not support "
                        "outputting attention weights. Please use 'eager' "
                        "attn_implementation when loading the model."
                    )
                else:
                    raise ValueError(
                        "Attention weights are None but pruning is requested."
                    )

            (
                hidden_states,
                position_ids,
                attention_mask,
                next_decoder_cache,
                new_seq_len,
                new_image_range,
            ) = _compute_and_apply_pruning(
                model=self,
                hidden_states=hidden_states,
                attention_weights=attention_weights,
                position_ids=position_ids,
                attention_mask=attention_mask,
                cache_tuple=next_decoder_cache,
                image_token_range=current_image_range,
                vtr_ctx=vtr_ctx,
            )
            seq_length = new_seq_len
            current_image_range = new_image_range
            self._current_image_token_range = new_image_range

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(
            v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if v is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


# ---------------------------------------------------------------------------
# 5. Pruning logic
# ---------------------------------------------------------------------------

def _compute_and_apply_pruning(
    model,
    hidden_states: torch.Tensor,
    attention_weights: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_tuple: tuple,                     # old-style tuple of (key, value) pairs
    image_token_range: Tuple[int, int],
    vtr_ctx: Dict,
) -> Tuple[
    torch.Tensor,                           # pruned hidden_states
    torch.Tensor,                           # pruned position_ids
    Optional[torch.Tensor],                 # pruned attention_mask
    tuple,                                  # pruned cache tuple
    int,                                    # new seq_len
    Tuple[int, int],                        # new image_token_range
]:
    """
    Execute one round of visual-token pruning after a K-1 layer.

    This mirrors ``PrunableLlamaModel._compute_and_apply_pruning`` but operates
    on InternLM2's old-style tuple KV cache instead of DynamicCache.

    Args:
        model:              The InternLM2Model instance (for _vtr_config / _vtr_strategy).
        hidden_states:      [batch, seq_len, hidden_dim]
        attention_weights:  [batch, heads, seq_len, seq_len]
        position_ids:       [batch, seq_len]
        attention_mask:     [batch, 1, seq_len, seq_len] or None
        cache_tuple:        Tuple of (key, value) pairs, one per layer processed so far.
        image_token_range:  (img_start, img_end)
        vtr_ctx:            Strategy-specific context dict.

    Returns:
        (pruned_hidden_states, pruned_position_ids, pruned_attention_mask,
         pruned_cache_tuple, new_seq_len, new_image_token_range)
    """
    device = hidden_states.device
    img_start, img_end = image_token_range
    num_img_tokens = img_end - img_start
    seq_length = hidden_states.shape[1]

    # Nothing to prune
    if num_img_tokens <= 0:
        return (
            hidden_states, position_ids, attention_mask,
            cache_tuple, seq_length, image_token_range,
        )

    vtr_cfg: VTRConfig = model._vtr_config
    strategy: PruningStrategy = model._vtr_strategy

    # Step 1: compute scores using strategy
    scores = strategy.compute_scores(
        attention_weights, image_token_range, vtr_cfg, **vtr_ctx,
    )

    # Step 2: select tokens to keep (global top-k within image region)
    keep_img_indices = strategy.select_tokens(scores, num_img_tokens, vtr_cfg)
    num_kept_img_tokens = len(keep_img_indices)

    # If keeping everything, early return
    if num_kept_img_tokens >= num_img_tokens:
        return (
            hidden_states, position_ids, attention_mask,
            cache_tuple, seq_length, image_token_range,
        )

    # Step 3: build full keep_indices (pre-image + kept-image + post-image)
    keep_img_indices_abs = keep_img_indices + img_start
    keep_indices = torch.cat([
        torch.arange(img_start, device=device, dtype=torch.long),
        keep_img_indices_abs.view(-1).long(),
        torch.arange(img_end, seq_length, device=device, dtype=torch.long),
    ])
    keep_indices = keep_indices.sort().values
    new_seq_len = keep_indices.shape[0]

    # Step 4: new image_token_range
    new_img_end = img_start + num_kept_img_tokens
    new_image_range = (img_start, new_img_end)

    # Step 5: prune hidden_states
    hidden_states = hidden_states[:, keep_indices, :].contiguous()

    # Step 6: prune position_ids (preserve original position values)
    position_ids = position_ids.index_select(1, keep_indices)

    # Step 7: prune attention_mask (4D: [batch, 1, tgt, src])
    if attention_mask is not None:
        attention_mask = (
            attention_mask
            .index_select(2, keep_indices)
            .index_select(3, keep_indices)
            .contiguous()
        )

    # Step 8: prune KV cache -- old-style tuple of (key, value) pairs
    # Each entry is (key_states, value_states) with shape [batch, heads, seq, head_dim]
    if cache_tuple is not None:
        cache_list = list(cache_tuple)
        for i in range(len(cache_list)):
            if cache_list[i] is not None:
                k, v = cache_list[i]
                cache_list[i] = (
                    k.index_select(2, keep_indices).contiguous(),
                    v.index_select(2, keep_indices).contiguous(),
                )
        cache_tuple = tuple(cache_list)

    logger.debug(
        "Pruning: %d -> %d image tokens, seq_len: %d -> %d, "
        "image_range: %s -> %s",
        num_img_tokens, num_kept_img_tokens,
        seq_length, new_seq_len,
        image_token_range, new_image_range,
    )

    return (
        hidden_states, position_ids, attention_mask,
        cache_tuple, new_seq_len, new_image_range,
    )
