# Prunable LlamaModel
# LlamaModel implementation with multi-layer visual token pruning support.
from __future__ import annotations


import logging
from typing import List, Optional, Tuple, Union, Set, Dict, TYPE_CHECKING

import torch

from transformers import LlamaModel, LlamaConfig
from transformers.models.llama.modeling_llama import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast

import transformers
from packaging import version

if version.parse(transformers.__version__) < version.parse("4.37.0"):
    raise RuntimeError("DynamicCache requires transformers>=4.37.0, please upgrade your environment")

from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import LlamaAttention

if TYPE_CHECKING:
    from ..config import VTRConfig
    from ..strategy import PruningStrategy

from ..config import VTRConfig
from ..strategy import get_strategy, PruningStrategy
from .rope_utils import UnboundedLlamaRotaryEmbedding

logger = logging.getLogger(__name__)


class PrunableLlamaModel(LlamaModel):
    """
    LlamaModel with visual token pruning support.

    Supports pruning at multiple layers:
    - Single-layer: get attention at the specified layer and prune
    - Multi-layer: prune at multiple layers sequentially per config

    Pruning flow:
    1. Get attention weights at layer K-1
    2. Use strategy to compute scores and select tokens to keep
    3. Prune hidden_states, position_ids, attention_mask, KV cache
    4. Continue forward through remaining layers

    For multi-layer pruning, image_token_range is updated after each layer
    so that subsequent layers can correctly locate remaining image tokens.
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self._vtr_config: Optional[VTRConfig] = None
        self._vtr_strategy: Optional[PruningStrategy] = None
        # Track current image token range (dynamically updated during multi-layer pruning)
        self._current_image_token_range: Optional[Tuple[int, int]] = None

    def setup_vtr(self, config: VTRConfig) -> None:
        """
        Set up VTR configuration and replace RoPE modules.

        Args:
            config: VTR configuration object
        """
        self._vtr_config = config

        # CLSE: when selected with only a budget knob, fill in the single-stage video
        # prune schedule (prune_layer=[3]) + ref_layers=[2] for the spectral snapshot,
        # so the user does not have to spell them out. No-op for other strategies.
        from ..strategy.clse import apply_clse_defaults
        apply_clse_defaults(config, self.config)

        # [Critical] Always replace RoPE regardless of VTR enabled state
        # This is a compatibility fix to prevent sparse position_ids from going OOB
        self._replace_rope_with_unbounded()

        if config.enabled:
            # Prune at config.prune_layers using keep_tokens directly (LLaVA-style): no
            # layer-averaged token recompute, so an explicit keep_tokens is never rewritten.
            self._vtr_strategy = get_strategy(config.strategy)
            logger.debug(f"VTR enabled with strategy: {config.strategy}, "
                        f"prune_layers: {config.prune_layers}, "
                        f"keep_ratio: {config.keep_ratio}, "
                        f"keep_tokens: {config.keep_tokens}")

        else:
            self._vtr_strategy = None

    def _replace_rope_with_unbounded(self) -> None:
        """
        [Critical Fix] Replace all layers' RoPE with Unbounded version.
        Allows using original (larger) position_ids when physical sequence is shorter.

        Why this is needed:
        - After pruning, physical length shrinks (e.g., 190), but position_ids retain
          original values (max could be 621)
        - Standard RoPE slices cache by physical length, causing OOB for large position_ids
        - Unbounded RoPE returns the full cache, preventing OOB errors
        """
        for i, layer in enumerate(self.layers):
            old_rope = layer.self_attn.rotary_emb

            # Skip if already Unbounded
            if isinstance(old_rope, UnboundedLlamaRotaryEmbedding):
                continue

            # Create new instance inheriting old parameters
            new_rope = UnboundedLlamaRotaryEmbedding(
                dim=old_rope.dim,
                max_position_embeddings=old_rope.max_position_embeddings,
                base=old_rope.base,
                device=old_rope.inv_freq.device
            )

            # Replace module
            layer.self_attn.rotary_emb = new_rope

    def _get_prune_layer_set(self) -> Set[int]:
        """
        Get the set of layer indices after which pruning should be performed.

        Returns:
            Set of layer indices (K-1 layers) where attention is captured for pruning
        """
        if self._vtr_config is None:
            return set()
        # prune_layers are the pruning layers; we capture attention at K-1
        return {layer - 1 for layer in self._vtr_config.prune_layers}

    def forward(
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
        # VTR extra parameters (passed via args or injected state)
        image_token_range: Optional[Tuple[int, int]] = None,
        vtr_ctx: Optional[dict] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Forward with multi-layer pruning support.

        Supports pruning at multiple layers; image_token_range is updated after each prune.
        """

        # [Early Exit] If VTR is not configured or not enabled, use standard forward
        if self._vtr_config is None or not self._vtr_config.enabled:
            # Align attention_mask length with past_key_values if legacy cache is used
            try:
                if attention_mask is not None:
                    if input_ids is not None:
                        seq_length = input_ids.shape[1]
                    elif inputs_embeds is not None:
                        seq_length = inputs_embeds.shape[1]
                    else:
                        seq_length = None
                    past_len = None
                    if past_key_values is not None and seq_length is not None:
                        if isinstance(past_key_values, tuple):
                            if len(past_key_values) > 0 and past_key_values[0] is not None:
                                past_len = past_key_values[0][0].shape[2]
                            else:
                                past_len = 0
                        elif hasattr(past_key_values, "get_usable_length"):
                            past_len = past_key_values.get_usable_length(seq_length)
                    if seq_length is not None and past_len is not None:
                        target_len = seq_length + past_len
                        if attention_mask.shape[1] != target_len:
                            pad_len = target_len - attention_mask.shape[1]
                            if pad_len > 0:
                                pad = torch.ones(
                                    (attention_mask.shape[0], pad_len),
                                    dtype=attention_mask.dtype,
                                    device=attention_mask.device,
                                )
                                attention_mask = torch.cat([attention_mask, pad], dim=1)
                            else:
                                attention_mask = attention_mask[:, :target_len]
            except Exception:
                pass
            return super().forward(
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

        # [State Injection Support] If args not provided, try to get from injected state
        if image_token_range is None and hasattr(self, "_vtr_image_token_range"):
            image_token_range = self._vtr_image_token_range
        if vtr_ctx is None and hasattr(self, "_vtr_ctx"):
            vtr_ctx = self._vtr_ctx

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # [Key: following new HF logic] If first run and cache needed, create DynamicCache
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        past_key_values_length = 0
        use_legacy_cache = False
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            if past_key_values is not None:
                past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # =================================================================
        # [Fix 1] Dynamic RoPE cache pre-check expansion
        # Prevent position_ids (e.g., 5000) from exceeding default cache (4096)
        # =================================================================
        if position_ids is not None and position_ids.numel() > 0:
            current_max_pos = position_ids.max().item()
            # Get first layer's RoPE cache limit
            rope_module = self.layers[0].self_attn.rotary_emb
            # Compatible with different transformers versions
            current_limit = getattr(rope_module, "max_seq_len_cached", rope_module.cos_cached.shape[0])

            if current_max_pos >= current_limit:
                # Expansion target: current max ID + 1024 buffer
                new_limit = current_max_pos + 1024
                dtype = inputs_embeds.dtype if inputs_embeds is not None else torch.float16
                device = position_ids.device

                # Manually trigger expansion for all layers
                for layer in self.layers:
                    layer.self_attn.rotary_emb._set_cos_sin_cache(
                        seq_len=new_limit, device=device, dtype=dtype
                    )

        # Determine whether pruning is needed
        should_prune = (
            self._vtr_config is not None and
            self._vtr_config.enabled and
            self._vtr_strategy is not None and
            image_token_range is not None and
            past_key_values_length == 0  # Only prune during Prefill
        )

        # Get the set of layers to prune after (K-1 layers)
        prune_layer_set = self._get_prune_layer_set() if should_prune else set()
        vtr_ctx = vtr_ctx or {}

        # Track current image_token_range (dynamically updated during multi-layer pruning)
        current_image_range = image_token_range

        # When pruning is needed, capture K-1 layer attention
        need_attention = should_prune or output_attentions

        if self._use_flash_attention_2:
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # SDPA format mask
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4D causal mask (supports output_attentions)
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_idx = decoder_layer.self_attn.layer_idx

            # [CLSE] Snapshot reference image features at ref_layers (e.g. layer 2, before
            # this layer runs) into the shared vtr_ctx, for cross-layer spectral-evolution
            # scoring. Empty ref_layers (the default) makes this a no-op for other strategies.
            if should_prune and current_image_range is not None:
                _ref_layers = getattr(self._vtr_config, "ref_layers", None) or []
                if layer_idx in _ref_layers:
                    _rs, _re = current_image_range
                    vtr_ctx["z_ref"] = hidden_states[:, _rs:_re, :]

            # Check if attention is needed at this layer (K-1 layer)
            need_attention_for_pruning = should_prune and layer_idx in prune_layer_set
            layer_output_attentions = output_attentions or need_attention_for_pruning

            if need_attention_for_pruning:
                # This layer must fall back to Eager; feed it a 4D mask to prevent causal leakage
                layer_mask = _prepare_4d_causal_attention_mask(
                    None,
                    (batch_size, hidden_states.shape[1]),
                    hidden_states,
                    past_key_values_length
                )
                layer_output_attentions = True
            else:
                # Other layers: use SDPA acceleration (mask is usually None)
                layer_mask = attention_mask
                layer_output_attentions = output_attentions

            # Normal layer forward
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    layer_mask,
                    position_ids,
                    past_key_values,
                    layer_output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=layer_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=layer_output_attentions,
                    use_cache=use_cache,
                )

            # Update hidden_states and cache
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if layer_output_attentions else 1]
                past_key_values = next_decoder_cache

            # Prune immediately after K-1 layer
            if need_attention_for_pruning and current_image_range is not None:

                attention_weights = layer_outputs[1]
                if attention_weights is None:
                    if self._use_flash_attention_2:
                        raise ValueError(
                            "VTR Pruning error: You are using 'flash_attention_2' implementation which does not support outputting attention weights. "
                            "Please load model with attn_implementation='sdpa' or 'eager'."
                        )
                    else:
                        raise ValueError("Attention weights are None but pruning is requested.")

                # [CLSE] expose the current layer index (K-1) so stage-aware strategies can
                # determine which progressive pruning stage they are in.
                vtr_ctx["layer_idx"] = layer_idx
                # Execute full pruning (past_key_values already contains K-1 layer's kv)
                hidden_states, position_ids, attention_mask, past_key_values, new_seq_len, new_image_range = \
                    self._compute_and_apply_pruning(
                        hidden_states=hidden_states,
                        attention_weights=attention_weights,
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        image_token_range=current_image_range,
                        vtr_ctx=vtr_ctx,
                    )
                seq_length = new_seq_len

                # Update image_token_range for next pruning layer
                current_image_range = self._current_image_token_range = new_image_range

                # Flash Attention 2 mask fix
                if self._use_flash_attention_2:
                    attention_mask = None

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _compute_and_apply_pruning(
        self,
        hidden_states: torch.Tensor,
        attention_weights: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: DynamicCache,
        image_token_range: Tuple[int, int],
        vtr_ctx: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], DynamicCache, int, Tuple[int, int]]:
        """
        Execute the full pruning flow after layer K-1.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            attention_weights: [batch, heads, seq_len, seq_len]
            position_ids: [batch, seq_len]
            attention_mask: [batch, 1, seq_len, seq_len] or None
            past_key_values: DynamicCache
            image_token_range: (img_start, img_end) current image token range
            vtr_ctx: dict with strategy-specific info

        Returns:
            Tuple of:
            - pruned_hidden_states
            - pruned_position_ids
            - pruned_attention_mask
            - pruned_past_key_values
            - new_seq_len
            - new_image_token_range: updated image token range
        """
        device = hidden_states.device
        img_start, img_end = image_token_range
        num_img_tokens = img_end - img_start
        seq_length = hidden_states.shape[1]

        # If no image tokens, return as-is
        if num_img_tokens <= 0:
            return hidden_states, position_ids, attention_mask, past_key_values, seq_length, image_token_range

        # [CLSE] route the current-layer image features z_Lk in through vtr_ctx so the
        # spectral-evolution score can compare them with the z_ref snapshot. Other
        # strategies simply ignore this key.
        vtr_ctx["hidden_states"] = hidden_states

        # Step 1: Compute scores
        scores = self._vtr_strategy.compute_scores(
            attention_weights,
            image_token_range,
            self._vtr_config,
            **vtr_ctx
        )

        # Step 2: Select image tokens to keep
        keep_img_indices = self._vtr_strategy.select_tokens(
            scores, num_img_tokens, self._vtr_config
        )

        num_kept_img_tokens = len(keep_img_indices)

        # If keeping all tokens, return as-is
        if num_kept_img_tokens >= num_img_tokens:
            return hidden_states, position_ids, attention_mask, past_key_values, seq_length, image_token_range

        # Step 3: Build full keep_indices
        keep_img_indices_abs = keep_img_indices + img_start
        keep_indices = torch.cat([
            torch.arange(img_start, device=device, dtype=torch.long),  # tokens before image
            keep_img_indices_abs.view(-1).long(),  # kept image tokens
            torch.arange(img_end, seq_length, device=device, dtype=torch.long),  # tokens after image
        ])
        keep_indices = keep_indices.sort().values
        new_seq_len = keep_indices.shape[0]

        # Step 4: Compute new image_token_range
        # Image token new range: start stays the same, end = start + kept count
        new_img_end = img_start + num_kept_img_tokens
        new_image_range = (img_start, new_img_end)

        # Step 5: Prune hidden_states
        hidden_states = hidden_states[:, keep_indices, :].contiguous()

        # Step 6: Update position_ids (preserve original position values)
        position_ids = position_ids.index_select(1, keep_indices)

        # Step 7: Prune attention_mask
        if attention_mask is not None:
            attention_mask = attention_mask[:,:,:hidden_states.shape[1],:hidden_states.shape[1]]

        # Step 8: Prune KV cache (all existing layers)
        if past_key_values is not None and hasattr(past_key_values, 'key_cache'):
            for layer_i in range(len(past_key_values.key_cache)):
                if past_key_values.key_cache[layer_i] is not None:
                    # key/value: [batch, heads, seq_len, head_dim]
                    past_key_values.key_cache[layer_i] = \
                        past_key_values.key_cache[layer_i].index_select(2, keep_indices).contiguous()
                    past_key_values.value_cache[layer_i] = \
                        past_key_values.value_cache[layer_i].index_select(2, keep_indices).contiguous()
            # Update seen_tokens to logical length
            past_key_values._seen_tokens = position_ids.max().item() + 1
            if hasattr(past_key_values, 'seen_tokens'):
                past_key_values.seen_tokens = past_key_values._seen_tokens

        return hidden_states, position_ids, attention_mask, past_key_values, new_seq_len, new_image_range
