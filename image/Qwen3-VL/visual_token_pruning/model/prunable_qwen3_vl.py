"""Prunable Qwen3-VL Text Model for Visual Token Reduction.

This module provides PrunableQwen3VLTextModel, which extends Qwen3VLTextModel
to support visual token pruning at specified layers during the forward pass.

The pruning uses a Lightweight Look-ahead approach: at the score layer, the
decoder layer is decomposed to intercept Q and K after projection + RoPE.
Attention weights are computed via Q@K^T (in no_grad), then the normal
attention_interface (SDPA/flash) runs on the full sequence to produce correct
hidden_states. Pruning is applied AFTER the layer completes.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextModel,
    apply_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv,
)

from ..config import VTRConfig
from ..strategy.base import VTRStrategy
from .deepstack_handler import DeepStackSyncHandler

logger = logging.getLogger(__name__)


class PrunableQwen3VLTextModel(Qwen3VLTextModel):
    """Qwen3-VL Text Model with visual token pruning support.

    Extends Qwen3VLTextModel to perform visual token pruning at specified
    decoder layers. When VTR is disabled (vtr_config=None or enabled=False),
    the model behaves identically to the original Qwen3VLTextModel.

    The pruning process (Lightweight Look-ahead):
    1. At the score layer, the decoder_layer.forward() is decomposed:
       - Q, K, V are computed via projections + RoPE
       - Attention weights are computed via Q@K^T (in torch.no_grad)
       - The normal attention_interface (SDPA/flash) runs on FULL sequence
       - O_proj + Residual + MLP complete the layer
    2. After the layer, the VTR strategy uses the saved attention weights
       to compute importance scores and select tokens to keep.
    3. Low-scoring image tokens are removed from hidden_states, attention_mask,
       position_embeddings, KV cache, etc.
    4. Subsequent layers operate on the reduced sequence.

    This ensures that when keep_ratio=1.0 (no pruning), the output is
    bit-for-bit identical to the non-VTR path.
    """

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # DeepStack args
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[List[torch.Tensor]] = None,
        # VTR args
        vtr_config: Optional[VTRConfig] = None,
        vtr_strategy: Optional[VTRStrategy] = None,
        vtr_context: Optional[Dict[str, Any]] = None,
        image_token_range: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        """Forward pass with optional visual token pruning.

        When VTR is not active, delegates to the parent Qwen3VLTextModel.forward().
        When active, runs a custom layer loop with Look-ahead at the score layer.
        """
        # Determine if VTR is active
        vtr_active = (
            vtr_config is not None
            and vtr_config.enabled
            and vtr_strategy is not None
            and image_token_range is not None
        )

        # If VTR is not active, delegate to the parent forward
        if not vtr_active:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                cache_position=cache_position,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                **kwargs,
            )

        # --- VTR-enabled forward pass ---
        if vtr_context is None:
            vtr_context = {}

        # Resolve pruning layers (score layer = prune_layer - 1)
        prune_layers = vtr_config.get_prune_layers()

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # Resolve use_cache from config if not explicitly passed
        # (matches @check_model_inputs decorator behavior on the parent class)
        if use_cache is None:
            use_cache = getattr(self.config, "use_cache", None)

        # Initialize cache if needed
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        # Handle position_ids (3D MRoPE)
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        from transformers.masking_utils import create_causal_mask

        attention_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds

        # Compute position embeddings (shared across decoder layers)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Track current image_token_range (updated after each pruning)
        current_image_range = image_token_range

        # Call strategy.prepare() for one-time setup (e.g., SparseVLM text token identification)
        vtr_strategy.prepare(hidden_states, vtr_config, vtr_context)

        # Decoder layer loop
        for layer_idx, decoder_layer in enumerate(self.layers):
            # [CLSE] Snapshot reference image features z_ref at config.ref_layers (before this
            # layer runs), honoring ref_layers like the LLaVA/Video backbones. Default [0]
            # => snapshot at the input embeddings, identical to the previous behaviour.
            if current_image_range is not None:
                _ref_layers = getattr(vtr_config, "ref_layers", None) or []
                if layer_idx in _ref_layers:
                    _rs, _re = current_image_range
                    if _re > _rs:
                        vtr_context["z_ref"] = hidden_states[:, _rs:_re, :]

            # Check if this is the score layer (layer before a prune_layer)
            is_score_layer = (layer_idx + 1) in prune_layers

            if is_score_layer:
                # === Lightweight Look-ahead: decompose decoder_layer.forward() ===
                hidden_states, attn_weights = self._score_layer_forward(
                    decoder_layer,
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

                # Apply pruning using the captured attention weights
                (
                    hidden_states,
                    position_ids,
                    text_position_ids,
                    attention_mask,
                    past_key_values,
                    position_embeddings,
                    cache_position,
                    visual_pos_masks,
                    deepstack_visual_embeds,
                    current_image_range,
                    pruned_token_hidden_states,
                ) = self._apply_pruning(
                    hidden_states=hidden_states,
                    position_ids=position_ids,
                    text_position_ids=text_position_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_embeddings=position_embeddings,
                    cache_position=cache_position,
                    attention_weights=attn_weights,
                    image_token_range=current_image_range,
                    vtr_config=vtr_config,
                    vtr_strategy=vtr_strategy,
                    vtr_context=vtr_context,
                    layer_idx=layer_idx,
                    visual_pos_masks=visual_pos_masks,
                    deepstack_visual_embeds=deepstack_visual_embeds,
                )

                # Build keep_indices for post_prune_hook
                # (absolute positions of kept image tokens in the pruned sequence)
                new_img_start, new_img_end = current_image_range
                keep_img_indices_full = torch.arange(
                    new_img_start, new_img_end, device=hidden_states.device
                )

                # Post-prune hook (token merge for SparseVLM)
                (
                    hidden_states,
                    position_ids,
                    text_position_ids,
                    attention_mask,
                    past_key_values,
                    position_embeddings,
                    cache_position,
                    current_image_range,
                    visual_pos_masks,
                    deepstack_visual_embeds,
                ) = self._post_prune_hook(
                    hidden_states=hidden_states,
                    pruned_token_hidden_states=pruned_token_hidden_states,
                    keep_indices=keep_img_indices_full,
                    position_ids=position_ids,
                    text_position_ids=text_position_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_embeddings=position_embeddings,
                    cache_position=cache_position,
                    image_token_range=current_image_range,
                    vtr_config=vtr_config,
                    vtr_strategy=vtr_strategy,
                    vtr_context=vtr_context,
                    layer_idx=layer_idx,
                    visual_pos_masks=visual_pos_masks,
                    deepstack_visual_embeds=deepstack_visual_embeds,
                )
            else:
                # Standard layer forward
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=text_position_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            # DeepStack injection (after layer execution and possible pruning)
            if (
                deepstack_visual_embeds is not None
                and layer_idx < len(deepstack_visual_embeds)
                and deepstack_visual_embeds[layer_idx] is not None
            ):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def _score_layer_forward(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute a decoder layer with inline Look-ahead for attention weights.

        Decomposes decoder_layer.forward() to intercept Q, K after projection
        and RoPE. Computes attention weights via Q@K^T in no_grad (Look-ahead),
        then runs the normal attention_interface (SDPA/flash) on the FULL
        sequence. The layer completes normally (O_proj + residual + MLP).

        For keep_ratio=1.0: the attention_interface receives identical Q, K, V,
        mask as the normal layer, producing bit-for-bit identical hidden_states.

        Args:
            layer: The decoder layer to execute.
            hidden_states: Input hidden states [batch, seq, hidden].
            attention_mask: 4D causal mask [batch, 1, seq, kv_seq].
            past_key_values: KV cache.
            cache_position: Cache position indices.
            position_embeddings: Precomputed (cos, sin) from rotary embedding.
            **kwargs: Additional keyword arguments (FlashAttentionKwargs, etc.).

        Returns:
            Tuple of (hidden_states, attention_weights) where:
            - hidden_states: output of the full layer [batch, seq, hidden]
            - attention_weights: [batch, num_attention_heads, seq, kv_seq]
        """
        attn = layer.self_attn

        # --- Replicate decoder_layer.forward() with Look-ahead ---

        # 1. Input LayerNorm
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)

        # 2. Q, K, V projections (replicate self_attn.forward() internals)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)

        query_states = attn.q_norm(
            attn.q_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        key_states = attn.k_norm(
            attn.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # 3. RoPE
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # 4. KV Cache update
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, attn.layer_idx, cache_kwargs
            )

        # 5. Look-ahead: compute attention weights from Q @ K^T (no_grad)
        with torch.no_grad():
            # GQA expansion needed for manual matmul (Q has more heads than K)
            key_states_expanded = repeat_kv(key_states, attn.num_key_value_groups)

            attn_scores = torch.matmul(
                query_states, key_states_expanded.transpose(2, 3)
            ) * attn.scaling

            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, :key_states_expanded.shape[-2]]
                attn_scores = attn_scores + causal_mask

            attn_weights = F.softmax(
                attn_scores, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)

        # 6. Run the actual attention via the same interface as normal path
        # (K, V are passed pre-GQA; the interface handles expansion internally)
        attention_interface: Callable = eager_attention_forward
        if attn.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[attn.config._attn_implementation]

        attn_output, _ = attention_interface(
            attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not attn.training else attn.attention_dropout,
            scaling=attn.scaling,
            **kwargs,
        )

        # 7. Output projection
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn.o_proj(attn_output)

        # 8. Residual connection
        hidden_states = residual + attn_output

        # 9. MLP
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, attn_weights

    def _apply_pruning(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        text_position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[DynamicCache],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        cache_position: torch.LongTensor,
        attention_weights: torch.Tensor,
        image_token_range: Tuple[int, int],
        vtr_config: VTRConfig,
        vtr_strategy: VTRStrategy,
        vtr_context: Dict[str, Any],
        layer_idx: int,
        visual_pos_masks: Optional[torch.Tensor],
        deepstack_visual_embeds: Optional[List[torch.Tensor]],
    ) -> Tuple[
        torch.Tensor,  # hidden_states
        torch.Tensor,  # position_ids
        torch.Tensor,  # text_position_ids
        Optional[torch.Tensor],  # attention_mask
        Optional[DynamicCache],  # past_key_values
        Tuple[torch.Tensor, torch.Tensor],  # position_embeddings
        torch.LongTensor,  # cache_position
        Optional[torch.Tensor],  # visual_pos_masks
        Optional[List[torch.Tensor]],  # deepstack_visual_embeds
        Tuple[int, int],  # new image_token_range
        torch.Tensor,  # pruned_token_hidden_states
    ]:
        """Apply visual token pruning after the score layer completes.

        Uses the attention weights from the Look-ahead to compute importance
        scores, determines which tokens to keep, and prunes all relevant
        tensors (hidden_states, mask, position_embeddings, cache, etc.).

        When keep_ratio=1.0, all tokens are kept and this is a no-op.
        """
        img_start, img_end = image_token_range
        num_img_tokens = img_end - img_start
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device

        if num_img_tokens <= 0:
            logger.warning("No image tokens to prune. Skipping pruning.")
            return (
                hidden_states, position_ids, text_position_ids,
                attention_mask, past_key_values, position_embeddings,
                cache_position, visual_pos_masks, deepstack_visual_embeds,
                image_token_range,
                torch.empty(batch_size, 0, hidden_size, device=device),
            )

        # 1. Compute importance scores
        # Determine which prune step this is (0, 1, 2, ...)
        prune_layers = vtr_config.get_prune_layers()
        prune_step = prune_layers.index(layer_idx + 1) if (layer_idx + 1) in prune_layers else 0

        # Filter keys that conflict with positional args of compute_scores
        score_context = {k: v for k, v in vtr_context.items() if k != "image_token_range"}
        # [CLSE] route the current layer features (z_Lk) to feature-based strategies.
        # Kept local to score_context so it does not leak into the post_prune hook
        # (whose first positional arg is also named hidden_states).
        score_context["hidden_states"] = hidden_states
        scores = vtr_strategy.compute_scores(
            attention_weights, image_token_range, vtr_config, layer_idx=prune_step, **score_context
        )

        # 2. Select which image tokens to keep (indices relative to image range)
        keep_img_indices = vtr_strategy.select_tokens(scores, num_img_tokens, vtr_config, layer_idx=prune_step)
        num_kept = len(keep_img_indices)

        if vtr_config.debug:
            logger.debug(
                f"Layer {layer_idx}: pruning {num_img_tokens - num_kept}/{num_img_tokens} "
                f"image tokens (keeping {num_kept})"
            )

        # Edge case: keeping all tokens (keep_ratio=1.0)
        if num_kept >= num_img_tokens:
            return (
                hidden_states, position_ids, text_position_ids,
                attention_mask, past_key_values, position_embeddings,
                cache_position, visual_pos_masks, deepstack_visual_embeds,
                image_token_range,
                torch.empty(batch_size, 0, hidden_size, device=device),
            )

        # 3. Build full keep_indices (relative to entire sequence)
        sys_indices = torch.arange(img_start, device=device)
        kept_img_absolute = keep_img_indices.to(device) + img_start
        text_indices = torch.arange(img_end, seq_len, device=device)
        keep_indices = torch.cat([sys_indices, kept_img_absolute, text_indices])

        new_seq_len = len(keep_indices)

        # Save pruned image token hidden_states for potential merge
        pruned_img_mask = torch.ones(num_img_tokens, dtype=torch.bool, device=device)
        pruned_img_mask[keep_img_indices.to(device)] = False
        pruned_img_absolute = torch.arange(img_start, img_end, device=device)[pruned_img_mask]
        pruned_token_hidden_states = hidden_states.index_select(1, pruned_img_absolute)

        # Save pruned token scores for merge candidate selection
        vtr_context["pruned_token_scores"] = scores[pruned_img_mask]

        # 4. Prune hidden_states
        hidden_states = hidden_states.index_select(1, keep_indices)

        # 5. Prune position_ids (preserve original semantic positions)
        position_ids = position_ids.index_select(-1, keep_indices)
        text_position_ids = text_position_ids.index_select(-1, keep_indices)

        # 6. Prune attention_mask (both query and key dimensions)
        if attention_mask is not None:
            attention_mask = attention_mask.index_select(2, keep_indices)
            attention_mask = attention_mask.index_select(3, keep_indices)

        # 7. Prune KV cache for all layers computed so far
        if past_key_values is not None:
            for i in range(layer_idx + 1):
                if i < len(past_key_values.layers):
                    past_key_values.layers[i].keys = (
                        past_key_values.layers[i].keys.index_select(2, keep_indices)
                    )
                    past_key_values.layers[i].values = (
                        past_key_values.layers[i].values.index_select(2, keep_indices)
                    )

        # 8. Prune position_embeddings (cos, sin)
        cos, sin = position_embeddings
        cos = cos.index_select(1, keep_indices)
        sin = sin.index_select(1, keep_indices)
        position_embeddings = (cos, sin)

        # 9. Update cache_position
        cache_position = torch.arange(new_seq_len, device=device)

        # 10. Sync DeepStack: prune visual_pos_masks and deepstack embeds
        deepstack_handler = DeepStackSyncHandler(
            num_deepstack_layers=(
                len(deepstack_visual_embeds) if deepstack_visual_embeds else 0
            )
        )
        deepstack_visual_embeds, visual_pos_masks = deepstack_handler.sync_after_pruning(
            deepstack_visual_embeds=deepstack_visual_embeds,
            visual_pos_masks=visual_pos_masks,
            keep_img_indices=keep_img_indices,
            keep_indices=keep_indices,
            current_layer_idx=layer_idx,
        )

        # 11. Update image_token_range
        new_image_token_range = (img_start, img_start + num_kept)

        # 12. Track total pruned tokens for generation position correction
        num_pruned = num_img_tokens - num_kept
        vtr_context["total_pruned_tokens"] = (
            vtr_context.get("total_pruned_tokens", 0) + num_pruned
        )

        return (
            hidden_states,
            position_ids,
            text_position_ids,
            attention_mask,
            past_key_values,
            position_embeddings,
            cache_position,
            visual_pos_masks,
            deepstack_visual_embeds,
            new_image_token_range,
            pruned_token_hidden_states,
        )

    def _post_prune_hook(
        self,
        hidden_states: torch.Tensor,
        pruned_token_hidden_states: torch.Tensor,
        keep_indices: torch.Tensor,
        position_ids: torch.Tensor,
        text_position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[DynamicCache],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        cache_position: torch.LongTensor,
        image_token_range: Tuple[int, int],
        vtr_config: VTRConfig,
        vtr_strategy: VTRStrategy,
        vtr_context: Dict[str, Any],
        layer_idx: int,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor,  # hidden_states
        torch.Tensor,  # position_ids
        torch.Tensor,  # text_position_ids
        Optional[torch.Tensor],  # attention_mask
        Optional[DynamicCache],  # past_key_values
        Tuple[torch.Tensor, torch.Tensor],  # position_embeddings
        torch.LongTensor,  # cache_position
        Tuple[int, int],  # image_token_range
        Optional[torch.Tensor],  # visual_pos_masks
        Optional[List[torch.Tensor]],  # deepstack_visual_embeds
    ]:
        """Call strategy.post_prune and update all state tensors if sequence changed.

        After pruning, the strategy may insert merged tokens (e.g., SparseVLM
        density-peak clustering). This method handles updating all tensors
        (position_ids, attention_mask, KV cache, etc.) when the sequence length
        changes due to token merging.

        For strategies without token merge (FastV, PriorTR-2F), post_prune is a
        no-op and this method returns immediately with no changes.
        """
        seq_before = hidden_states.shape[1]

        # Determine prune step index
        prune_layers = vtr_config.get_prune_layers()
        prune_step = prune_layers.index(layer_idx + 1) if (layer_idx + 1) in prune_layers else 0

        # Filter keys that conflict with positional args of post_prune
        prune_context = {k: v for k, v in vtr_context.items() if k != "image_token_range"}
        hidden_states = vtr_strategy.post_prune(
            hidden_states, pruned_token_hidden_states, keep_indices,
            image_token_range, vtr_config, prune_step, **prune_context,
        )

        seq_after = hidden_states.shape[1]
        n_inserted = seq_after - seq_before

        if n_inserted <= 0:
            return (
                hidden_states, position_ids, text_position_ids,
                attention_mask, past_key_values, position_embeddings,
                cache_position, image_token_range,
                visual_pos_masks, deepstack_visual_embeds,
            )

        # Merged tokens were inserted at img_end (before text tokens)
        img_start, img_end = image_token_range
        insert_pos = img_end
        device = hidden_states.device

        # Update image_token_range
        new_image_range = (img_start, img_end + n_inserted)

        # --- Extend position_ids ---
        pre = position_ids[..., :insert_pos]
        post = position_ids[..., insert_pos:]
        last_vis_pos = position_ids[..., insert_pos - 1:insert_pos]
        merge_pos = last_vis_pos.expand(*last_vis_pos.shape[:-1], n_inserted) + 1
        position_ids = torch.cat([pre, merge_pos, post], dim=-1)

        # Same for text_position_ids
        pre_t = text_position_ids[..., :insert_pos]
        post_t = text_position_ids[..., insert_pos:]
        last_t = text_position_ids[..., insert_pos - 1:insert_pos]
        merge_t = last_t.expand(*last_t.shape[:-1], n_inserted) + 1
        text_position_ids = torch.cat([pre_t, merge_t, post_t], dim=-1)

        # --- Extend position_embeddings ---
        cos, sin = position_embeddings
        pre_cos = cos[:, :insert_pos]
        post_cos = cos[:, insert_pos:]
        merge_cos = cos[:, insert_pos - 1:insert_pos].expand(-1, n_inserted, -1)
        cos = torch.cat([pre_cos, merge_cos, post_cos], dim=1)

        pre_sin = sin[:, :insert_pos]
        post_sin = sin[:, insert_pos:]
        merge_sin = sin[:, insert_pos - 1:insert_pos].expand(-1, n_inserted, -1)
        sin = torch.cat([pre_sin, merge_sin, post_sin], dim=1)
        position_embeddings = (cos, sin)

        # --- Extend attention_mask ---
        if attention_mask is not None:
            from transformers.masking_utils import create_causal_mask
            attention_mask = create_causal_mask(
                config=self.config,
                input_embeds=hidden_states,
                attention_mask=None,
                cache_position=torch.arange(seq_after, device=device),
                past_key_values=None,
                position_ids=text_position_ids,
            )

        # --- Extend KV cache ---
        if past_key_values is not None:
            for i in range(layer_idx + 1):
                if i < len(past_key_values.layers):
                    k = past_key_values.layers[i].keys
                    v = past_key_values.layers[i].values
                    pre_k = k[:, :, :insert_pos, :]
                    post_k = k[:, :, insert_pos:, :]
                    merge_k = torch.zeros(
                        k.shape[0], k.shape[1], n_inserted, k.shape[3],
                        dtype=k.dtype, device=k.device
                    )
                    past_key_values.layers[i].keys = torch.cat([pre_k, merge_k, post_k], dim=2)

                    pre_v = v[:, :, :insert_pos, :]
                    post_v = v[:, :, insert_pos:, :]
                    merge_v = torch.zeros_like(merge_k)
                    past_key_values.layers[i].values = torch.cat([pre_v, merge_v, post_v], dim=2)

        # --- Update cache_position ---
        cache_position = torch.arange(seq_after, device=device)

        # --- Extend visual_pos_masks ---
        # Merged tokens are NOT original visual positions for DeepStack,
        # so insert False values at the merge insertion point.
        if visual_pos_masks is not None:
            pre_mask = visual_pos_masks[:, :insert_pos]
            post_mask = visual_pos_masks[:, insert_pos:]
            merge_mask = torch.zeros(
                visual_pos_masks.shape[0], n_inserted,
                dtype=torch.bool, device=device,
            )
            visual_pos_masks = torch.cat(
                [pre_mask, merge_mask, post_mask], dim=1
            )

        # Track merged tokens
        vtr_context["total_merged_tokens"] = (
            vtr_context.get("total_merged_tokens", 0) + n_inserted
        )

        return (
            hidden_states, position_ids, text_position_ids,
            attention_mask, past_key_values, position_embeddings,
            cache_position, new_image_range,
            visual_pos_masks, deepstack_visual_embeds,
        )
