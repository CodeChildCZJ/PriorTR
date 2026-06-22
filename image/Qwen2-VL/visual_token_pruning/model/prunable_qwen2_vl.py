"""Prunable Qwen2-VL text model with strategy-pattern visual token reduction.

This mirrors the Qwen3-VL ``prunable_qwen3_vl`` design but targets Qwen2-VL
(transformers 4.57.x). The pruning *mechanics* (visual-range detection via the
visual position mask, reference-feature snapshot at L_list=[0], progressive
multi-stage pruning at K_list, and the last-text-token attention capture) are kept
faithful to the original CLSE Qwen2-VL implementation so that the CLSE strategy
reproduces the original numbers exactly. Only the *scoring* and the *keep count*
are delegated to the pluggable strategy / config, so PriorTR, FastV and CLSE all
share one model.
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLTextModel

from ..config import VTRConfig
from ..strategy.base import VTRStrategy


class PrunableQwen2VLTextModel(Qwen2VLTextModel):
    """Qwen2-VL text model that prunes visual tokens via a pluggable strategy.

    Use :meth:`setup_vtr` to attach a config + strategy. With VTR disabled the
    forward is the stock Qwen2-VL forward (delegated to the parent class).
    ``visual_pos_masks`` and ``image_grid_thw`` are set on this module per-forward
    by the (patched) Qwen2VLForConditionalGeneration wrapper.
    """

    def setup_vtr(self, config: VTRConfig, strategy: VTRStrategy) -> None:
        self.vtr_config = config
        self.vtr_strategy = strategy
        # Disable any inherited env-var pruning from the patched base class; this
        # framework drives pruning through vtr_config / strategy instead.
        self.prune = False

    # ------------------------------------------------------------------ #
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        vtr_config: Optional[VTRConfig] = getattr(self, "vtr_config", None)

        # --- Fall back to the stock forward when VTR is off ---
        if vtr_config is None or not vtr_config.enabled:
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
                cache_position=cache_position,
                **kwargs,
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # 3D MRoPE position ids: [3, B, S]
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
            text_position_ids = None
        else:
            if position_ids.ndim == 3 and position_ids.shape[0] == 4:
                text_position_ids = position_ids[0]
                position_ids = position_ids[1:]
            elif position_ids.ndim == 3 and position_ids.shape[0] == 3:
                text_position_ids = None
            elif position_ids.ndim == 2:
                position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
                text_position_ids = None
            else:
                text_position_ids = None

        # 4D causal mask mapping
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            from transformers.models.qwen2_vl.modeling_qwen2_vl import (
                create_causal_mask,
                create_sliding_window_causal_mask,
            )
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
            }
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
            if getattr(self, "has_sliding_layers", False) or "sliding_attention" in getattr(self.config, "layer_types", []):
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        # Locate the visual token range via the visual position mask.
        has_visual = False
        img_start, img_end = 0, 0
        visual_mask = kwargs.get("visual_pos_masks", getattr(self, "visual_pos_masks", None))
        if visual_mask is not None:
            current_mask = visual_mask[0] if visual_mask.dim() == 2 else visual_mask
            vis_indices = torch.nonzero(current_mask).squeeze()
            if vis_indices.numel() > 0:
                img_start, img_end = vis_indices[0].item(), vis_indices[-1].item() + 1
                has_visual = True

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        should_prune = has_visual and inputs_embeds.shape[1] > 1
        prune_layers = vtr_config.get_prune_layers()            # K_list (1-indexed-ish, see below)
        # K_list semantics (match the original): prune *at* layer index k for k in prune_layers,
        # using attention captured one layer earlier (k-1). ref features at L_list (default [0]).
        ref_layers = getattr(vtr_config, "ref_layers", None) or [0]

        original_visual_len = img_end - img_start if has_visual else 0
        vtr_context: Dict[str, Any] = {"image_token_range": (img_start, img_end)}
        # grid (h, w) after the 2x2 spatial merge, for spectral scoring.
        if should_prune and self.image_grid_thw is not None:
            try:
                _h = int(self.image_grid_thw[0][1].item()) // 2
                _w = int(self.image_grid_thw[0][2].item()) // 2
                vtr_context["grid_hw"] = (_h, _w)
            except Exception:
                pass

        # one-time setup (snapshots reference features z_L from the input embeddings)
        if should_prune:
            self.vtr_strategy.prepare(hidden_states, vtr_config, vtr_context)

        last_attention = None  # full attention captured at layer k-1

        for layer_idx, decoder_layer in enumerate(self.layers):
            # --- Prune at each layer in prune_layers (K_list) ---
            if should_prune and layer_idx in prune_layers and last_attention is not None and (img_end - img_start) > 0:
                prune_step = prune_layers.index(layer_idx)
                seq_len_before = hidden_states.shape[1]

                vtr_context["hidden_states"] = hidden_states  # z_Lk (current features)
                score_context = {k: v for k, v in vtr_context.items() if k != "image_token_range"}
                scores = self.vtr_strategy.compute_scores(
                    last_attention, (img_start, img_end), vtr_config,
                    layer_idx=prune_step, **score_context,
                )
                # Keep count: ratio is applied to the ORIGINAL visual length (CLSE semantics),
                # so we pass original_visual_len as the token count to select_tokens.
                keep_rel = self.vtr_strategy.select_tokens(
                    scores, original_visual_len, vtr_config, layer_idx=prune_step,
                )
                num_kept = len(keep_rel)
                current_len = img_end - img_start

                if 0 < num_kept < current_len:
                    keep_rel = keep_rel.sort().values
                    keep_indexs = torch.cat((
                        torch.arange(img_start, device=hidden_states.device),
                        keep_rel.to(hidden_states.device) + img_start,
                        torch.arange(img_end, hidden_states.shape[1], device=hidden_states.device),
                    )).long()

                    hidden_states = hidden_states[:, keep_indexs, :]
                    position_ids = position_ids[..., keep_indexs]
                    if text_position_ids is not None:
                        text_position_ids = text_position_ids[..., keep_indexs]
                    cache_position = cache_position[keep_indexs]

                    if causal_mask_mapping is not None:
                        for mk, mv in causal_mask_mapping.items():
                            if mv is not None and isinstance(mv, torch.Tensor) and mv.shape[-1] == seq_len_before:
                                causal_mask_mapping[mk] = mv.index_select(-2, keep_indexs).index_select(-1, keep_indexs)

                    position_embeddings = self.rotary_emb(hidden_states, position_ids)
                    img_end = img_start + num_kept

            mask_to_use = causal_mask_mapping[decoder_layer.attention_type] if causal_mask_mapping is not None else None

            # Capture full attention one layer before the next prune layer (layer k-1).
            next_layer_idx = layer_idx + 1
            if should_prune and next_layer_idx in prune_layers and hidden_states.shape[1] > 1:
                last_attention = self._capture_full_attention(
                    decoder_layer, hidden_states, mask_to_use, position_embeddings
                )

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=mask_to_use,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # ------------------------------------------------------------------ #
    def _capture_full_attention(self, layer, hidden_states, attention_mask, position_embeddings):
        """Compute the full softmax attention weights at one layer (no value/output proj).

        Returns [B, heads, q_len, k_len]. The last query row is identical to the
        original CLSE last-text-token capture, so attention-only / last-token
        strategies reproduce the original; the full matrix additionally lets
        PriorTR read the <|vision_end|> prior row.
        """
        from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_multimodal_rotary_pos_emb

        with torch.no_grad():
            attn = layer.self_attn
            hidden_norm = layer.input_layernorm(hidden_states)
            bsz, q_len, _ = hidden_norm.size()

            query_states = attn.q_proj(hidden_norm).view(bsz, q_len, -1, attn.head_dim).transpose(1, 2)
            key_states = attn.k_proj(hidden_norm).view(bsz, q_len, -1, attn.head_dim).transpose(1, 2)

            cos, sin = position_embeddings
            mrope_section = attn.rope_scaling["mrope_section"]
            query_states, key_states = apply_multimodal_rotary_pos_emb(
                query_states, key_states, cos, sin, mrope_section
            )

            if attn.num_key_value_groups > 1:
                key_states = key_states.repeat_interleave(attn.num_key_value_groups, dim=1)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * attn.scaling
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            return attn_weights.detach()
