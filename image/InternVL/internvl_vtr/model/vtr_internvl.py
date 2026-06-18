# VTR wrapper for InternVLChatModel
# Monkeypatches InternVLChatModel to support Visual Token Reduction (VTR).
#
# Architecture:
#   InternVLChatModel (outer, handles ViT + embedding replacement)
#     └── language_model: InternLM2ForCausalLM (CausalLM wrapper)
#           └── model: InternLM2Model (backbone, where pruning happens)
#
# This file provides:
#   - setup_vtr_model(model, config, tokenizer)  — main entry point
#   - _compute_image_token_range()               — scan input_ids for image token positions
#   - _vtr_generate()                            — wraps original generate with state injection
#   - _vtr_prepare_inputs_for_generation()       — VTR-aware decode fix
#
# Key token IDs (InternVL2.5-8B):
#   <IMG_CONTEXT> = 92546   (visual tokens)
#   </img>        = 92545   (Q anchor — sees all visual tokens under causal mask)

from __future__ import annotations

import logging
import types
from typing import Optional, Tuple, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

from ..config import VTRConfig
from .prunable_internlm2 import setup_vtr

logger = logging.getLogger(__name__)

# Token IDs for InternVL2.5-8B (trust_remote_code tokenizer)
_IMG_CONTEXT_TOKEN_ID = 92546   # <IMG_CONTEXT>
_IMG_END_TOKEN_ID = 92545       # </img>


# ---------------------------------------------------------------------------
# 1. Compute image token range from input_ids
# ---------------------------------------------------------------------------

def _compute_image_token_range(
    model,
    input_ids: torch.LongTensor,
) -> Optional[Tuple[int, int]]:
    """
    Scan ``input_ids`` for the image token region.

    The InternVL token layout is::

        ... <img> <IMG_CONTEXT> <IMG_CONTEXT> ... <IMG_CONTEXT> </img> ...

    We return ``(img_start, img_end)`` where:
      - ``img_start`` = index of the first ``<IMG_CONTEXT>`` token
      - ``img_end``   = index of the ``</img>`` token

    The visual tokens occupy positions ``[img_start, img_end)`` (exclusive end).
    ``</img>`` at position ``img_end`` is the Q-anchor that sees all visual
    tokens under the causal mask — it is NOT counted as an image token.

    Args:
        model:     The InternVLChatModel instance (used to resolve
                   ``img_context_token_id`` if set dynamically by ``chat()``).
        input_ids: [batch, seq_len] token IDs.

    Returns:
        ``(img_start, img_end)`` or ``None`` if no image tokens found.
    """
    if input_ids is None:
        return None

    # Flatten to 1D for scanning (batch_size must be 1 for VTR)
    ids = input_ids.view(-1)

    # Resolve the IMG_CONTEXT token ID.
    # The model may set img_context_token_id dynamically in chat()/batch_chat().
    img_ctx_id = getattr(model, 'img_context_token_id', None) or _IMG_CONTEXT_TOKEN_ID
    img_end_id = _IMG_END_TOKEN_ID

    # Find all IMG_CONTEXT positions
    img_ctx_mask = (ids == img_ctx_id)
    if not img_ctx_mask.any():
        return None

    img_ctx_positions = img_ctx_mask.nonzero(as_tuple=False).view(-1)
    img_start = img_ctx_positions[0].item()

    # Find </img> position (should be right after the last IMG_CONTEXT)
    img_end_mask = (ids == img_end_id)
    if not img_end_mask.any():
        # Fallback: use position after last IMG_CONTEXT
        img_end = img_ctx_positions[-1].item() + 1
        logger.warning(
            "Could not find </img> token (ID=%d) in input_ids. "
            "Falling back to img_end=%d (after last IMG_CONTEXT).",
            img_end_id, img_end,
        )
    else:
        img_end_positions = img_end_mask.nonzero(as_tuple=False).view(-1)
        # Use the first </img> that comes after img_start
        valid = img_end_positions[img_end_positions > img_start]
        if valid.numel() == 0:
            img_end = img_ctx_positions[-1].item() + 1
            logger.warning(
                "No </img> token found after img_start=%d. "
                "Falling back to img_end=%d.",
                img_start, img_end,
            )
        else:
            img_end = valid[0].item()

    num_img_tokens = img_end - img_start
    logger.debug(
        "Image token range: [%d, %d), num_img_tokens=%d",
        img_start, img_end, num_img_tokens,
    )
    return (img_start, img_end)


# ---------------------------------------------------------------------------
# 2. VTR-aware prepare_inputs_for_generation
# ---------------------------------------------------------------------------

def _vtr_prepare_inputs_for_generation(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    **kwargs,
):
    """
    VTR-aware replacement for ``InternLM2ForCausalLM.prepare_inputs_for_generation``.

    Problem:
        After pruning, the physical KV cache length (e.g. 200) is much smaller
        than the logical sequence length (e.g. 1809).  HF's generate loop
        accumulates input_ids to ~1810 tokens, so the standard logic sees
        ``input_ids.shape[1] > past_length(=200)`` and removes only 200 tokens,
        keeping 1610 — which is wrong.

    Fix (decode phase only):
        - Always use ``input_ids[:, -1:]`` (just the last token)
        - Position: ``attention_mask.long().cumsum(-1)[:, -1:] - 1`` (logical pos)
        - Attention mask: short ``ones(batch, physical_cache_len + 1)``

    For baseline (VTR disabled) or prefill (no past_key_values):
        Delegate to the original ``prepare_inputs_for_generation``.
    """
    # ---- Check if VTR is active on the inner model ----
    inner = self.model  # InternLM2Model
    vtr_cfg = getattr(inner, '_vtr_config', None)
    vtr_active = (
        vtr_cfg is not None
        and vtr_cfg.enabled
        and hasattr(inner, '_vtr_image_token_range')
    )

    # ---- VTR disabled or prefill: delegate to original ----
    if not vtr_active or past_key_values is None:
        return self._original_prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    # ---- VTR decode phase ----
    # Physical cache length from old-style tuple KV cache
    if isinstance(past_key_values, tuple):
        cache_length = past_key_values[0][0].shape[2]
    elif hasattr(past_key_values, 'get_seq_length'):
        cache_length = past_key_values.get_seq_length()
    else:
        cache_length = past_key_values[0][0].shape[2]

    # Always take just the last token
    input_ids = input_ids[:, -1:]
    curr_query_len = 1

    # Position IDs: derive from logical position via attention_mask cumsum
    if attention_mask is not None and attention_mask.shape[1] > 1:
        position_ids = attention_mask.long().cumsum(-1)[:, -1:] - 1
    else:
        # Fallback: use cache_length as position (should not normally happen)
        position_ids = torch.tensor(
            [[cache_length]], dtype=torch.long, device=input_ids.device,
        )

    # Attention mask: short mask matching physical KV cache + current query
    attention_mask = torch.ones(
        (input_ids.shape[0], cache_length + curr_query_len),
        dtype=torch.long,
        device=input_ids.device,
    )

    return {
        'input_ids': input_ids,
        'position_ids': position_ids,
        'attention_mask': attention_mask,
        'past_key_values': past_key_values,
        'use_cache': kwargs.get('use_cache', True),
    }


# ---------------------------------------------------------------------------
# 3. VTR generate (wraps InternVLChatModel.generate)
# ---------------------------------------------------------------------------

def _vtr_generate(
    self,
    pixel_values=None,
    input_ids=None,
    attention_mask=None,
    visual_features=None,
    generation_config=None,
    output_hidden_states=None,
    **generate_kwargs,
):
    """
    VTR-aware replacement for ``InternVLChatModel.generate``.

    Flow:
        1. Compute image_token_range from input_ids (scan for token IDs)
        2. Inject VTR state into inner model (language_model.model)
        3. Call original generate (ViT + embed replacement + LLM generate)
        4. Clean up injected state

    Note: The original generate already handles ViT feature extraction and
    embedding replacement. We only need to inject the VTR state so that
    the prunable forward (on InternLM2Model) can pick it up.
    """
    inner = self.language_model.model  # InternLM2Model (patched by setup_vtr)
    vtr_cfg = getattr(inner, '_vtr_config', None)

    # ---- Compute image_token_range ----
    image_token_range = None
    if vtr_cfg is not None and vtr_cfg.enabled and input_ids is not None:
        image_token_range = _compute_image_token_range(self, input_ids)

    # ---- Inject VTR state ----
    if image_token_range is not None:
        inner._vtr_image_token_range = image_token_range
        inner._vtr_ctx = {}
        logger.debug(
            "VTR generate: injected image_token_range=%s", image_token_range,
        )

    try:
        # ---- Call original generate ----
        return self._original_generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_features=visual_features,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            **generate_kwargs,
        )
    finally:
        # ---- Cleanup injected state ----
        for attr in ('_vtr_image_token_range', '_vtr_ctx'):
            if hasattr(inner, attr):
                delattr(inner, attr)


# ---------------------------------------------------------------------------
# 4. setup_vtr_model — main entry point
# ---------------------------------------------------------------------------

def setup_vtr_model(model, config: VTRConfig, tokenizer=None) -> None:
    """
    Apply VTR (Visual Token Reduction) patches to an InternVLChatModel instance.

    This function:
      1. Calls ``setup_vtr()`` on the inner InternLM2Model to install the
         prunable forward (from ``prunable_internlm2.py``).
      2. Monkeypatches ``InternLM2ForCausalLM.prepare_inputs_for_generation``
         with a VTR-aware version that fixes decode-phase position/mask issues.
      3. Monkeypatches ``InternVLChatModel.generate`` with a VTR-aware version
         that injects/cleans up pruning state.
      4. Ensures ``img_context_token_id`` is set on the model (needed for
         ``_compute_image_token_range`` to work outside of ``chat()``).

    Args:
        model:     An ``InternVLChatModel`` instance.
        config:    A ``VTRConfig`` object.
        tokenizer: Optional tokenizer. If provided, used to resolve
                   ``img_context_token_id`` in case it has not been set yet.
    """
    # ---- 1. Patch inner InternLM2Model with prunable forward ----
    inner = model.language_model.model  # InternLM2Model
    setup_vtr(inner, config)
    logger.info(
        "VTR setup: patched InternLM2Model (enabled=%s, strategy=%s, "
        "prune_layers=%s, keep_ratio=%s, keep_tokens=%s)",
        config.enabled, config.strategy, config.prune_layers,
        config.keep_ratio, config.keep_tokens,
    )

    # ---- 2. Patch prepare_inputs_for_generation on InternLM2ForCausalLM ----
    lm = model.language_model  # InternLM2ForCausalLM
    lm._original_prepare_inputs_for_generation = lm.prepare_inputs_for_generation
    lm.prepare_inputs_for_generation = types.MethodType(
        _vtr_prepare_inputs_for_generation, lm,
    )

    # ---- 3. Patch generate on InternVLChatModel ----
    model._original_generate = model.generate
    model.generate = types.MethodType(_vtr_generate, model)

    # ---- 4. Ensure img_context_token_id is set ----
    if model.img_context_token_id is None:
        if tokenizer is not None:
            try:
                ctx_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
                model.img_context_token_id = ctx_id
                logger.debug("Set img_context_token_id=%d from tokenizer", ctx_id)
            except Exception:
                model.img_context_token_id = _IMG_CONTEXT_TOKEN_ID
                logger.debug(
                    "Tokenizer could not resolve <IMG_CONTEXT>; "
                    "using default ID=%d", _IMG_CONTEXT_TOKEN_ID,
                )
        else:
            model.img_context_token_id = _IMG_CONTEXT_TOKEN_ID
            logger.debug(
                "No tokenizer provided; using default img_context_token_id=%d",
                _IMG_CONTEXT_TOKEN_ID,
            )

    # ---- Store config on outer model for external access ----
    model._vtr_config = config
