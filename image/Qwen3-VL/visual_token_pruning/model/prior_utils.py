"""Prior forward utilities for InfoVTR strategy.

This module provides functions to construct prior inputs and extract
prior attention scores for the InfoVTR visual token reduction strategy.

The prior forward pass generates baseline attention (Q) by running the model
on an input without the task-specific question, allowing the InfoVTR strategy
to compute V-Information scores: S = P * log(P / Q).

Two modes for constructing prior input:
    - truncate: Truncate the original input at the vision_end token
    - template: Use a fixed template with only system prompt and image

Prior attention extraction uses the Lightweight Look-ahead approach:
manually iterating decoder layers to the target layer, then computing
attention weights via Q@K^T. This avoids relying on output_attentions=True
which Qwen3-VL's architecture does not support.
"""

import logging
from typing import Optional, Tuple

import torch
from transformers.masking_utils import create_causal_mask

from ..config import VTRConfig

logger = logging.getLogger(__name__)


def build_prior_input(
    input_ids: torch.Tensor,
    processor: object,
    config: VTRConfig,
    model_config: object,
) -> torch.Tensor:
    """Construct prior input IDs for the prior forward pass.

    The prior input removes the task-specific question from the input,
    keeping only the system prompt and image tokens. This provides a
    task-agnostic baseline for attention comparison.

    Args:
        input_ids: Original input IDs tensor with shape [batch, seq_len].
        processor: The Qwen3VL processor (used for tokenization in template mode).
        config: VTR configuration object containing prior_mode setting.
        model_config: Model configuration object with special token IDs.

    Returns:
        Prior input IDs tensor with shape [batch, prior_seq_len].

    Raises:
        ValueError: If prior_mode is unknown or required tokens are not found.
    """
    if config.prior_mode == "truncate":
        return _build_prior_truncate(input_ids, model_config)
    elif config.prior_mode == "template":
        return _build_prior_template(input_ids, processor, model_config)
    else:
        raise ValueError(f"Unknown prior_mode: {config.prior_mode}")


def _build_prior_truncate(
    input_ids: torch.Tensor,
    model_config: object,
) -> torch.Tensor:
    """Build prior input by truncating at vision_end token.

    Truncates the input sequence right after the last vision_end token,
    effectively removing the question/instruction text.

    Original: [SYS] + [vision_start] + [IMG] + [vision_end] + [QUESTION]
    Prior:    [SYS] + [vision_start] + [IMG] + [vision_end]

    Args:
        input_ids: Original input IDs [batch, seq_len].
        model_config: Model config with vision_end_token_id attribute.

    Returns:
        Truncated prior input IDs [batch, truncated_len].

    Raises:
        ValueError: If no vision_end_token is found in input_ids.
    """
    vision_end_id = model_config.vision_end_token_id

    # Find all positions of vision_end_token
    vision_end_mask = input_ids == vision_end_id
    vision_end_positions = vision_end_mask.nonzero(as_tuple=False)

    if len(vision_end_positions) == 0:
        raise ValueError(
            "No vision_end_token found in input_ids. "
            "Ensure the input contains image tokens."
        )

    # Use the last vision_end position (handles multi-image cases)
    last_vision_end = vision_end_positions[-1, 1].item()

    # Truncate to include vision_end_token
    prior_input_ids = input_ids[:, : last_vision_end + 1]

    logger.debug(
        f"Prior truncate: original_len={input_ids.shape[1]}, "
        f"prior_len={prior_input_ids.shape[1]}, "
        f"vision_end_pos={last_vision_end}"
    )

    return prior_input_ids


def _build_prior_template(
    input_ids: torch.Tensor,
    processor: object,
    model_config: object,
) -> torch.Tensor:
    """Build prior input using a fixed template.

    Constructs a minimal input with the standard chat template structure
    containing only the system prompt and image tokens, without any
    question text.

    Template:
        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        <|vision_start|><|image_pad|>*N<|vision_end|><|im_end|>

    Args:
        input_ids: Original input IDs [batch, seq_len] (used to count image tokens).
        processor: Qwen3VL processor with tokenizer for encoding the template.
        model_config: Model config with image_token_id attribute.

    Returns:
        Template-based prior input IDs [batch, template_len].
    """
    image_token_id = model_config.image_token_id

    # Count image tokens in original input
    num_image_tokens = (input_ids == image_token_id).sum().item()

    if num_image_tokens == 0:
        logger.warning(
            "No image tokens found in input_ids for template mode. "
            "Prior will have no image content."
        )

    # Build template string
    image_placeholder = "<|image_pad|>" * num_image_tokens
    template = (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"<|vision_start|>{image_placeholder}<|vision_end|><|im_end|>"
    )

    # Tokenize the template
    prior_input_ids = processor.tokenizer(
        template,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(input_ids.device)

    logger.debug(
        f"Prior template: num_image_tokens={num_image_tokens}, "
        f"prior_len={prior_input_ids.shape[1]}"
    )

    return prior_input_ids


def compute_prior_image_token_range(
    prior_input_ids: torch.Tensor,
    model_config: object,
) -> Tuple[int, int]:
    """Compute the image token range in the prior input sequence.

    Since the prior input may have a different sequence length than the
    original task input, the image token positions need to be recalculated.

    Args:
        prior_input_ids: Prior input IDs [batch, prior_seq_len].
        model_config: Model config with image_token_id attribute.

    Returns:
        Tuple of (start, end) indices for image tokens in the prior sequence.

    Raises:
        ValueError: If no image tokens are found in prior_input_ids.
    """
    image_token_id = model_config.image_token_id

    image_positions = (prior_input_ids[0] == image_token_id).nonzero(as_tuple=False)

    if len(image_positions) == 0:
        raise ValueError(
            "No image tokens found in prior_input_ids. "
            "Cannot compute image token range."
        )

    img_start = image_positions[0, 0].item()
    img_end = image_positions[-1, 0].item() + 1

    return (img_start, img_end)


def extract_prior_attention(
    model: object,
    prior_input_ids: torch.Tensor,
    pixel_values: Optional[torch.Tensor],
    image_grid_thw: Optional[torch.Tensor],
    image_token_range: Tuple[int, int],
    prune_layer: int,
    config: VTRConfig,
) -> torch.Tensor:
    """Execute prior forward pass and extract attention scores.

    Uses the Lightweight Look-ahead approach: manually iterates decoder layers
    up to the target layer, then uses _score_layer_forward to extract attention
    weights via Q@K^T computation. This avoids relying on output_attentions=True
    which Qwen3-VL does not support.

    Args:
        model: The VTRQwen3VLForConditionalGeneration (or compatible) model.
        prior_input_ids: Prior input IDs [batch, prior_seq_len].
        pixel_values: Pixel values tensor for image encoding, or None.
        image_grid_thw: Image grid dimensions [num_images, 3], or None.
        image_token_range: Tuple of (start, end) for image tokens in
            the prior sequence.
        prune_layer: The layer at which pruning will occur. Attention
            is extracted from layer (prune_layer - 1).
        config: VTR configuration with aggregation settings.

    Returns:
        Prior attention scores [num_image_tokens] as a 1D tensor.
        These scores represent the baseline attention distribution (Q)
        for V-Information computation.
    """
    target_layer_idx = prune_layer - 1 if prune_layer > 0 else 0
    lm = model.model.language_model  # PrunableQwen3VLTextModel

    with torch.no_grad():
        # Step 1: Prepare hidden_states by iterating layers 0..(target-1)
        hidden_states, attention_mask, cache_position, position_embeddings = \
            _prepare_prior_hidden_states(
                model, prior_input_ids, pixel_values,
                image_grid_thw, target_layer_idx
            )

        # Step 2: Extract attention weights at target layer via Look-ahead
        _, attn_weights = lm._score_layer_forward(
            lm.layers[target_layer_idx],
            hidden_states,
            attention_mask=attention_mask,
            past_key_values=None,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

    # Step 3: Aggregate attention weights to per-token scores
    prior_scores = _aggregate_prior_scores(attn_weights, image_token_range, config)

    logger.debug(
        f"Prior attention extracted: layer={target_layer_idx}, "
        f"shape={prior_scores.shape}, "
        f"min={prior_scores.min():.6f}, max={prior_scores.max():.6f}"
    )

    return prior_scores


def _prepare_prior_hidden_states(
    model: object,
    prior_input_ids: torch.Tensor,
    pixel_values: Optional[torch.Tensor],
    image_grid_thw: Optional[torch.Tensor],
    target_layer_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Prepare hidden states for the prior forward pass.

    Builds input embeddings (with scattered image features), computes
    position IDs and causal mask, then iterates decoder layers 0 through
    (target_layer_idx - 1) to produce the hidden states that will be
    input to the target score layer.

    Handles DeepStack injection at each layer as needed.

    Args:
        model: The VTRQwen3VLForConditionalGeneration model.
        prior_input_ids: Prior input IDs [batch, prior_seq_len].
        pixel_values: Pixel values for image encoding, or None.
        image_grid_thw: Image grid dimensions [num_images, 3], or None.
        target_layer_idx: The layer index at which to extract attention
            (layers 0..target_layer_idx-1 are iterated).

    Returns:
        Tuple of:
            - hidden_states: [batch, prior_seq_len, hidden_dim]
            - attention_mask: 4D causal mask [batch, 1, seq, seq]
            - cache_position: [prior_seq_len]
            - position_embeddings: (cos, sin) tuple from rotary embedding
    """
    qwen_model = model.model  # Qwen3VLModel
    lm = qwen_model.language_model  # PrunableQwen3VLTextModel

    # --- 1. Build input embeddings ---
    inputs_embeds = lm.embed_tokens(prior_input_ids)

    visual_pos_masks = None
    deepstack_visual_embeds = None

    if pixel_values is not None:
        vision_output = qwen_model.get_image_features(
            pixel_values, image_grid_thw
        )
        image_embeds = vision_output.pooler_output
        deepstack_image_embeds = vision_output.deepstack_features
        image_embeds = torch.cat(image_embeds, dim=0).to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        image_mask, _ = qwen_model.get_placeholder_mask(
            prior_input_ids, inputs_embeds=inputs_embeds,
            image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        visual_pos_masks = image_mask[..., 0]  # [batch, seq]
        deepstack_visual_embeds = deepstack_image_embeds

    # --- 2. Compute position IDs (3D MRoPE) ---
    position_ids, _ = qwen_model.get_rope_index(
        prior_input_ids, image_grid_thw, video_grid_thw=None
    )

    # Separate text position IDs from 3D spatial position IDs
    if position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]   # [batch, seq]
        position_ids_3d = position_ids[1:]    # [3, batch, seq]
    else:
        text_position_ids = position_ids[0]   # [batch, seq]
        position_ids_3d = position_ids        # [3, batch, seq]

    # --- 3. Build causal mask + cache_position ---
    seq_len = inputs_embeds.shape[1]
    device = inputs_embeds.device
    cache_position = torch.arange(seq_len, device=device)

    attention_mask = create_causal_mask(
        config=lm.config,
        input_embeds=inputs_embeds,
        attention_mask=None,         # prior has no padding
        cache_position=cache_position,
        past_key_values=None,        # no cache for prior
        position_ids=text_position_ids,
    )

    # --- 4. Compute RoPE position embeddings ---
    hidden_states = inputs_embeds
    position_embeddings = lm.rotary_emb(hidden_states, position_ids_3d)

    # --- 5. Iterate layers 0..(target_layer_idx - 1) ---
    for i in range(target_layer_idx):
        hidden_states = lm.layers[i](
            hidden_states,
            attention_mask=attention_mask,
            position_ids=text_position_ids,
            past_key_values=None,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        # DeepStack injection (after each layer, same as main forward)
        if (
            deepstack_visual_embeds is not None
            and i < len(deepstack_visual_embeds)
            and deepstack_visual_embeds[i] is not None
        ):
            hidden_states = lm._deepstack_process(
                hidden_states, visual_pos_masks, deepstack_visual_embeds[i]
            )

    return hidden_states, attention_mask, cache_position, position_embeddings


def _aggregate_prior_scores(
    attn_weights: torch.Tensor,
    image_token_range: Tuple[int, int],
    config: VTRConfig,
) -> torch.Tensor:
    """Aggregate attention weights to per-image-token prior scores.

    Extracts the attention from the last token (query) to all image tokens
    (keys), then aggregates across attention heads.

    Args:
        attn_weights: Full attention weights [batch, num_heads, seq, seq].
        image_token_range: Tuple of (start, end) for image tokens.
        config: VTR configuration with head_aggregation setting.

    Returns:
        Prior scores [num_image_tokens] as a 1D tensor.
    """
    img_start, img_end = image_token_range

    # Use last token as query (prior has no question, last = vision_end or im_end)
    attn_to_img = attn_weights[:, :, -1, img_start:img_end]  # [batch, heads, num_img]

    # Aggregate across heads
    if config.head_aggregation == "mean":
        prior_scores = attn_to_img.mean(dim=1).squeeze(0)  # [num_img]
    elif config.head_aggregation == "max":
        prior_scores = attn_to_img.max(dim=1).values.squeeze(0)  # [num_img]
    else:
        raise ValueError(f"Unknown head_aggregation: {config.head_aggregation}")

    return prior_scores
