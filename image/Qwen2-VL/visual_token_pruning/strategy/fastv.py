"""FastV strategy implementation.

FastV uses attention weights from the last token to image tokens as importance scores.
Reference: "FastV: An Image is Worth 1/2 Tokens After Layer 2"
"""

import logging
from typing import Any, Tuple

import torch

from ..config import VTRConfig
from .base import VTRStrategy

logger = logging.getLogger(__name__)


class FastVStrategy(VTRStrategy):
    """FastV strategy for visual token pruning.

    FastV computes importance scores based on attention weights from query tokens
    (typically the last token) to image tokens. Tokens with higher attention
    receive higher importance scores.

    The algorithm:
    1. Extract attention weights from query token(s) to image tokens
    2. Aggregate across attention heads (mean or max)
    3. Select top-k tokens based on aggregated scores

    Attributes:
        None (stateless strategy)

    Example:
        >>> strategy = FastVStrategy()
        >>> # attention shape: [batch, heads, seq_len, seq_len]
        >>> attention = torch.randn(1, 32, 100, 100)
        >>> image_range = (10, 50)  # image tokens at positions 10-49
        >>> config = VTRConfig(enabled=True, keep_ratio=0.5)
        >>> scores = strategy.compute_scores(attention, image_range, config)
        >>> scores.shape
        torch.Size([40])
    """

    def compute_scores(
        self,
        attention: torch.Tensor,
        image_token_range: Tuple[int, int],
        config: VTRConfig,
        layer_idx: int = 0,
        **context: Any,
    ) -> torch.Tensor:
        """Compute importance scores using attention weights.

        Args:
            attention: Attention weights tensor with shape [batch, heads, seq, seq].
                The attention should be from the layer before pruning (layer K-1
                if pruning at layer K).
            image_token_range: Tuple of (start, end) indices for image tokens
                in the sequence. The range is [start, end) (end exclusive).
            config: VTR configuration object containing:
                - query_aggregation: "last" or "question"
                - head_aggregation: "mean" or "max"
            **context: Additional context (unused by FastV).

        Returns:
            A 1D tensor of shape [num_image_tokens] containing importance scores.
            Higher scores indicate more important tokens.

        Raises:
            ValueError: If attention tensor has unexpected shape.
        """
        img_start, img_end = image_token_range
        num_image_tokens = img_end - img_start

        if num_image_tokens <= 0:
            logger.warning(
                f"Invalid image token range: ({img_start}, {img_end}). "
                f"Returning empty scores."
            )
            return torch.tensor([], dtype=attention.dtype, device=attention.device)

        # Validate attention shape
        if attention.dim() != 4:
            raise ValueError(
                f"Expected attention tensor with 4 dimensions [batch, heads, seq, seq], "
                f"got shape {attention.shape}"
            )

        batch_size, num_heads, seq_len, _ = attention.shape

        # Extract attention based on query aggregation strategy
        if config.query_aggregation == "last":
            # Use attention from the last token to image tokens
            # Shape: [batch, heads, num_img]
            relevant_attn = attention[:, :, -1, img_start:img_end]

        elif config.query_aggregation == "question":
            # Use attention from question tokens (after image) to image tokens
            # Question tokens are from img_end to seq_len-1 (excluding last)
            question_start = img_end
            if question_start >= seq_len - 1:
                # No question tokens, fall back to last token
                logger.warning(
                    f"No question tokens found (question_start={question_start}, "
                    f"seq_len={seq_len}). Falling back to 'last' aggregation."
                )
                relevant_attn = attention[:, :, -1, img_start:img_end]
            else:
                # Average attention from all question tokens
                # Shape: [batch, heads, num_question, num_img] -> [batch, heads, num_img]
                relevant_attn = attention[:, :, question_start:-1, img_start:img_end]
                relevant_attn = relevant_attn.mean(dim=2)
        else:
            # This should not happen if config validation is correct
            raise ValueError(f"Unknown query_aggregation: {config.query_aggregation}")

        # Aggregate across attention heads
        if config.head_aggregation == "mean":
            # Average across heads: [batch, heads, num_img] -> [batch, num_img]
            scores = relevant_attn.mean(dim=1)
        elif config.head_aggregation == "max":
            # Max across heads: [batch, heads, num_img] -> [batch, num_img]
            scores = relevant_attn.max(dim=1).values
        else:
            # This should not happen if config validation is correct
            raise ValueError(f"Unknown head_aggregation: {config.head_aggregation}")

        # Remove batch dimension (assuming batch_size=1 for now)
        # TODO: Support batch_size > 1 in future
        if batch_size == 1:
            scores = scores.squeeze(0)  # [num_img]
        else:
            logger.warning(
                f"Batch size > 1 ({batch_size}) detected. "
                f"Using first batch element only."
            )
            scores = scores[0]  # [num_img]

        if config.debug:
            logger.debug(
                f"FastV scores computed: shape={scores.shape}, "
                f"min={scores.min():.4f}, max={scores.max():.4f}, "
                f"mean={scores.mean():.4f}"
            )

        return scores

    def __repr__(self) -> str:
        """Return string representation of the strategy."""
        return "FastVStrategy()"
