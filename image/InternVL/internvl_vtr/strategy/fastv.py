# FastV Strategy
from __future__ import annotations

from typing import Tuple, TYPE_CHECKING

import torch

from .base import PruningStrategy
from .registry import register_strategy

if TYPE_CHECKING:
    from ..config import VTRConfig


@register_strategy("fastv")
class FastVStrategy(PruningStrategy):
    """
    FastV pruning strategy.

    Selects Top-K image tokens based on the last token's average attention
    to the image region.

    Reference: FastV — An Image is Worth 1/2 Tokens After Layer 2
    """

    def compute_scores(
        self,
        attention: torch.Tensor,
        image_token_range: Tuple[int, int],
        config: VTRConfig,
        **ctx,
    ) -> torch.Tensor:
        """
        Compute FastV scores.

        Args:
            attention: [batch, heads, seq_len, seq_len] attention weights
            image_token_range: (img_start, img_end)
            config: VTR config (query_aggregation / head_aggregation)

        Returns:
            scores: [num_img_tokens] importance score per image token
        """
        if not isinstance(attention, torch.Tensor):
            raise TypeError(f"Attention must be a tensor, got type: {type(attention)}")

        img_start, img_end = image_token_range

        # Extract attention from query tokens to image tokens
        if config.query_aggregation == "last":
            relevant_attn = attention[:, :, -1, img_start:img_end]  # [batch, heads, num_img_tokens]
        elif config.query_aggregation == "question":
            relevant_attn = attention[:, :, img_end:, img_start:img_end].sum(dim=2)  # [batch, heads, num_img_tokens]
        else:
            raise ValueError(f"Invalid query aggregation mode: {config.query_aggregation}")

        # Aggregate across heads
        if config.head_aggregation == "mean":
            scores = relevant_attn.mean(dim=1).squeeze(0)  # [num_img_tokens]
        elif config.head_aggregation == "max":
            scores = relevant_attn.max(dim=1).values.squeeze(0)  # [num_img_tokens]
        else:
            raise ValueError(f"Invalid head aggregation mode: {config.head_aggregation}")

        return scores
