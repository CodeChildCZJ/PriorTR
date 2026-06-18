# PriorTR Strategy
# Single-forward V-Information pruning: extracts P and Q from the same
# attention matrix using the causal mask.
from __future__ import annotations

from typing import Tuple, TYPE_CHECKING

import torch

from .base import PruningStrategy
from .registry import register_strategy

if TYPE_CHECKING:
    from ..config import VTRConfig


@register_strategy("priortr")
class PriorTRStrategy(PruningStrategy):
    """
    PriorTR pruning strategy.

    Key insight: causal attention means the '\\n' token (at img_end+1)
    can only attend to preceding content (image tokens), not the question.
    In a single task forward pass:
        Q = attn[:, :, img_end, img_start:img_end]  -- prior attention
        P = attn[:, :, -1,      img_start:img_end]  -- task attention
        S = P * log(P / Q)                           -- V-Information score

    Tokens with S > 0 carry extra information gain for the task; Top-K are kept.
    """

    def compute_scores(
        self,
        attention: torch.Tensor,
        image_token_range: Tuple[int, int],
        config: VTRConfig,
        **ctx,
    ) -> torch.Tensor:
        """
        Compute PriorTR scores.

        Args:
            attention: [batch, heads, seq_len, seq_len] attention weights
            image_token_range: (img_start, img_end)
            config: VTR config (query_aggregation / head_aggregation)

        Returns:
            scores: [num_img_tokens] V-Information score per image token
        """
        if not isinstance(attention, torch.Tensor):
            raise TypeError(f"Attention must be a tensor, got type: {type(attention)}")

        img_start, img_end = image_token_range

        # ---- Q: '\n' token (img_end + 1) attending to image tokens ----
        # InternVL token sequence: ... [IMG_CONTEXT*N] [</img>] [\n] [question] ...
        # img_end is the position of </img>, img_end+1 is \n
        # Under the causal mask, \n can only see image tokens + </img>, not the question.
        # We use \n (not </img>) as the Q anchor, consistent with LLaVA PriorTR.
        Q = attention[:, :, img_end + 1, img_start:img_end]  # [batch, heads, num_img]

        # ---- P: task attention (supports last / question aggregation) ----
        if config.query_aggregation == "last":
            P = attention[:, :, -1, img_start:img_end]  # [batch, heads, num_img]
        elif config.query_aggregation == "question":
            # question tokens: everything after img_end to end of sequence
            P = attention[:, :, img_end:, img_start:img_end]  # [batch, heads, num_q, num_img]
            P = P.mean(dim=2)  # [batch, heads, num_img]
        else:
            raise ValueError(f"Invalid query aggregation mode: {config.query_aggregation}")

        # ---- Head aggregation ----
        if config.head_aggregation == "mean":
            P = P.mean(dim=1).squeeze(0)  # [num_img]
            Q = Q.mean(dim=1).squeeze(0)  # [num_img]
        elif config.head_aggregation == "max":
            P = P.max(dim=1).values.squeeze(0)
            Q = Q.max(dim=1).values.squeeze(0)
        else:
            raise ValueError(f"Invalid head aggregation mode: {config.head_aggregation}")

        # ---- Normalization & V-Information ----
        eps = 1e-10
        P = P / (P.sum() + eps)
        Q = Q / (Q.sum() + eps)
        S = P * torch.log((P + eps) / (Q + eps))

        return S
