# PriorTR Strategy
# Single-forward V-Information pruning: extract P and Q from the same attention matrix via causal mask
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

    Key insight: LLaVA uses causal attention, so the '\\n' token (at img_end)
    can only attend to preceding content (i.e., image tokens), independent of the question.
    Therefore, in a single task forward pass:
        Q = attn[:, :, img_end, img_start:img_end]  — prior attention
        P = attn[:, :, -1,      img_start:img_end]  — task attention
        S = P · log(P / Q)                           — V-Information score

    Tokens with S > 0 carry extra information gain for the current task; keep Top-K.
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
            attention: [batch, heads, seq_len, seq_len] attention weights from the current layer
            image_token_range: (img_start, img_end)
            config: VTR config (supports query_aggregation / head_aggregation)

        Returns:
            scores: [num_img_tokens] V-Information score for each image token
        """
        if not isinstance(attention, torch.Tensor):
            raise TypeError(f"Attention must be a tensor, got type: {type(attention)}")

        img_start, img_end = image_token_range

        # ---- Q: '\n' token (img_end) attention to image tokens ----
        # Causal mask ensures '\n' can only see image tokens, equivalent to a prior forward
        Q = attention[:, :, img_end, img_start:img_end]  # [batch, heads, num_img]

        # ---- P: task attention (supports last / question aggregation) ----
        if config.query_aggregation == "last":
            P = attention[:, :, -1, img_start:img_end]  # [batch, heads, num_img]
        elif config.query_aggregation == "question":
            # question tokens: from img_end to end of sequence
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

        # ---- Normalize to probability distributions ----
        # Upcast to float32 first: in fp16/bf16 the eps below underflows and a
        # near-zero attention column can drive log((P+eps)/(Q+eps)) to inf/NaN.
        # The Qwen/Video PriorTR backbones all score in float32; match them here.
        original_dtype = P.dtype
        P = P.float()
        Q = Q.float()
        eps = 1e-10
        P = P / (P.sum() + eps)
        Q = Q / (Q.sum() + eps)

        # ---- V-Information: S = P · log(P / Q) ----
        S = P * torch.log((P + eps) / (Q + eps))

        # ---- Numerical stability (guard against any residual inf/NaN) ----
        if torch.isnan(S).any():
            S = torch.nan_to_num(S, nan=0.0)
        if torch.isinf(S).any():
            S = torch.clamp(S, min=-1e6, max=1e6)

        return S.to(original_dtype)
