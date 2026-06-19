# PriorTR-2F Strategy
from __future__ import annotations

import logging
from typing import Tuple, TYPE_CHECKING

import torch

from .base import PruningStrategy
from .registry import register_strategy

if TYPE_CHECKING:
    from ..config import VTRConfig, PriorTR2FConfig

logger = logging.getLogger(__name__)


@register_strategy("priortr_2f")
class PriorTR2FStrategy(PruningStrategy):
    """
    PriorTR-2F pruning strategy (two-forward variant of PriorTR).

    Same task attention P and V-Information score S = P * log(P / Q) as PriorTR;
    the prior Q comes from an explicit question-free forward pass rather than
    PriorTR's single-forward causal-mask shortcut. Video-LLaVA uses this
    two-forward form because video lacks the causal-mask shortcut that lets
    image-only PriorTR read the prior from a single forward.

    - P: Task attention (with the question)
    - Q: Prior attention (explicit prior forward, without the question)
    - S > 0 indicates extra information gain for the current task

    Reference: V-Information based Visual Token Reduction
    """

    def compute_scores(
        self,
        attention: torch.Tensor,
        image_token_range: Tuple[int, int],
        config: VTRConfig,
        **ctx,
    ) -> torch.Tensor:
        """
        Compute PriorTR-2F scores (S = P * log(P / Q)).

        Logic:
        ctx["prior_attentions"] is a {layer_idx: tensor} dictionary.
        Each call pops the shallowest (smallest layer_idx) prior attention,
        simulating layer-by-layer consumption of prior information during forward.

        Args:
            attention: Task attention (P) [batch, heads, seq, seq]
            image_token_range: (img_start, img_end)
            config: VTR configuration
            ctx["prior_attentions"]: Dict[int, torch.Tensor] storing each layer's prior attention

        Returns:
            scores: [num_image_tokens] V-Information scores
        """
        # 1. Get prior_attentions dictionary
        prior_attentions = ctx.get("prior_attentions")
        if prior_attentions is None or not isinstance(prior_attentions, dict):
            raise ValueError(f"PriorTR-2F requires 'prior_attentions' to be a dict in ctx. Got: {type(prior_attentions)}")

        if len(prior_attentions) == 0:
            raise ValueError("prior_attentions dict is empty! No prior info available for this layer.")

        # 2. Simulate forward: get the current lowest layer number
        target_layer = min(prior_attentions.keys())

        # 3. Pop and consume this layer's prior attention
        # This modifies the ctx dict, ensuring the next forward call won't reuse this layer
        prior_attention = prior_attentions.pop(target_layer)

        # Record original dtype
        original_dtype = attention.dtype

        # Compute P (task attention)
        P = self._aggregate_attention(attention, image_token_range, config)

        # Shape mismatch handling for multi-layer pruning
        if P.shape[0] != prior_attention.shape[0]:
             if prior_attention.shape[0] > P.shape[0]:
                 logger.warning(f"Shape mismatch in PriorTR-2F: P={P.shape}, Q={prior_attention.shape}. "
                                f"This might cause errors if Q was not pruned along with P.")

        # Convert to float32 for high-precision computation
        P = P.float()
        Q = prior_attention.to(P.device).float()

        eps = 1e-10
        # Normalize to probability distributions
        P = P / (P.sum() + eps)
        Q = Q / (Q.sum() + eps)

        # Compute S = P * log(P / Q) (V-Information)
        S = P * torch.log((P + eps) / (Q + eps))
        # Numerical stability checks
        if torch.isnan(S).any():
            logger.warning("NaN detected in PriorTR-2F scores, replacing with zeros")
            S = torch.nan_to_num(S, nan=0.0)
        if torch.isinf(S).any():
            logger.warning("Inf detected in PriorTR-2F scores, clamping values")
            S = torch.clamp(S, min=-1e6, max=1e6)

        return S.to(original_dtype)
