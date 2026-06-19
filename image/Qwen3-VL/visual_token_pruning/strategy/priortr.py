"""PriorTR strategy implementation.

PriorTR computes V-Information scores from a single forward pass by
exploiting the causal attention mask. The <|vision_end|> token (at img_end)
can only attend to tokens before it (system + image), making its attention
to image tokens a natural prior (Q). The last token's attention serves as
the task attention (P).

S = P * log(P / Q)

Reference: PriorTR — single-forward V-Information for visual token reduction.
"""

import logging
from typing import Any, Tuple

import torch

from ..config import VTRConfig
from .base import VTRStrategy

logger = logging.getLogger(__name__)


class PriorTRStrategy(VTRStrategy):
    """PriorTR strategy for visual token pruning.

    Extracts both task attention (P) and prior attention (Q) from the same
    attention matrix using the causal mask property:
        Q = attn[:, :, img_end, img_start:img_end]  (vision_end -> images)
        P = attn[:, :, -1, img_start:img_end]        (last token -> images)
        S = P * log(P / Q)                            (V-Information)

    No prior forward pass needed — single forward, same as FastV.

    Example:
        >>> strategy = PriorTRStrategy()
        >>> config = VTRConfig(enabled=True, strategy="priortr", keep_ratio=0.5)
        >>> scores = strategy.compute_scores(attention, (10, 50), config)
    """

    def compute_scores(
        self,
        attention: torch.Tensor,
        image_token_range: Tuple[int, int],
        config: VTRConfig,
        layer_idx: int = 0,
        **context: Any,
    ) -> torch.Tensor:
        """Compute V-Information scores from a single attention matrix.

        Args:
            attention: Attention weights [batch, heads, seq, seq].
            image_token_range: (img_start, img_end) for image tokens.
            config: VTR configuration with aggregation settings.
            layer_idx: Pruning layer index (unused, for interface compat).
            **context: Additional context (unused by PriorTR).

        Returns:
            V-Information scores [num_image_tokens].
        """
        img_start, img_end = image_token_range
        batch_size, num_heads, seq_len, _ = attention.shape

        # Q: <|vision_end|> token's attention to image tokens (causal prior)
        Q = attention[:, :, img_end, img_start:img_end]  # [B, H, num_img]

        # P: task attention (last token or question tokens)
        if config.query_aggregation == "last":
            P = attention[:, :, -1, img_start:img_end]  # [B, H, num_img]
        elif config.query_aggregation == "question":
            question_start = img_end
            if question_start >= seq_len - 1:
                logger.warning(
                    "No question tokens found for query_aggregation='question'. "
                    "Falling back to 'last'."
                )
                P = attention[:, :, -1, img_start:img_end]
            else:
                P = attention[:, :, question_start:-1, img_start:img_end]
                P = P.mean(dim=2)  # [B, H, num_img]
        else:
            raise ValueError(
                f"Unknown query_aggregation: {config.query_aggregation}"
            )

        # Head aggregation
        if config.head_aggregation == "mean":
            P = P.mean(dim=1)  # [B, num_img]
            Q = Q.mean(dim=1)
        elif config.head_aggregation == "max":
            P = P.max(dim=1).values
            Q = Q.max(dim=1).values
        else:
            raise ValueError(
                f"Unknown head_aggregation: {config.head_aggregation}"
            )

        # Remove batch dimension
        if batch_size == 1:
            P = P.squeeze(0)  # [num_img]
            Q = Q.squeeze(0)
        else:
            logger.warning(
                f"Batch size > 1 ({batch_size}). Using first element only."
            )
            P = P[0]
            Q = Q[0]

        # Convert to float32 for numerical precision
        original_dtype = P.dtype
        P = P.float()
        Q = Q.float()

        # Normalize to probability distributions
        eps = 1e-10
        P = P / (P.sum() + eps)
        Q = Q / (Q.sum() + eps)

        # V-Information: S = P * log(P / Q)
        S = P * torch.log((P + eps) / (Q + eps))

        # Numerical stability
        if torch.isnan(S).any():
            logger.warning("NaN detected in PriorTR scores, replacing with zeros.")
            S = torch.nan_to_num(S, nan=0.0)
        if torch.isinf(S).any():
            logger.warning("Inf detected in PriorTR scores, clamping values.")
            S = torch.clamp(S, min=-1e6, max=1e6)

        if config.debug:
            logger.debug(
                f"PriorTR scores: shape={S.shape}, "
                f"min={S.min():.6f}, max={S.max():.6f}, "
                f"mean={S.mean():.6f}, "
                f"positive_ratio={((S > 0).sum().item() / S.numel()):.2%}"
            )

        return S.to(original_dtype)

    def __repr__(self) -> str:
        return "PriorTRStrategy()"
