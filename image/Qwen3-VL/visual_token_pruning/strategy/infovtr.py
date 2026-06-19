"""InfoVTR strategy implementation.

InfoVTR uses V-Information (S scores) to measure the task-specific
information gain of each visual token relative to a prior baseline.

S = P * log(P / Q)
- P: Task attention (with question)
- Q: Prior attention (without question, baseline)
- S > 0: Token provides extra information for the current task
- S < 0: Token is less important than baseline
- S ~ 0: Token has no task-specific contribution

Reference: V-Information based Visual Token Reduction
"""

import logging
from typing import Any, Dict, Tuple, Union

import torch

from ..config import VTRConfig
from .base import VTRStrategy

logger = logging.getLogger(__name__)


class InfoVTRStrategy(VTRStrategy):
    """InfoVTR strategy for visual token pruning.

    InfoVTR computes V-Information scores by comparing task-specific attention (P)
    against a prior baseline attention (Q). Tokens with higher S scores contribute
    more task-relevant information and should be preserved.

    The strategy requires a prior attention tensor (Q) computed from a prior
    forward pass (without the task question). This is passed via the context
    parameter as either:
        - prior_attention: A single tensor [num_image_tokens]
        - prior_attentions: A dict {layer_idx: tensor} for multi-layer support

    Example:
        >>> strategy = InfoVTRStrategy()
        >>> config = VTRConfig(enabled=True, strategy="infovtr", keep_ratio=0.25)
        >>> # P from task forward, Q from prior forward
        >>> scores = strategy.compute_scores(
        ...     attention, image_range, config,
        ...     prior_attention=prior_attn
        ... )
        >>> keep_indices = strategy.select_tokens(scores, num_img_tokens, config)
    """

    def compute_scores(
        self,
        attention: torch.Tensor,
        image_token_range: Tuple[int, int],
        config: VTRConfig,
        layer_idx: int = 0,
        **context: Any,
    ) -> torch.Tensor:
        """Compute V-Information scores (S = P * log(P / Q)).

        Args:
            attention: Task attention weights [batch, heads, seq, seq].
            image_token_range: Tuple of (start, end) indices for image tokens.
            config: VTR configuration with aggregation settings.
            **context: Must include one of:
                - prior_attention: Pre-computed prior scores [num_image_tokens].
                - prior_attentions: Dict[int, Tensor] mapping layer indices to
                  prior attention tensors. The smallest layer key is consumed
                  (popped) on each call, supporting multi-layer pruning.

        Returns:
            V-Information scores [num_image_tokens]. Higher values indicate
            tokens with more task-specific information.

        Raises:
            ValueError: If neither prior_attention nor prior_attentions is provided,
                or if prior_attentions dict is empty.
        """
        # Resolve prior attention (Q)
        prior_attention = self._resolve_prior_attention(context)

        # Record original dtype for output
        original_dtype = attention.dtype

        # Compute P (task attention aggregated to per-token scores)
        P = self._aggregate_attention(attention, image_token_range, config)

        # Prepare Q
        Q = prior_attention.to(P.device).float()

        # Validate shape compatibility
        if P.shape[0] != Q.shape[0]:
            logger.warning(
                f"Shape mismatch: P={P.shape[0]}, Q={Q.shape[0]}. "
                f"This may indicate prior was computed on a different sequence. "
                f"Truncating Q to match P."
            )
            Q = Q[: P.shape[0]]

        # Convert to float32 for numerical precision
        P = P.float()

        eps = 1e-10

        # Normalize to probability distributions
        P = P / (P.sum() + eps)
        Q = Q / (Q.sum() + eps)

        # Compute V-Information: S = P * log(P / Q)
        S = P * torch.log((P + eps) / (Q + eps))

        # Handle numerical issues
        if torch.isnan(S).any():
            logger.warning("NaN detected in InfoVTR scores, replacing with zeros.")
            S = torch.nan_to_num(S, nan=0.0)
        if torch.isinf(S).any():
            logger.warning("Inf detected in InfoVTR scores, clamping values.")
            S = torch.clamp(S, min=-1e6, max=1e6)

        if config.debug:
            logger.debug(
                f"InfoVTR scores: shape={S.shape}, "
                f"min={S.min():.6f}, max={S.max():.6f}, "
                f"mean={S.mean():.6f}, "
                f"positive_ratio={((S > 0).sum().item() / S.numel()):.2%}"
            )

        return S.to(original_dtype)

    def _resolve_prior_attention(
        self, context: Dict[str, Any]
    ) -> torch.Tensor:
        """Resolve prior attention from context.

        Supports two modes:
        1. Direct: context["prior_attention"] is a single tensor.
        2. Multi-layer: context["prior_attentions"] is a dict {layer: tensor}.
           The smallest layer key is consumed (popped) to support sequential
           multi-layer pruning.

        Args:
            context: The keyword arguments passed to compute_scores.

        Returns:
            Prior attention tensor [num_image_tokens].

        Raises:
            ValueError: If no valid prior attention source is found.
        """
        # Mode 1: Direct prior_attention tensor
        prior_attention = context.get("prior_attention")
        if prior_attention is not None:
            return prior_attention

        # Mode 2: Multi-layer prior_attentions dict
        prior_attentions = context.get("prior_attentions")
        if prior_attentions is not None:
            if not isinstance(prior_attentions, dict):
                raise ValueError(
                    f"prior_attentions must be a dict, got {type(prior_attentions)}"
                )
            if len(prior_attentions) == 0:
                raise ValueError(
                    "prior_attentions dict is empty. "
                    "No prior info available for this layer."
                )
            # Consume the smallest layer (simulate sequential forward)
            target_layer = min(prior_attentions.keys())
            return prior_attentions.pop(target_layer)

        raise ValueError(
            "InfoVTR requires 'prior_attention' or 'prior_attentions' in context. "
            "Run a prior forward pass first using extract_prior_attention()."
        )

    def _aggregate_attention(
        self,
        attention: torch.Tensor,
        image_token_range: Tuple[int, int],
        config: VTRConfig,
    ) -> torch.Tensor:
        """Aggregate attention weights to per-image-token scores.

        Extracts attention from query tokens to image tokens and aggregates
        across heads and query positions.

        Args:
            attention: Full attention tensor [batch, heads, seq, seq].
            image_token_range: Tuple of (start, end) for image tokens.
            config: Configuration with query_aggregation and head_aggregation.

        Returns:
            Aggregated scores [num_image_tokens].
        """
        img_start, img_end = image_token_range
        batch_size, num_heads, seq_len, _ = attention.shape

        # Query aggregation
        if config.query_aggregation == "last":
            # Use last token as query
            attn_to_img = attention[:, :, -1, img_start:img_end]
        elif config.query_aggregation == "question":
            # Average over question tokens (after image tokens)
            question_start = img_end
            if question_start >= seq_len - 1:
                # No question tokens available, fall back to last
                logger.warning(
                    "No question tokens found for query_aggregation='question'. "
                    "Falling back to 'last'."
                )
                attn_to_img = attention[:, :, -1, img_start:img_end]
            else:
                attn_to_img = attention[
                    :, :, question_start:-1, img_start:img_end
                ].mean(dim=2)
        else:
            raise ValueError(
                f"Unknown query_aggregation: {config.query_aggregation}"
            )

        # Head aggregation
        if config.head_aggregation == "mean":
            scores = attn_to_img.mean(dim=1)  # [batch, num_img]
        elif config.head_aggregation == "max":
            scores = attn_to_img.max(dim=1).values  # [batch, num_img]
        else:
            raise ValueError(
                f"Unknown head_aggregation: {config.head_aggregation}"
            )

        # Remove batch dimension
        if batch_size == 1:
            scores = scores.squeeze(0)
        else:
            logger.warning(
                f"Batch size > 1 ({batch_size}). Using first element only."
            )
            scores = scores[0]

        return scores

    def __repr__(self) -> str:
        """Return string representation of the strategy."""
        return "InfoVTRStrategy()"
