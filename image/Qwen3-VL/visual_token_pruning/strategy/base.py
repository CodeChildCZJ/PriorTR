"""VTR Strategy base class module.

This module defines the abstract base class for visual token reduction strategies.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch

from ..config import VTRConfig

logger = logging.getLogger(__name__)


class VTRStrategy(ABC):
    """Abstract base class for Visual Token Reduction strategies.

    This class defines the interface for all VTR strategies. Concrete implementations
    must override the `compute_scores` method to provide their scoring logic.

    The `select_tokens` method provides a common implementation for selecting
    which tokens to keep based on the computed scores.

    Example:
        >>> class MyStrategy(VTRStrategy):
        ...     def compute_scores(self, attention, image_token_range, config, **context):
        ...         # Custom scoring logic
        ...         return scores
        >>> strategy = MyStrategy()
        >>> scores = strategy.compute_scores(attn, (10, 50), config)
        >>> keep_indices = strategy.select_tokens(scores, 40, config)
    """

    def prepare(
        self,
        hidden_states: torch.Tensor,
        config: VTRConfig,
        context: Dict[str, Any],
    ) -> None:
        """Called once before the layer loop. Override for setup work."""
        pass

    @abstractmethod
    def compute_scores(
        self,
        attention: torch.Tensor,
        image_token_range: Tuple[int, int],
        config: VTRConfig,
        layer_idx: int = 0,
        **context: Any,
    ) -> torch.Tensor:
        """Compute importance scores for image tokens.

        Args:
            attention: Attention weights tensor with shape [batch, heads, seq, seq].
            image_token_range: Tuple of (start, end) indices for image tokens
                in the sequence.
            config: VTR configuration object.
            layer_idx: Index of the current pruning layer (0-based among prune layers).
            **context: Additional context that may be needed by specific strategies.
                For example, PriorTR-2F requires `prior_attention`.

        Returns:
            A 1D tensor of shape [num_image_tokens] containing importance scores
            for each image token. Higher scores indicate more important tokens.

        Raises:
            NotImplementedError: This method must be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement compute_scores")

    def select_tokens(
        self,
        scores: torch.Tensor,
        num_tokens: int,
        config: VTRConfig,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """Select which tokens to keep based on their scores.

        Priority: keep_tokens > score_threshold > keep_ratio.
        Supports per-layer config via layer_idx.

        Args:
            scores: Importance scores for each image token, shape [num_image_tokens].
            num_tokens: Total number of image tokens.
            config: VTR configuration object.
            layer_idx: Index of the current pruning layer (0-based among prune layers).

        Returns:
            A 1D tensor of indices (relative to image token range) indicating
            which tokens to keep. Indices are sorted in ascending order to
            preserve the original token order.
        """
        device = scores.device

        # Priority 1: keep_tokens (exact count)
        if config.keep_tokens is not None:
            k = config.get_keep_count(num_tokens, layer_idx)
            if k >= num_tokens:
                return torch.arange(num_tokens, dtype=torch.long, device=device)
            if k <= 0:
                return torch.tensor([], dtype=torch.long, device=device)
            keep_indices = scores.topk(k).indices
            return keep_indices.sort().values

        # Priority 2: score_threshold
        if config.score_threshold is not None:
            threshold = config.score_threshold
            if isinstance(threshold, list):
                threshold = threshold[layer_idx]
            mask = scores > threshold
            if mask.sum() == 0:
                # No tokens above threshold, keep the highest scoring one
                keep_indices = scores.topk(1).indices
            else:
                keep_indices = mask.nonzero(as_tuple=True)[0]
            return keep_indices.sort().values

        # Priority 3: keep_ratio
        k = config.get_keep_count(num_tokens, layer_idx)
        if k >= num_tokens:
            return torch.arange(num_tokens, dtype=torch.long, device=device)
        if k <= 0:
            return torch.tensor([], dtype=torch.long, device=device)
        keep_indices = scores.topk(k).indices
        return keep_indices.sort().values

    def post_prune(
        self,
        hidden_states: torch.Tensor,
        pruned_token_hidden_states: torch.Tensor,
        keep_indices: torch.Tensor,
        image_token_range: Tuple[int, int],
        config: VTRConfig,
        layer_idx: int,
        **context: Any,
    ) -> torch.Tensor:
        """Post-pruning hook. Override for token merge. Returns hidden_states."""
        return hidden_states

    def __repr__(self) -> str:
        """Return string representation of the strategy."""
        return f"{self.__class__.__name__}()"
