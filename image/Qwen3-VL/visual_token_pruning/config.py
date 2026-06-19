"""VTR (Visual Token Reduction) configuration module.

This module defines the configuration dataclass for visual token pruning strategies.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class VTRConfig:
    """Configuration for Visual Token Reduction (VTR).

    This configuration class controls the behavior of visual token pruning
    strategies such as FastV and InfoVTR.

    Attributes:
        enabled: Whether VTR is enabled. If False, no pruning occurs.
        strategy: The pruning strategy to use. Options: "fastv", "infovtr", "sparsevlm", "priortr", "vispruner".

        prune_layer: The layer(s) at which to perform pruning.
            - int: Single layer pruning (FastV/InfoVTR style)
            - List[int]: Multi-layer pruning (SparseVLMs style, reserved)

        keep_ratio: Fraction of image tokens to keep (0.0 to 1.0).
            Used when neither keep_tokens nor score_threshold is set.
        keep_tokens: Exact number of tokens to keep.
            Takes priority over keep_ratio and score_threshold.
        score_threshold: Score threshold for keeping tokens.
            Only tokens with scores above this threshold are kept.
            Takes priority over keep_ratio but not keep_tokens.

        query_aggregation: How to aggregate query tokens for attention extraction.
            - "auto": Resolve per strategy ("question" for priortr/infovtr, "last" for others)
            - "last": Use only the last token
            - "question": Average over all question tokens
        head_aggregation: How to aggregate attention across heads.
            - "mean": Average attention across heads
            - "max": Take maximum attention across heads

        prior_prompt: Prompt used for prior attention in InfoVTR.
        prior_mode: How to construct prior input for InfoVTR.
            - "truncate": Truncate input at image tokens
            - "template": Use a template prompt

        video_pruning_mode: Video pruning mode (reserved for future use).
            - "none": No special video handling
            - "frame": Per-frame pruning
            - "window": Sliding window pruning

        debug: Enable debug logging.

    Example:
        >>> config = VTRConfig(enabled=True, strategy="fastv", keep_ratio=0.5)
        >>> config.enabled
        True
        >>> config.keep_ratio
        0.5
    """

    # Basic switches
    enabled: bool = False
    strategy: str = "priortr"

    # Pruning layer configuration
    prune_layer: Union[int, List[int]] = 3

    # Token retention strategy (priority: keep_tokens > score_threshold > keep_ratio)
    keep_ratio: Union[float, List[float]] = 0.1111
    keep_tokens: Optional[Union[int, List[int]]] = None
    score_threshold: Optional[Union[float, List[float]]] = None

    # Token merge (SparseVLM)
    token_merge: bool = False
    merge_clusters: Union[int, List[int]] = 10

    # VisPruner specific
    important_ratio: float = 0.5  # Fraction of kept tokens selected by importance (rest by diversity)

    # Attention aggregation methods
    # "auto" resolves per strategy: "question" for priortr/infovtr, "last" for fastv/others
    query_aggregation: str = "auto"
    head_aggregation: str = "mean"

    # InfoVTR specific
    prior_prompt: str = ""
    prior_mode: str = "truncate"

    # Video support (reserved for future)
    video_pruning_mode: str = "none"

    # Debug mode
    debug: bool = False

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If configuration values are invalid.
        """
        # Validate strategy
        valid_strategies = {"fastv", "infovtr", "sparsevlm", "priortr", "vispruner"}
        if self.strategy not in valid_strategies:
            raise ValueError(
                f"Invalid strategy '{self.strategy}'. "
                f"Must be one of: {valid_strategies}"
            )

        # Resolve "auto" query_aggregation based on strategy
        if self.query_aggregation == "auto":
            if self.strategy in ("priortr", "infovtr"):
                self.query_aggregation = "question"
            else:
                self.query_aggregation = "last"

        # VisPruner: force prune_layer=1 (pre-LLM pruning at layer 0)
        if self.strategy == "vispruner" and self.prune_layer != 1:
            logger.info("VisPruner: overriding prune_layer to 1 (pre-LLM pruning)")
            self.prune_layer = 1

        # Validate important_ratio
        if not 0.0 < self.important_ratio <= 1.0:
            raise ValueError(
                f"important_ratio must be in (0.0, 1.0], got {self.important_ratio}"
            )

        # Validate keep_ratio
        if isinstance(self.keep_ratio, list):
            for r in self.keep_ratio:
                if not 0.0 < r <= 1.0:
                    raise ValueError(
                        f"keep_ratio elements must be in (0.0, 1.0], got {r}"
                    )
        else:
            if not 0.0 < self.keep_ratio <= 1.0:
                raise ValueError(
                    f"keep_ratio must be in (0.0, 1.0], got {self.keep_ratio}"
                )

        # Validate keep_tokens if provided
        if self.keep_tokens is not None:
            if isinstance(self.keep_tokens, list):
                for kt in self.keep_tokens:
                    if kt < 0:
                        raise ValueError(
                            f"keep_tokens elements must be non-negative, got {kt}"
                        )
            else:
                if self.keep_tokens < 0:
                    raise ValueError(
                        f"keep_tokens must be non-negative, got {self.keep_tokens}"
                    )

        # Validate prune_layer (must come before list-length checks)
        if isinstance(self.prune_layer, int):
            if self.prune_layer < 0:
                raise ValueError(
                    f"prune_layer must be non-negative, got {self.prune_layer}"
                )
        elif isinstance(self.prune_layer, list):
            if len(self.prune_layer) == 0:
                raise ValueError("prune_layer list cannot be empty")
            if not all(isinstance(l, int) and l >= 0 for l in self.prune_layer):
                raise ValueError(
                    f"All prune_layer values must be non-negative integers, "
                    f"got {self.prune_layer}"
                )

        # Validate list lengths match prune_layer count
        num_layers = len(self.get_prune_layers())
        if isinstance(self.keep_ratio, list) and len(self.keep_ratio) != num_layers:
            raise ValueError(
                f"keep_ratio length ({len(self.keep_ratio)}) must equal "
                f"prune_layer count ({num_layers})"
            )
        if isinstance(self.keep_tokens, list) and len(self.keep_tokens) != num_layers:
            raise ValueError(
                f"keep_tokens length ({len(self.keep_tokens)}) must equal "
                f"prune_layer count ({num_layers})"
            )
        if isinstance(self.score_threshold, list) and len(self.score_threshold) != num_layers:
            raise ValueError(
                f"score_threshold length ({len(self.score_threshold)}) must equal "
                f"prune_layer count ({num_layers})"
            )
        if isinstance(self.merge_clusters, list) and len(self.merge_clusters) != num_layers:
            raise ValueError(
                f"merge_clusters length ({len(self.merge_clusters)}) must equal "
                f"prune_layer count ({num_layers})"
            )

        # Validate aggregation methods ("auto" is already resolved above)
        valid_query_agg = {"last", "question"}
        if self.query_aggregation not in valid_query_agg:
            raise ValueError(
                f"Invalid query_aggregation '{self.query_aggregation}'. "
                f"Must be one of: {valid_query_agg}"
            )

        valid_head_agg = {"mean", "max"}
        if self.head_aggregation not in valid_head_agg:
            raise ValueError(
                f"Invalid head_aggregation '{self.head_aggregation}'. "
                f"Must be one of: {valid_head_agg}"
            )

        # Validate prior_mode
        valid_prior_modes = {"truncate", "template"}
        if self.prior_mode not in valid_prior_modes:
            raise ValueError(
                f"Invalid prior_mode '{self.prior_mode}'. "
                f"Must be one of: {valid_prior_modes}"
            )

        # Validate video_pruning_mode
        valid_video_modes = {"none", "frame", "window"}
        if self.video_pruning_mode not in valid_video_modes:
            raise ValueError(
                f"Invalid video_pruning_mode '{self.video_pruning_mode}'. "
                f"Must be one of: {valid_video_modes}"
            )

    def get_prune_layers(self) -> List[int]:
        """Get pruning layers as a list.

        Returns:
            List of layer indices where pruning should occur.
        """
        if isinstance(self.prune_layer, int):
            return [self.prune_layer]
        return self.prune_layer

    def is_prune_layer(self, layer_idx: int) -> bool:
        """Check if the given layer index is a pruning layer.

        Args:
            layer_idx: The layer index to check.

        Returns:
            True if pruning should occur at this layer.
        """
        return layer_idx in self.get_prune_layers()

    def get_score_layer(self) -> int:
        """Get the layer to extract attention scores from.

        For pruning at layer K, we extract attention from layer K-1.

        Returns:
            The layer index to extract attention scores from.
        """
        prune_layers = self.get_prune_layers()
        # Return the layer before the first pruning layer
        return prune_layers[0] - 1 if prune_layers[0] > 0 else 0

    def get_keep_count(self, num_tokens: int, layer_idx: int = 0) -> int:
        """Get number of tokens to keep at a specific layer.

        Handles both scalar and per-layer (List) config values.
        Priority: keep_tokens > score_threshold > keep_ratio.

        Args:
            num_tokens: Total number of image tokens available.
            layer_idx: Index into the per-layer list (0-based).

        Returns:
            Number of tokens to keep.
        """
        # Resolve keep_tokens for this layer
        if self.keep_tokens is not None:
            if isinstance(self.keep_tokens, list):
                return min(self.keep_tokens[layer_idx], num_tokens)
            return min(self.keep_tokens, num_tokens)

        # Resolve keep_ratio for this layer
        ratio = self.keep_ratio
        if isinstance(ratio, list):
            ratio = ratio[layer_idx]
        target = max(1, int(num_tokens * ratio))

        # When token_merge is enabled, reduce keep count so that
        # kept + dynamic_cluster_count = target.
        # Dynamic formula (SparseVLMs): clusters = int(int(n_pruned*0.3)+1)/10) + 1
        if self.token_merge and self.strategy == "sparsevlm":
            kept = target
            for _ in range(3):  # converges in 1-2 iterations
                n_pruned = num_tokens - kept
                if n_pruned <= 0:
                    break
                top30 = int(n_pruned * 0.3) + 1
                mc = int(top30 / 10) + 1
                kept = max(1, target - mc)
            return kept

        return target

    def get_merge_cluster_count(self, layer_idx: int = 0) -> int:
        """Get number of merge clusters at a specific layer.

        Args:
            layer_idx: Index into the per-layer list (0-based).

        Returns:
            Number of merge clusters.
        """
        if isinstance(self.merge_clusters, list):
            return self.merge_clusters[layer_idx]
        return self.merge_clusters
