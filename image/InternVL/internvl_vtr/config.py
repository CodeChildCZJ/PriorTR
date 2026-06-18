# VTR Configuration Classes
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class VTRConfig:
    """
    Visual Token Reduction (VTR) configuration.

    Token retention priority (highest to lowest):
    1. keep_tokens: keep an exact number of tokens
    2. keep_ratio: keep a fraction of tokens
    """
    # Whether VTR is enabled
    enabled: bool = False

    # Strategy name (resolved via registry)
    strategy: str = "priortr"

    # Pruning layer config
    # - int: prune once at the specified layer (backward compatible)
    # - List[int]: prune at multiple layers
    prune_layer: Union[int, List[int]] = 2

    # Fraction of visual tokens to keep (Top-K%)
    # Ignored when keep_tokens is set
    keep_ratio: float = 0.25

    # Exact number of tokens to keep
    # Takes priority over keep_ratio; None means use keep_ratio
    keep_tokens: Optional[int] = None

    # Attention aggregation modes
    query_aggregation: str = "question"    # question / last
    head_aggregation: str = "mean"     # mean / max

    def __post_init__(self) -> None:
        """Validate config parameters."""
        # Normalize prune_layer to a list for uniform handling
        if isinstance(self.prune_layer, int):
            self._prune_layers: List[int] = [self.prune_layer]
        else:
            self._prune_layers = sorted(self.prune_layer)

        if self.keep_tokens is not None and self.keep_tokens < 0:
            raise ValueError(f"keep_tokens must be non-negative, got {self.keep_tokens}")

        if not 0.0 <= self.keep_ratio <= 1.0:
            raise ValueError(f"keep_ratio must be in [0, 1], got {self.keep_ratio}")

    @property
    def prune_layers(self) -> List[int]:
        """Return the list of pruning layers (uniform interface)."""
        return self._prune_layers

    @property
    def num_prune_layers(self) -> int:
        """Return the number of pruning layers."""
        return len(self._prune_layers)

    def get_keep_count(self, num_tokens: int) -> int:
        """
        Compute the number of tokens to keep.

        Args:
            num_tokens: total number of image tokens

        Returns:
            Number of tokens to keep (can be 0)
        """
        if self.keep_tokens is not None:
            return min(self.keep_tokens, num_tokens)
        else:
            return max(0, int(num_tokens * self.keep_ratio))

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "strategy": self.strategy,
            "prune_layer": self.prune_layer,
            "keep_ratio": self.keep_ratio,
            "keep_tokens": self.keep_tokens,
            "query_aggregation": self.query_aggregation,
            "head_aggregation": self.head_aggregation,
        }

    @classmethod
    def from_dict(cls, d: dict) -> VTRConfig:
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
