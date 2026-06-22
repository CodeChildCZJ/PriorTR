# VTR Configuration Classes
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class VTRConfig:
    """
    Visual Token Reduction configuration (PriorTR).

    Token retention priority (highest to lowest):
    1. keep_tokens: keep an exact number of tokens
    2. keep_ratio: keep tokens by ratio
    """
    # Whether to enable VTR
    enabled: bool = False

    # Strategy name
    strategy: str = "priortr"
    
    # Pruning layer configuration
    # - Single int: prune once at the specified layer (backward compatible)
    # - List[int]: prune at multiple layers, each using keep_ratio/keep_tokens
    prune_layer: Union[int, List[int]] = 3
    
    # Keep ratio (retain Top-K% of visual tokens)
    # Ignored when keep_tokens is set
    keep_ratio: float = 0.25
    
    # Exact number of tokens to keep
    # Takes priority over keep_ratio when set
    # None means auto (set by setup_vtr based on model version: 1.5→192, 1.6→320)
    keep_tokens: Optional[int] = None

    # Attention aggregation method
    # query_aggregation: None means auto (set by setup_vtr based on model version: 1.5→question, 1.6→last)
    query_aggregation: Optional[str] = None    # None(auto) / last / question
    head_aggregation: str = "mean"              # mean / max

    # [CLSE] Layers at which to snapshot reference image features into vtr_ctx["z_ref"]
    # for cross-layer (spectral-evolution) scoring. Empty = no snapshot (default; other
    # strategies are unaffected).
    ref_layers: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        # Convert single prune_layer to list for unified handling
        if isinstance(self.prune_layer, int):
            self._prune_layers: List[int] = [self.prune_layer]
        else:
            self._prune_layers = sorted(self.prune_layer)
        
        # Validate keep_tokens
        if self.keep_tokens is not None and self.keep_tokens < 0:
            raise ValueError(f"keep_tokens must be non-negative, got {self.keep_tokens}")
        
        # Validate keep_ratio
        if not 0.0 <= self.keep_ratio <= 1.0:
            raise ValueError(f"keep_ratio must be in [0, 1], got {self.keep_ratio}")
    
    @property
    def prune_layers(self) -> List[int]:
        """Get the list of pruning layers (unified interface)."""
        return self._prune_layers
    
    @property
    def num_prune_layers(self) -> int:
        """Get the number of pruning layers."""
        return len(self._prune_layers)
    
    def get_keep_count(self, num_tokens: int) -> int:
        """
        Compute the number of tokens to keep.

        Args:
            num_tokens: Total number of image tokens

        Returns:
            Number of tokens to keep (can be 0, meaning remove all)
        """
        if self.keep_tokens is not None:
            # Exact count mode, but cannot exceed total
            return min(self.keep_tokens, num_tokens)
        else:
            # Ratio mode, allows 0 (removes all when keep_ratio=0)
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
            "ref_layers": self.ref_layers,
        }

    @classmethod
    def from_dict(cls, d: dict) -> VTRConfig:
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

