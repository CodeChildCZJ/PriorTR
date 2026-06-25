# VTR Configuration Classes
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class VTRConfig:
    """
    Visual Token Reduction base configuration.

    Token retention priority (highest to lowest):
    1. keep_tokens: keep an exact number of tokens
    2. score_threshold: keep tokens with score above threshold (PriorTR2FConfig only)
    3. keep_ratio: keep tokens by ratio
    """
    # Whether to enable VTR
    enabled: bool = False

    # Strategy name (resolved via registry)
    strategy: str = "fastv"

    # Pruning layer configuration
    # - Single int: prune once at the specified layer (backward compatible)
    # - List[int]: prune at multiple layers, each using keep_ratio/keep_tokens
    prune_layer: Union[int, List[int]] = 16

    # Keep ratio (retain Top-K% of visual tokens)
    # Ignored when keep_tokens is set
    keep_ratio: float = 0.25

    # Exact number of tokens to keep
    # Takes priority over keep_ratio when set
    # None means use keep_ratio
    keep_tokens: Optional[int] = None

    # Attention aggregation method
    query_aggregation: str = "last"    # last / question
    head_aggregation: str = "mean"     # mean / max

    # ---------------------------------------------------------------------
    # [CLSE] Cross-Layer Spectral Evolution options (drop-in strategy="clse").
    # No-op for every other strategy: ref_layers stays empty so the spectral
    # snapshot hook never fires, and the spectral knobs are simply unused.
    # ---------------------------------------------------------------------
    # Layer(s) at which to snapshot the reference image features z_L for the
    # cross-layer spectral-evolution score (CLSE L_list; video default [2]).
    ref_layers: List[int] = field(default_factory=list)

    # Spectral hyper-parameters (defaults match the CLSE reference).
    clse_cutoff_ratio: float = 0.1     # Gaussian 2D-FFT high-pass cutoff ratio
    clse_temp: float = 0.1             # evolution-factor sigmoid temperature

    # Visual-token grid (T, H, W). Video-LLaVA emits 8 x 16 x 16 = 2048 tokens.
    # The per-frame 2D FFT only runs when T*H*W equals the visual-token count.
    clse_grid_t: int = 8
    clse_grid_h: int = 16
    clse_grid_w: int = 16

    # Use a single 3D FFT instead of the per-frame 2D FFT. The CLSE reference
    # ships both but defaults to per-frame (sparse 8-frame video makes the
    # temporal axis alias under a 3D FFT); kept here for parity / ablation.
    clse_fft_3d: bool = False

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
            "clse_cutoff_ratio": self.clse_cutoff_ratio,
            "clse_temp": self.clse_temp,
            "clse_grid_t": self.clse_grid_t,
            "clse_grid_h": self.clse_grid_h,
            "clse_grid_w": self.clse_grid_w,
            "clse_fft_3d": self.clse_fft_3d,
        }

    @classmethod
    def from_dict(cls, d: dict) -> VTRConfig:
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PriorTR2FConfig(VTRConfig):
    """
    PriorTR-2F-specific configuration.

    Inherits VTRConfig and adds PriorTR-2F-specific options.

    Token retention priority (highest to lowest):
    1. keep_tokens: keep an exact number of tokens
    2. score_threshold: keep tokens with score above threshold
    3. keep_ratio: keep tokens by ratio
    """
    # Override default strategy name
    strategy: str = "priortr_2f"

    # Prior prompt (empty string means empty prompt)
    prior_prompt: str = ""

    # Whether to use adaptive layer selection (Pipeline B)
    adaptive_layer: bool = False

    # Candidate layers for adaptive layer selection
    candidate_layers: List[int] = field(
        default_factory=lambda: [4, 8, 12, 16, 20, 24, 28]
    )

    # Threshold mode: when not None, keep tokens with score > threshold
    # Priority: lower than keep_tokens, higher than keep_ratio
    score_threshold: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        super().__post_init__()

        # Validate score_threshold
        if self.score_threshold is not None and self.score_threshold < 0:
            raise ValueError(f"score_threshold must be non-negative, got {self.score_threshold}")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = super().to_dict()
        d.update({
            "prior_prompt": self.prior_prompt,
            "adaptive_layer": self.adaptive_layer,
            "candidate_layers": self.candidate_layers,
            "score_threshold": self.score_threshold,
        })
        return d

    @classmethod
    def from_dict(cls, d: dict) -> PriorTR2FConfig:
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
