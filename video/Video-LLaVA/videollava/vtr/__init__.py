# Visual Token Reduction (VTR) Framework
# Prunes visual tokens during LLM inference.

from .config import VTRConfig, InfoVTRConfig
from .strategy import (
    PruningStrategy,
    VTR_REGISTRY,
    register_strategy,
    get_strategy,
    FastVStrategy,
    InfoVTRStrategy,
)
from .model import (
    PrunableLlamaModel,
    VTRLlavaForCausalLM,
    FastVLlava,
    InfoVTRBaseLlava,
    FixedLayerInfoVTR,
    AdaptiveLayerInfoVTR,
)

__all__ = [
    # Config
    "VTRConfig",
    "InfoVTRConfig",
    # Strategy
    "PruningStrategy",
    "VTR_REGISTRY",
    "register_strategy",
    "get_strategy",
    "FastVStrategy",
    "InfoVTRStrategy",
    # Model
    "PrunableLlamaModel",
    "VTRLlavaForCausalLM",
    "FastVLlava",
    "InfoVTRBaseLlava",
    "FixedLayerInfoVTR",
    "AdaptiveLayerInfoVTR",
]
