# Visual Token Reduction (VTR) Framework
# Prunes visual tokens during LLM inference.

from .config import VTRConfig, PriorTR2FConfig
from .strategy import (
    PruningStrategy,
    VTR_REGISTRY,
    register_strategy,
    get_strategy,
    FastVStrategy,
    PriorTR2FStrategy,
)
from .model import (
    PrunableLlamaModel,
    VTRLlavaForCausalLM,
    FastVLlava,
    PriorTR2FBaseLlava,
    FixedLayerPriorTR2F,
    AdaptiveLayerPriorTR2F,
)

__all__ = [
    # Config
    "VTRConfig",
    "PriorTR2FConfig",
    # Strategy
    "PruningStrategy",
    "VTR_REGISTRY",
    "register_strategy",
    "get_strategy",
    "FastVStrategy",
    "PriorTR2FStrategy",
    # Model
    "PrunableLlamaModel",
    "VTRLlavaForCausalLM",
    "FastVLlava",
    "PriorTR2FBaseLlava",
    "FixedLayerPriorTR2F",
    "AdaptiveLayerPriorTR2F",
]
