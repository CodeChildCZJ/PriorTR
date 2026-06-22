"""VTR Strategy module (Qwen2-VL).

Visual token reduction strategies for pruning image tokens in Qwen2-VL.

Available strategies:
    - VTRStrategy: Abstract base class for all strategies
    - FastVStrategy: Attention-based pruning (FastV method)
    - PriorTRStrategy: Single-forward V-Information pruning (PriorTR method)
    - CLSEStrategy: Cross-Layer Spectral Evolution pruning (CLSE method)
"""

from .base import VTRStrategy
from .fastv import FastVStrategy
from .priortr import PriorTRStrategy
from .clse import CLSEStrategy

__all__ = [
    "VTRStrategy",
    "FastVStrategy",
    "PriorTRStrategy",
    "CLSEStrategy",
]
