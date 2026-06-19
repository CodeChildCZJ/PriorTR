"""VTR Strategy module.

This module provides visual token reduction strategies for pruning image tokens
in vision-language models.

Available strategies:
    - VTRStrategy: Abstract base class for all strategies
    - FastVStrategy: Attention-based pruning (FastV method)
    - InfoVTRStrategy: V-Information based pruning (InfoVTR method)
    - SparseVLMStrategy: Text-guided pruning (SparseVLM method)
    - PriorTRStrategy: Single-forward V-Information pruning (PriorTR method)
    - VisPrunerStrategy: Visual-cue-based pre-LLM pruning (VisPruner method)

Example:
    >>> from visual_token_pruning.strategy import FastVStrategy, InfoVTRStrategy
    >>> strategy = FastVStrategy()
    >>> infovtr = InfoVTRStrategy()
"""

from .base import VTRStrategy
from .fastv import FastVStrategy
from .infovtr import InfoVTRStrategy
from .priortr import PriorTRStrategy
from .sparsevlm import SparseVLMStrategy
from .vispruner import VisPrunerStrategy

__all__ = [
    "VTRStrategy",
    "FastVStrategy",
    "InfoVTRStrategy",
    "PriorTRStrategy",
    "SparseVLMStrategy",
    "VisPrunerStrategy",
]
