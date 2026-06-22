"""VTR Strategy module.

This module provides visual token reduction strategies for pruning image tokens
in vision-language models.

Available strategies:
    - VTRStrategy: Abstract base class for all strategies
    - FastVStrategy: Attention-based pruning (FastV method)
    - PriorTR2FStrategy: V-Information based pruning (PriorTR-2F method)
    - SparseVLMStrategy: Text-guided pruning (SparseVLM method)
    - PriorTRStrategy: Single-forward V-Information pruning (PriorTR method)
    - VisPrunerStrategy: Visual-cue-based pre-LLM pruning (VisPruner method)

Example:
    >>> from visual_token_pruning.strategy import FastVStrategy, PriorTR2FStrategy
    >>> strategy = FastVStrategy()
    >>> priortr_2f = PriorTR2FStrategy()
"""

from .base import VTRStrategy
from .fastv import FastVStrategy
from .priortr_2f import PriorTR2FStrategy
from .priortr import PriorTRStrategy
from .sparsevlm import SparseVLMStrategy
from .vispruner import VisPrunerStrategy
from .clse import CLSEStrategy

__all__ = [
    "VTRStrategy",
    "FastVStrategy",
    "PriorTR2FStrategy",
    "PriorTRStrategy",
    "SparseVLMStrategy",
    "VisPrunerStrategy",
    "CLSEStrategy",
]
