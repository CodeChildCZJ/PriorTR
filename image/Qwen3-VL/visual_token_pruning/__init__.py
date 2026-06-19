"""Visual Token Pruning module for Qwen3-VL.

This module provides visual token reduction (VTR) functionality to reduce
computational cost by pruning less important image tokens during inference.

Main components:
    - VTRConfig: Configuration dataclass for VTR settings
    - VTRStrategy: Abstract base class for pruning strategies
    - FastVStrategy: Attention-based pruning strategy
    - PrunableQwen3VLTextModel: Qwen3-VL text model with pruning support

Example:
    >>> from visual_token_pruning import VTRConfig
    >>> from visual_token_pruning.strategy import FastVStrategy
    >>> from visual_token_pruning.model import PrunableQwen3VLTextModel
    >>>
    >>> config = VTRConfig(enabled=True, strategy="fastv", keep_ratio=0.5)
    >>> strategy = FastVStrategy()
"""

from .config import VTRConfig
from .model import PrunableQwen3VLTextModel, VTRQwen3VLForConditionalGeneration

__all__ = ["VTRConfig", "PrunableQwen3VLTextModel", "VTRQwen3VLForConditionalGeneration"]
