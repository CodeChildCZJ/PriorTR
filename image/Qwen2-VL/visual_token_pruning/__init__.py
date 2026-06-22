"""Visual Token Pruning module for Qwen2-VL.

Reduces compute by pruning less-important image tokens during inference via a
pluggable strategy (fastv / priortr / clse).

Example:
    >>> from visual_token_pruning import VTRConfig
    >>> from visual_token_pruning.model import VTRQwen2VLForConditionalGeneration
    >>> config = VTRConfig(enabled=True, strategy="clse",
    ...                    prune_layer=[1, 10, 19], keep_ratio=[0.57, 0.36, 0.098])
"""

from .config import VTRConfig
from .model import PrunableQwen2VLTextModel, VTRQwen2VLForConditionalGeneration

__all__ = ["VTRConfig", "PrunableQwen2VLTextModel", "VTRQwen2VLForConditionalGeneration"]
