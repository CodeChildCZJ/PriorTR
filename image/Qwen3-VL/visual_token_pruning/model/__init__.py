"""Model module for Visual Token Pruning.

This module provides the PrunableQwen3VLTextModel, which extends the standard
Qwen3-VL text decoder to support visual token pruning at specified layers.

It also provides prior forward utilities for the PriorTR-2F strategy:
    - build_prior_input: Construct prior input IDs
    - extract_prior_attention: Execute prior forward and extract attention

Example:
    >>> from visual_token_pruning.model import PrunableQwen3VLTextModel
    >>> from visual_token_pruning.model import build_prior_input, extract_prior_attention
    >>> from visual_token_pruning import VTRConfig
    >>> from visual_token_pruning.strategy import FastVStrategy
    >>>
    >>> # The model can be used as a drop-in replacement for Qwen3VLTextModel
    >>> config = VTRConfig(enabled=True, strategy="fastv", keep_ratio=0.5)
    >>> strategy = FastVStrategy()
"""

from .deepstack_handler import DeepStackSyncHandler
from .prunable_qwen3_vl import PrunableQwen3VLTextModel
from .prior_utils import (
    build_prior_input,
    compute_prior_image_token_range,
    extract_prior_attention,
)
from .token_merge import cluster_and_merge
from .vtr_qwen3_vl import VTRQwen3VLForConditionalGeneration

__all__ = [
    "DeepStackSyncHandler",
    "PrunableQwen3VLTextModel",
    "VTRQwen3VLForConditionalGeneration",
    "build_prior_input",
    "cluster_and_merge",
    "compute_prior_image_token_range",
    "extract_prior_attention",
]
