"""Qwen2-VL ForConditionalGeneration with Visual Token Reduction (VTR).

Mirrors the Qwen3-VL ``VTRQwen3VLForConditionalGeneration``: it loads a stock
Qwen2-VL model and, when VTR is enabled, swaps the text decoder's class to
``PrunableQwen2VLTextModel`` (in-place, no weight copy) and attaches the chosen
strategy. The visual position mask and ``image_grid_thw`` are fed to the text
model per-forward by the underlying ForConditionalGeneration forward.
"""

import logging
from typing import Optional

from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLForConditionalGeneration,
)

from ..config import VTRConfig
from ..strategy.base import VTRStrategy
from .prunable_qwen2_vl import PrunableQwen2VLTextModel

logger = logging.getLogger(__name__)


class VTRQwen2VLForConditionalGeneration(Qwen2VLForConditionalGeneration):
    """Qwen2-VL model with Visual Token Reduction support.

    Strategies: ``fastv``, ``priortr``, ``clse``. When ``vtr_config.enabled`` is
    False, behaviour is identical to the base model.
    """

    def __init__(self, config, vtr_config: Optional[VTRConfig] = None):
        super().__init__(config)
        self.vtr_config = vtr_config or VTRConfig()
        self.vtr_strategy = self._create_strategy(self.vtr_config.strategy)
        if self.vtr_config.enabled:
            self._replace_text_model()
        logger.info(
            f"VTRQwen2VLForConditionalGeneration: enabled={self.vtr_config.enabled}, "
            f"strategy={self.vtr_config.strategy}, keep_ratio={self.vtr_config.keep_ratio}"
        )

    def _create_strategy(self, strategy_name: str) -> VTRStrategy:
        if strategy_name == "fastv":
            from ..strategy.fastv import FastVStrategy
            return FastVStrategy()
        elif strategy_name == "priortr":
            from ..strategy.priortr import PriorTRStrategy
            return PriorTRStrategy()
        elif strategy_name == "clse":
            from ..strategy.clse import CLSEStrategy
            return CLSEStrategy()
        else:
            raise ValueError(
                f"Unknown VTR strategy: '{strategy_name}'. "
                f"Must be one of: 'fastv', 'priortr', 'clse'"
            )

    def _replace_text_model(self) -> None:
        """Swap the language model class to PrunableQwen2VLTextModel in-place."""
        lm = self.model.language_model
        lm.__class__ = PrunableQwen2VLTextModel
        lm.setup_vtr(self.vtr_config, self.vtr_strategy)
        logger.info("Swapped language_model class to PrunableQwen2VLTextModel (in-place)")

    @classmethod
    def from_pretrained_vtr(
        cls,
        pretrained_model_name_or_path: str,
        vtr_config: Optional[VTRConfig] = None,
        **kwargs,
    ) -> "VTRQwen2VLForConditionalGeneration":
        """Load a pretrained Qwen2-VL model with VTR configured."""
        model = cls.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model.vtr_config = vtr_config or VTRConfig()
        model.vtr_strategy = model._create_strategy(model.vtr_config.strategy)
        if model.vtr_config.enabled:
            model._replace_text_model()
        logger.info(
            f"Loaded VTR Qwen2-VL from '{pretrained_model_name_or_path}': "
            f"enabled={model.vtr_config.enabled}, strategy={model.vtr_config.strategy}"
        )
        return model
