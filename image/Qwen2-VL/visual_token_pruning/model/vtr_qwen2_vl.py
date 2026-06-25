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

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        rope_deltas=None,
        cache_position=None,
        **kwargs,
    ):
        """Forward with self-contained visual-mask / grid plumbing.

        The prunable text model reads the visual position mask and the image grid
        off itself (``self.visual_pos_masks`` / ``self.image_grid_thw``). We set
        them here, per-forward, so the **stock** transformers
        ``Qwen2VLForConditionalGeneration.forward`` needs no patch. During decode
        ``image_grid_thw`` is ``None`` and the new token is not an image token, so
        no pruning happens (matches the prefill-only design).

        The full stock signature is mirrored on purpose: transformers'
        ``GenerationMixin.prepare_inputs_for_generation`` introspects this
        ``forward`` signature to decide whether to synthesise ``position_ids``.
        A narrowed ``(input_ids, image_grid_thw, **kwargs)`` signature hides
        ``position_ids`` inside ``**kwargs``, so the generic prep skips creating
        it and Qwen2-VL's ``prepare_inputs_for_generation`` then crashes on a
        ``None`` ``position_ids``. Keeping the named parameters avoids that.
        """
        if getattr(self, "vtr_config", None) is not None and self.vtr_config.enabled:
            lm = self.model.language_model
            if input_ids is not None:
                image_token_id = getattr(self.config, "image_token_id", 151655)
                lm.visual_pos_masks = (input_ids == image_token_id)
            lm.image_grid_thw = image_grid_thw
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
            **kwargs,
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
        # Initialise the self-contained plumbing attributes (set per-forward above),
        # so they are always defined even before the first forward.
        lm.visual_pos_masks = None
        lm.image_grid_thw = None
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
