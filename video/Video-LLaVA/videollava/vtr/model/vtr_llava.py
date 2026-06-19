# VTR LLaVA Base Model
# Provides VTR base framework; subclasses implement _prepare_vtr method.
from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union, Dict, Any, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from videollava.model.language_model.llava_llama import (
    LlavaLlamaForCausalLM,
    LlavaConfig,
    LlavaLlamaModel,
)
from videollava.model.llava_arch import LlavaMetaModel
from videollava.constants import IMAGE_TOKEN_INDEX

if TYPE_CHECKING:
    from ..config import VTRConfig

from ..config import VTRConfig
from .prunable_llama import PrunableLlamaModel

logger = logging.getLogger(__name__)


class PrunableLlavaLlamaModel(LlavaMetaModel, PrunableLlamaModel):
    """
    Prunable version of LlavaLlamaModel.

    Inherits LlavaMetaModel (vision encoder) and PrunableLlamaModel (prunable LM).
    """
    config_class = LlavaConfig

    def __init__(self, config):
        super(PrunableLlavaLlamaModel, self).__init__(config)


class VTRLlavaForCausalLM(LlavaLlamaForCausalLM):
    """
    VTR LLaVA base class.

    Provides:
    - setup_vtr(): configure VTR
    - _compute_image_token_range(): compute image token range (shared by all subclasses)
    - _prepare_vtr(): VTR preprocessing (implemented by subclasses)
    - generate(): unified generation entry point
    """
    config_class = LlavaConfig

    def __init__(self, config):
        # Call LlamaForCausalLM's parent init (skip LlavaLlamaForCausalLM's model creation)
        super(LlavaLlamaForCausalLM, self).__init__(config)

        # Use prunable version of model
        self.model = PrunableLlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # VTR configuration
        self._vtr_config: Optional[VTRConfig] = None

        self.post_init()

    def setup_vtr(self, config: VTRConfig):
        """
        Set up VTR configuration.

        Also propagates to internal PrunableLlamaModel.
        """
        self._vtr_config = config
        self.model.setup_vtr(config)

    def _compute_image_token_range(
        self,
        input_ids: torch.Tensor,
        images,
    ) -> Optional[Tuple[int, int]]:
        if not torch.is_tensor(input_ids):
            raise TypeError(f"Expected input_ids to be Tensor, got {type(input_ids)}")
        if images is None:
            return None

        # Find position of IMAGE_TOKEN_INDEX in input_ids
        image_token_mask = (input_ids == IMAGE_TOKEN_INDEX)
        if not image_token_mask.any():
            return None
        img_start = image_token_mask.nonzero(as_tuple=True)[1][0].item()

        # Video-LLaVA: images is typically a list (elements are 3D images / 4D videos)
        if torch.is_tensor(images):
            images_list = [images]
        elif isinstance(images, (list, tuple)):
            images_list = list(images)
        else:
            raise TypeError(f"Expected images to be a torch.Tensor or list, but got {type(images)}")

        image_tensors = [x for x in images_list if torch.is_tensor(x) and x.ndim == 3]  # [C,H,W]
        video_tensors = [x for x in images_list if torch.is_tensor(x) and x.ndim == 4]  # [C,T,H,W]

        num_img_tokens = 0
        with torch.no_grad():
            if len(image_tensors) > 0:
                images_minibatch = torch.stack(image_tensors, dim=0).to(self.device)
                image_features = self.encode_images(images_minibatch)  # [B, L, C]
                if image_features.dim() == 3:
                    num_img_tokens += image_features.shape[0] * image_features.shape[1]
                else:
                    num_img_tokens += image_features.shape[0]

            if len(video_tensors) > 0:
                videos_minibatch = torch.stack(video_tensors, dim=0).to(self.device)
                video_features = self.encode_videos(videos_minibatch)  # [B, T, L, C]
                if video_features.dim() == 4:
                    num_img_tokens += video_features.shape[0] * video_features.shape[1] * video_features.shape[2]
                else:
                    num_img_tokens += video_features.shape[0]

        if num_img_tokens <= 0:
            return None
        return (img_start, img_start + num_img_tokens)

    def _prepare_vtr(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor,
        image_token_range: Tuple[int, int],
        **kwargs,
    ) -> dict:
        """
        VTR preprocessing (implemented by subclasses).

        Called before generate to prepare VTR-specific context.

        Args:
            input_ids: input token ids
            images: image tensor
            image_token_range: image token range
            **kwargs: additional arguments

        Returns:
            vtr_ctx: extra context passed to forward (e.g., prior_attention)
        """
        # Base class returns empty ctx; subclasses override as needed
        return {}

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        # VTR extra parameters
        image_token_range: Optional[Tuple[int, int]] = None,
        vtr_ctx: Optional[dict] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """Forward with VTR support."""
        # Record image_token_range for downstream propagation
        self._current_image_token_range = image_token_range
        self._current_vtr_ctx = vtr_ctx

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )
        # Fix attention_mask length mismatch after prepare_inputs_labels_for_multimodal
        if attention_mask is not None and inputs_embeds is not None:
            if attention_mask.shape[1] != inputs_embeds.shape[1]:
                attention_mask = torch.ones(
                    (inputs_embeds.shape[0], inputs_embeds.shape[1]),
                    dtype=attention_mask.dtype,
                    device=inputs_embeds.device,
                )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Call model.forward (passes VTR parameters)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            image_token_range=image_token_range,
            vtr_ctx=vtr_ctx,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _register_vtr_state(self, image_token_range: Optional[Tuple[int, int]], vtr_ctx: dict):
        """
        [State Injection] Temporarily inject VTR state into self.model.

        This allows PrunableLlamaModel.forward to access these parameters
        during super().generate() calls.
        """
        self.model._vtr_image_token_range = image_token_range
        self.model._vtr_ctx = vtr_ctx

    def _clean_vtr_state(self):
        """
        [State Cleanup] Clean up temporarily injected VTR state.

        Prevents state leakage to subsequent calls.
        """
        if hasattr(self.model, "_vtr_image_token_range"):
            delattr(self.model, "_vtr_image_token_range")
        if hasattr(self.model, "_vtr_ctx"):
            delattr(self.model, "_vtr_ctx")

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        """
        Prepare inputs for the generation phase.

        Fixes:
        1. Dynamic curr_query_len handling (resolves 192 vs 193 conflict)
        2. Correct position_ids slicing and dimensions (avoids Llama broadcast errors)
        3. Correct super() call to ensure multimodal processing is not skipped
        """

        # ======================================================
        # [Decode Path] Unified past_key_values handling (even when VTR disabled)
        # ======================================================
        if past_key_values is not None and (self._vtr_config is None or not self._vtr_config.enabled):
            # 1. Get physical length
            if hasattr(past_key_values, "get_seq_length"):
                cache_length = past_key_values.get_seq_length()
                past_length = getattr(past_key_values, "seen_tokens", cache_length)
            elif isinstance(past_key_values, tuple):
                cache_length = past_key_values[0][0].shape[2]
                past_length = cache_length
            else:
                cache_length = 0
                past_length = 0
            input_ids = input_ids[:, -1:]
            curr_query_len = input_ids.shape[1]
            # Use logical length for position_ids to avoid being misled by short attention_mask
            position_ids = torch.arange(
                past_length, past_length + curr_query_len,
                dtype=torch.long, device=input_ids.device
            ).unsqueeze(0)
            attention_mask = torch.ones(
                (input_ids.shape[0], cache_length + curr_query_len),
                dtype=torch.long,
                device=input_ids.device
            )
            return {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache", True),
            }

        # ======================================================
        # [Early Exit] VTR Disabled: use standard flow (Prefill only)
        # ======================================================
        if self._vtr_config is None or not self._vtr_config.enabled:
            return super().prepare_inputs_for_generation(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs
            )

        # ======================================================
        # Decode phase: hybrid strategy
        # ======================================================
        if past_key_values is not None:
            # 1. Get physical length (memory footprint)
            if hasattr(past_key_values, "get_seq_length"):
                cache_length = past_key_values.get_seq_length()  # physical length
                past_length = past_key_values.seen_tokens         # logical length
            elif isinstance(past_key_values, tuple):
                cache_length = past_key_values[0][0].shape[2]
                raise TypeError("VTR mode does not support past_key_values type tuple. Please ensure DynamicCache is used.")
            else:
                cache_length = 0

            past_length = getattr(past_key_values, "seen_tokens", cache_length)
            input_ids = input_ids[:, -1:]

            # 2. Get current input query length
            curr_query_len = input_ids.shape[1]

            # 3. Position IDs: use logical length (semantic position)
            # Key: in VTR mode, use logical length for position_ids
            position_ids = torch.arange(
                past_length, past_length + curr_query_len,
                dtype=torch.long, device=input_ids.device
            ).unsqueeze(0)

            # 4. Attention Mask: align with physical length
            # Length = physical cache + current input
            attention_mask = torch.ones(
                (input_ids.shape[0], cache_length + curr_query_len),
                dtype=torch.long,
                device=input_ids.device
            )

            return {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache", True),
            }

        # ======================================================
        # Prefill phase: pass images + VTR state
        # ======================================================
        else:
            # 1. Get multimodal parameters
            images = kwargs.get("images", None)
            kwargs.pop("image_sizes", None)

            # 2. Call parent (LlavaLlamaForCausalLM)
            model_inputs = super().prepare_inputs_for_generation(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs
            )

            # 3. Add back multimodal and VTR state
            if images is not None:
                model_inputs["images"] = images

            if hasattr(self.model, "_vtr_image_token_range"):
                model_inputs["image_token_range"] = self.model._vtr_image_token_range
            if hasattr(self.model, "_vtr_ctx"):
                model_inputs["vtr_ctx"] = self.model._vtr_ctx

            return model_inputs

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """
        Generation entry point using State Injection + standard generate.

        Flow:
        1. Compute image_token_range
        2. Call _prepare_vtr() to get vtr_ctx
        3. Inject state into self.model
        4. Call super().generate() (handles prefill + decode automatically)
        5. Clean up state
        """
        # 1. Compute image_token_range
        image_token_range = None
        if self._vtr_config is not None and self._vtr_config.enabled and images is not None:
            image_token_range = self._compute_image_token_range(inputs, images)

        # 2. Call _prepare_vtr (subclass implementation)
        vtr_ctx = {}
        if image_token_range is not None:
            vtr_ctx = self._prepare_vtr(inputs, images, image_token_range, **kwargs)

        # 3. State Injection
        self._register_vtr_state(image_token_range, vtr_ctx)

        try:
            # 4. Call standard generate (internally calls prepare_inputs_for_generation and forward)
            return super().generate(
                inputs,
                images=images,
                **kwargs
            )
        finally:
            # 5. Clean up state (regardless of success or failure)
            self._clean_vtr_state()


# Note: Do not re-register LlavaConfig here; it is already registered in llava_llama.py.
# To register independently, create a new Config class.
