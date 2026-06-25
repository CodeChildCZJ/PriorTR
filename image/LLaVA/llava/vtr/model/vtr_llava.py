# VTR LLaVA Base Model
# Provides the VTR base framework; subclasses implement the _prepare_vtr method
from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union, Dict, Any, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from llava.model.language_model.llava_llama import (
    LlavaLlamaForCausalLM,
    LlavaConfig,
    LlavaLlamaModel,
)
from llava.model.llava_arch import LlavaMetaModel
from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import get_anyres_image_grid_shape

if TYPE_CHECKING:
    from ..config import VTRConfig

from ..config import VTRConfig
from .prunable_llama import PrunableLlamaModel

logger = logging.getLogger(__name__)


def compute_image_token_count(images, image_sizes, config):
    """
    Compute the number of image tokens produced by LLaVA's multimodal processing.
    Pure math — no neural network forward needed.

    Args:
        images: Image tensor. ndim==4 for 1.5, ndim==5 for 1.6.
        image_sizes: List of (width, height) tuples. Required for 1.6.
        config: Model config with mm_patch_merge_type, image_grid_pinpoints, etc.
                Can be None for 1.5 (ndim==4).
    Returns:
        int: Number of image tokens in the final input_embeds.
    """
    # LLaVA 1.5: fixed 576
    if images.ndim == 4:
        return 576

    # LLaVA 1.6 (ndim == 5): [batch, tiles, C, H, W]
    num_tiles = images.shape[1]
    mm_patch_merge_type = getattr(config, 'mm_patch_merge_type', 'flat')

    if mm_patch_merge_type == 'flat':
        return num_tiles * 576

    # spatial / spatial_unpad path
    if num_tiles == 1:
        if 'unpad' in mm_patch_merge_type:
            return 576 + 1
        return 576

    # Multi-tile with spatial_unpad
    if 'unpad' not in mm_patch_merge_type:
        return num_tiles * 576

    # spatial_unpad: compute exact count from image geometry
    patches_per_side = getattr(config, 'num_patches_per_side', 24)
    clip_image_size = getattr(config, 'image_size', 336)
    grid_pinpoints = config.image_grid_pinpoints
    image_size = image_sizes[0]  # (width, height) — batch_size=1

    num_patch_width, num_patch_height = get_anyres_image_grid_shape(
        image_size, grid_pinpoints, clip_image_size
    )

    feature_h = patches_per_side * num_patch_height
    feature_w = patches_per_side * num_patch_width
    orig_w, orig_h = image_size

    orig_aspect = orig_w / orig_h
    curr_aspect = feature_w / feature_h

    if orig_aspect > curr_aspect:
        scale = feature_w / orig_w
        new_h = int(orig_h * scale)
        padding = (feature_h - new_h) // 2
        unpadded_h = feature_h - 2 * padding
        unpadded_w = feature_w
    else:
        scale = feature_h / orig_h
        new_w = int(orig_w * scale)
        padding = (feature_w - new_w) // 2
        unpadded_h = feature_h
        unpadded_w = feature_w - 2 * padding

    return 576 + unpadded_h * (unpadded_w + 1)


class PrunableLlavaLlamaModel(LlavaMetaModel, PrunableLlamaModel):
    """
    Prunable version of LlavaLlamaModel.

    Inherits LlavaMetaModel (vision encoder) and PrunableLlamaModel (prunable language model).
    """
    config_class = LlavaConfig

    def __init__(self, config):
        super(PrunableLlavaLlamaModel, self).__init__(config)


class VTRLlavaForCausalLM(LlavaLlamaForCausalLM):
    """
    VTR LLaVA base class.

    Provides:
    - setup_vtr(): configure VTR settings
    - _compute_image_token_range(): compute image token range (shared by all subclasses)
    - _prepare_vtr(): VTR preprocessing (implemented by subclasses)
    - generate(): unified generation entry point
    """
    config_class = LlavaConfig

    def __init__(self, config):
        # Call LlamaForCausalLM's parent init (skip LlavaLlamaForCausalLM's model creation)
        super(LlavaLlamaForCausalLM, self).__init__(config)
        
        # Use the prunable version of the model
        self.model = PrunableLlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # VTR config
        self._vtr_config: Optional[VTRConfig] = None
        
        self.post_init()

    def _is_llava_16(self) -> bool:
        """Detect whether the current model is LLaVA-1.6 (based on mm_patch_merge_type)."""
        merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
        return 'spatial' in merge_type or 'unpad' in merge_type

    def setup_vtr(self, config: VTRConfig):
        """
        Set up VTR configuration.

        Auto-fills unset hyperparameters based on model version:
        - LLaVA-1.5: keep_tokens=192, query_aggregation="question"
        - LLaVA-1.6: keep_tokens=320, query_aggregation="last"

        Explicitly set values are not overridden.
        """
        is_16 = self._is_llava_16()

        # CLSE only supports LLaVA-1.5 (fixed 576-token 24x24 grid). LLaVA-1.6 / LLaVA-NeXT
        # (anyres, variable token count) has no regular grid for the 2D-FFT spectral term,
        # so reject it with a clear error instead of crashing later in the FFT reshape.
        if config.enabled and config.strategy == "clse" and is_16:
            raise ValueError(
                "CLSE is only supported on LLaVA-1.5 (576-token 24x24 grid); "
                "LLaVA-1.6 / LLaVA-NeXT (anyres) is not supported by CLSE. "
                "Use strategy='priortr' or 'fastv' on LLaVA-1.6."
            )

        # Don't auto-fill keep_tokens when CLSE was given an explicit retain_ratio, so its
        # ratio schedule actually takes effect (otherwise _schedule reads keep_tokens first
        # and the retain_ratio is silently ignored).
        if config.keep_tokens is None and not (
            config.strategy == "clse" and getattr(config, "retain_ratio", None) is not None
        ):
            config.keep_tokens = 320 if is_16 else 192
        if config.query_aggregation is None:
            config.query_aggregation = "last" if is_16 else "question"

        logger.info(f"VTR setup (LLaVA-{'1.6' if is_16 else '1.5'}): "
                     f"keep_tokens={config.keep_tokens}, query_aggregation={config.query_aggregation}")

        self._vtr_config = config
        self.model.setup_vtr(config)

    def _compute_image_token_range(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor,
        image_sizes=None,
    ) -> Optional[Tuple[int, int]]:
        """
        Compute image token range in the sequence.
        Works for both LLaVA 1.5 (fixed 576) and 1.6 (dynamic, from image geometry).
        """
        if not torch.is_tensor(input_ids):
            raise TypeError(f"Expected input_ids to be Tensor, got {type(input_ids)}")
        if images is None:
            return None
        if not torch.is_tensor(images):
            raise TypeError(f"Expected images to be a torch.Tensor, but got {type(images)}.")

        image_token_mask = (input_ids == IMAGE_TOKEN_INDEX)
        if not image_token_mask.any():
            return None

        img_token_pos = image_token_mask.nonzero(as_tuple=True)[1][0].item()

        # Build a lightweight config object for compute_image_token_count
        vision_tower = self.get_vision_tower()
        cfg = type('C', (), {
            'mm_patch_merge_type': getattr(self.config, 'mm_patch_merge_type', 'flat'),
            'image_aspect_ratio': getattr(self.config, 'image_aspect_ratio', 'square'),
            'image_grid_pinpoints': getattr(self.config, 'image_grid_pinpoints', None),
            'image_size': getattr(vision_tower.config, 'image_size', 336) if vision_tower else 336,
            'num_patches_per_side': getattr(vision_tower, 'num_patches_per_side', 24) if vision_tower else 24,
        })()

        num_img_tokens = compute_image_token_count(images, image_sizes, cfg)
        return (img_token_pos, img_token_pos + num_img_tokens)

    def _prepare_vtr(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor,
        image_token_range: Tuple[int, int],
        **kwargs,
    ) -> dict:
        """
        VTR preprocessing (implemented by subclasses).

        Called before generate to prepare extra information needed by VTR.

        Args:
            input_ids: Input token IDs
            images: Image tensor
            image_token_range: Image token range
            **kwargs: Additional arguments

        Returns:
            vtr_ctx: Extra context passed to forward (e.g., prior_attention)
        """
        # Base class returns empty ctx by default; subclasses override as needed
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
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        # VTR extra parameters
        image_token_range: Optional[Tuple[int, int]] = None,
        vtr_ctx: Optional[dict] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward with VTR support.
        """
        # Store image_token_range for downstream propagation
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
                images,
                image_sizes
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

        This allows PrunableLlamaModel.forward to access these parameters during super().generate().
        """
        self.model._vtr_image_token_range = image_token_range
        self.model._vtr_ctx = vtr_ctx
    
    def _clean_vtr_state(self):
        """
        [State Cleanup] Clean up temporarily injected VTR state.

        Prevents state leakage into subsequent calls.
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
        [Improved Override] Prepare inputs for the generation phase.
        Fixes:
        1. Dynamically handle curr_query_len (resolves 192 vs 193 conflict)
        2. Fix position_ids slicing and dimensions (avoid Llama internal broadcast errors)
        3. Fix super() call to ensure multimodal processing logic is not skipped
        """
        
        # ======================================================
        # [Early Exit] VTR Disabled: use standard flow
        # ======================================================
        if self._vtr_config is None or not self._vtr_config.enabled:
            # Note: use super() directly, do not skip the current class's parent
            return super().prepare_inputs_for_generation(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs
            )
        
        # ======================================================
        # Decode phase
        # ======================================================
        if past_key_values is not None:
            # 1. Get physical length (memory footprint)
            if hasattr(past_key_values, "get_seq_length"):
                cache_length = past_key_values.get_seq_length() # physical length
                past_length = past_key_values.seen_tokens      # logical length
            elif isinstance(past_key_values, tuple):
                cache_length = past_key_values[0][0].shape[2]
                raise TypeError("VTR mode does not support past_key_values type tuple. Please ensure DynamicCache is used.")
            else:
                cache_length = 0
            

            # Regardless of how many tokens HF resends, slice out the truly new ones based on past_length
            past_length = getattr(past_key_values, "seen_tokens", cache_length)
            # Debug info
            # print(f"DEBUG: Physical={cache_length}, Logical={past_length}, past_key_values={past_key_values.seen_tokens}")
            input_ids = input_ids[:, -1:]

            # 2. Get current input query length (may be 1 or 2)
            curr_query_len = input_ids.shape[1]

            # 3. Position IDs: use "logical length" (semantic positions)

            # Fix: dynamically generate or slice position_ids based on curr_query_len
            if attention_mask is not None and attention_mask.shape[1] > 1:
                # Option A: extract from the existing long mask
                full_position_ids = attention_mask.long().cumsum(-1) - 1
                full_position_ids.masked_fill_(attention_mask == 0, 1)
                # Important: slice the last N positions; do not unsqueeze(-1), Llama expects [bs, seq_len]
                position_ids = full_position_ids[:, -curr_query_len:]
            else:
                # Option B: generate via arange based on logical length (more robust)
                position_ids = torch.arange(
                    past_length, past_length + curr_query_len, 
                    dtype=torch.long, device=input_ids.device
                ).unsqueeze(0)
            
            # 4. Attention Mask: align with "physical length"
            # Length = physical cache + current input (1 or 2)
            attention_mask = torch.ones(
                (input_ids.shape[0], cache_length + curr_query_len),
                dtype=torch.long,
                device=input_ids.device
            )


            # past_key_values.seen_tokens += 1
            
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
            image_sizes = kwargs.get("image_sizes", None)
            
            # 2. Call parent class (LlavaLlamaForCausalLM)
            # Note: use super() to ensure Llava's prepare_inputs_labels_for_multimodal runs first
            model_inputs = super().prepare_inputs_for_generation(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs
            )
            
            # 3. Restore multimodal and VTR state
            if images is not None:
                model_inputs["images"] = images
            if image_sizes is not None:
                model_inputs["image_sizes"] = image_sizes
            
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
        [Refactored Entry Point] Uses State Injection + standard generate.

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
            image_token_range = self._compute_image_token_range(inputs, images, image_sizes)
        
        # 2. Call _prepare_vtr (subclass implementation)
        vtr_ctx = {}
        if image_token_range is not None:
            vtr_ctx = self._prepare_vtr(inputs, images, image_token_range, image_sizes=image_sizes, **kwargs)
        
        # 3. State Injection
        self._register_vtr_state(image_token_range, vtr_ctx)
        
        try:
            # 4. Call standard generate (internally calls prepare_inputs_for_generation and forward)
            # Pass images and image_sizes via kwargs
            return super().generate(
                inputs,
                images=images,
                image_sizes=image_sizes,
                **kwargs
            )
        finally:
            # 5. Clean up state (regardless of success or failure)
            self._clean_vtr_state()


# Note: do not re-register LlavaConfig; it is already registered in llava_llama.py
# If independent registration is needed, create a new Config class

