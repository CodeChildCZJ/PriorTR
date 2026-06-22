"""Qwen2-VL with Visual Token Reduction (VTR) for lmms-eval.

List-valued VTR parameters use ';' as separator in model_args (since commas
already separate key=value pairs). Examples:
    vtr_prune_layer=1;10;19
    vtr_keep_ratio=0.57;0.36;0.098
"""

import os
import sys
from typing import List, Optional, Union

import torch

from lmms_eval.api.registry import register_model

# Ensure the Qwen2-VL project root (containing visual_token_pruning) is importable.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from visual_token_pruning.config import VTRConfig
from visual_token_pruning.model.vtr_qwen2_vl import (
    VTRQwen2VLForConditionalGeneration,
)

from .qwen2_vl import Qwen2_VL


def _parse_int_or_list(val) -> Union[int, List[int]]:
    s = str(val)
    if ";" in s:
        return [int(x) for x in s.split(";")]
    return int(s)


def _parse_float_or_list(val) -> Union[float, List[float]]:
    s = str(val)
    if ";" in s:
        return [float(x) for x in s.split(";")]
    return float(s)


@register_model("qwen2_vl_vtr")
class Qwen2_VL_VTR(Qwen2_VL):
    """Qwen2-VL with VTR support for lmms-eval benchmarking."""

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2-VL-7B-Instruct",
        # VTR parameters
        vtr_enabled: bool = True,
        vtr_strategy: str = "priortr",
        vtr_keep_ratio: Union[str, float] = "0.5",
        vtr_keep_tokens: Optional[str] = None,
        vtr_prune_layer: Union[str, int] = "3",
        vtr_query_aggregation: str = "auto",
        vtr_head_aggregation: str = "mean",
        # Standard Qwen2_VL parameters
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        use_flash_attention_2: Optional[bool] = False,
        max_length: Optional[int] = 2048,
        max_pixels: int = 602112,
        min_pixels: int = 3136,
        max_num_frames: int = 32,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        # Build VTR config (supports ';'-separated list values)
        config_kwargs = dict(
            enabled=vtr_enabled,
            strategy=vtr_strategy,
            keep_ratio=_parse_float_or_list(vtr_keep_ratio),
            prune_layer=_parse_int_or_list(vtr_prune_layer),
            query_aggregation=vtr_query_aggregation,
            head_aggregation=vtr_head_aggregation,
        )
        if vtr_keep_tokens is not None:
            config_kwargs["keep_tokens"] = _parse_int_or_list(vtr_keep_tokens)
        self._vtr_config = VTRConfig(**config_kwargs)

        # Call grandparent (lmms) init to avoid Qwen2_VL loading the stock model.
        from lmms_eval.api.model import lmms
        lmms.__init__(self)

        from accelerate import Accelerator, DistributedType
        from transformers import AutoProcessor, AutoTokenizer
        from loguru import logger as eval_logger

        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        model_kwargs = {}
        if use_flash_attention_2:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        # Load model with VTR support
        self._model = VTRQwen2VLForConditionalGeneration.from_pretrained_vtr(
            pretrained,
            vtr_config=self._vtr_config,
            torch_dtype="auto",
            device_map=self.device_map,
            **model_kwargs,
        ).eval()

        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames
        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self._config = self.model.config
        self._max_length = max_length
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU]
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

        eval_logger.info(
            f"VTR Config: enabled={self._vtr_config.enabled}, strategy={self._vtr_config.strategy}, "
            f"keep_ratio={self._vtr_config.keep_ratio}, prune_layer={self._vtr_config.prune_layer}"
        )
