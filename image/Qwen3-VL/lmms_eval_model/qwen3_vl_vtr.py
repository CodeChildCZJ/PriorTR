"""Qwen3-VL with Visual Token Reduction (VTR) for lmms-eval.

List-valued VTR parameters use ';' as separator in model_args since
commas are already used to separate key=value pairs. Examples:
    vtr_prune_layer=3;7;16
    vtr_keep_tokens=300;200;110
    vtr_keep_ratio=0.52;0.35;0.19
"""

import sys
from typing import List, Optional, Union

import torch

from lmms_eval.api.registry import register_model

# Ensure VTR module is importable (expects PYTHONPATH to include the Qwen3-VL project root)
import os
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from visual_token_pruning.config import VTRConfig
from visual_token_pruning.model.vtr_qwen3_vl import (
    VTRQwen3VLForConditionalGeneration,
)

from .qwen3_vl import Qwen3_VL


def _parse_int_or_list(val) -> Union[int, List[int]]:
    """Parse '3' -> 3, '3;7;16' -> [3, 7, 16]."""
    s = str(val)
    if ";" in s:
        return [int(x) for x in s.split(";")]
    return int(s)


def _parse_float_or_list(val) -> Union[float, List[float]]:
    """Parse '0.5' -> 0.5, '0.52;0.35;0.19' -> [0.52, 0.35, 0.19]."""
    s = str(val)
    if ";" in s:
        return [float(x) for x in s.split(";")]
    return float(s)


@register_model("qwen3_vl_vtr")
class Qwen3_VL_VTR(Qwen3_VL):
    """Qwen3-VL with VTR support for lmms-eval benchmarking."""

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-8B-Instruct",
        # VTR parameters
        vtr_enabled: bool = True,
        vtr_strategy: str = "priortr",
        vtr_keep_ratio: Union[str, float] = 0.1111,
        vtr_keep_tokens: Optional[str] = None,
        vtr_retain_ratio: Optional[Union[str, float]] = None,
        vtr_prune_layer: Union[str, int] = 3,
        vtr_query_aggregation: str = "auto",
        vtr_head_aggregation: str = "mean",
        vtr_token_merge: bool = False,
        vtr_merge_clusters: Union[str, int] = 10,
        vtr_important_ratio: float = 0.5,
        # Standard parameters
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,
        max_image_size: Optional[int] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        # Build VTR config with support for list-valued parameters
        config_kwargs = dict(
            enabled=vtr_enabled,
            strategy=vtr_strategy,
            keep_ratio=_parse_float_or_list(vtr_keep_ratio),
            prune_layer=_parse_int_or_list(vtr_prune_layer),
            query_aggregation=vtr_query_aggregation,
            head_aggregation=vtr_head_aggregation,
            token_merge=vtr_token_merge,
            merge_clusters=_parse_int_or_list(vtr_merge_clusters),
            important_ratio=float(vtr_important_ratio),
        )
        if vtr_keep_tokens is not None:
            config_kwargs["keep_tokens"] = _parse_int_or_list(vtr_keep_tokens)
        if vtr_retain_ratio is not None:
            # CLSE convenience knob: one nominal retain ratio (0.334 / 0.223 / 0.112)
            # activates the hard-coded per-stage schedule (see strategy/clse.py).
            config_kwargs["retain_ratio"] = float(vtr_retain_ratio)

        self._vtr_config = VTRConfig(**config_kwargs)

        # Call grandparent __init__ (lmms) to avoid Qwen3_VL loading the model
        from lmms_eval.api.model import lmms
        lmms.__init__(self)

        # Replicate Qwen3_VL.__init__ logic but with VTR model loading
        from accelerate import Accelerator, DistributedType
        from transformers import AutoProcessor, AutoTokenizer
        from loguru import logger as eval_logger

        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}")

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        model_kwargs = {}
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        # Load model with VTR support
        self._model = VTRQwen3VLForConditionalGeneration.from_pretrained_vtr(
            pretrained,
            vtr_config=self._vtr_config,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
            **model_kwargs,
        ).eval()

        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames

        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals

        self._config = self.model.config
        self._max_length = 2048
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ]
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

        eval_logger.info(
            f"VTR Config: enabled={self._vtr_config.enabled}, "
            f"strategy={self._vtr_config.strategy}, "
            f"keep_ratio={self._vtr_config.keep_ratio}, "
            f"retain_ratio={self._vtr_config.retain_ratio}, "
            f"prune_layer={self._vtr_config.prune_layer}"
        )
