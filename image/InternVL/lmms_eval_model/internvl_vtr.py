"""
lmms-eval wrapper for InternVL2.5 with VTR (PriorTR / FastV / Baseline).

Usage in eval.sh:
    lmms-eval \
        --model internvl_vtr \
        --model_args "pretrained=OpenGVLab/InternVL2_5-8B,strategy=priortr,keep_tokens=192,prune_layer=2" \
        --tasks mme --batch_size 1

model_args:
    pretrained   : path to InternVL2_5-8B directory (required)
    strategy     : priortr | fastv | baseline (default: baseline)
    keep_tokens  : int, visual tokens to keep (e.g. 192)
    keep_ratio   : float, keep ratio (default: 0.25, ignored when keep_tokens set)
    prune_layer  : int, pruning layer 1-indexed (default: 2)
"""

import logging
import math
import os
import sys
from datetime import timedelta
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from accelerate.utils import InitProcessGroupKwargs
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# ---------------------------------------------------------------------------
# Add InternVL/ to sys.path so internvl_vtr package is importable
# ---------------------------------------------------------------------------
_INTERNVL_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
if _INTERNVL_DIR not in sys.path:
    sys.path.insert(0, _INTERNVL_DIR)

from internvl_vtr.config import VTRConfig
from internvl_vtr.model.vtr_internvl import setup_vtr_model

eval_logger = logging.getLogger("eval_logger")

# ---------------------------------------------------------------------------
# Image preprocessing (same as internvl2.py reference)
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_GEN_KWARGS = dict(
    num_beams=1,
    max_new_tokens=1024,
    do_sample=False,
)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image, input_size=448, max_num=6):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

@register_model("internvl_vtr")
class InternVLVTR(lmms):
    """
    lmms-eval wrapper for InternVL2.5 with Visual Token Reduction.

    Supports three strategies:
        - baseline: no token pruning, vanilla model.chat()
        - fastv:    FastV attention-based pruning
        - priortr:  PriorTR prior-guided pruning
    """

    def __init__(
        self,
        pretrained: str = "OpenGVLab/InternVL2_5-8B",
        strategy: str = "baseline",
        keep_tokens: str = "",
        keep_ratio: str = "0.25",
        prune_layer: str = "2",
        query_aggregation: str = "question",
        head_aggregation: str = "mean",
        max_num: str = "6",
        device: str = "cuda:0",
        device_map: str = "cuda:0",
        batch_size: str = "1",
        **kwargs,
    ):
        super().__init__()
        self._max_num = int(max_num)

        self.path = pretrained
        self._strategy = strategy

        batch_size = int(batch_size)
        assert batch_size == 1, f"Batch size must be 1 for InternVLVTR, but got {batch_size}."
        self.batch_size_per_gpu = batch_size

        # Parse VTR parameters (lmms-eval passes everything as strings)
        _keep_tokens = int(keep_tokens) if keep_tokens else None
        _keep_ratio = float(keep_ratio)
        _prune_layer = int(prune_layer)

        # ---- Accelerator setup ----
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator

        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        # ---- Load model and tokenizer ----
        self._model = AutoModel.from_pretrained(
            self.path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval()

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.path,
            trust_remote_code=True,
        )

        # ---- Multi-GPU handling ----
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type. Only DDP, FSDP, and DeepSpeed are supported."

            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                dp_kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(
                    must_match=True, **dp_kwargs
                )
                eval_logger.info(
                    "Detected DistributedType.DEEPSPEED. "
                    "Make sure you ran `accelerate config` and set zero stage to 0."
                )

            if accelerator.distributed_type in (DistributedType.FSDP, DistributedType.DEEPSPEED):
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)

            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

        # ---- Apply VTR patching (if not baseline) ----
        if strategy != "baseline":
            vtr_config = VTRConfig(
                enabled=True,
                strategy=strategy,
                prune_layer=_prune_layer,
                keep_tokens=_keep_tokens,
                keep_ratio=_keep_ratio,
                query_aggregation=query_aggregation,
                head_aggregation=head_aggregation,
            )
            setup_vtr_model(self.model, vtr_config, self._tokenizer)
            eval_logger.info(
                "VTR enabled: strategy=%s, prune_layer=%d, keep_tokens=%s, keep_ratio=%s, "
                "query_aggregation=%s, head_aggregation=%s",
                strategy, _prune_layer, _keep_tokens, _keep_ratio,
                query_aggregation, head_aggregation,
            )
        else:
            eval_logger.info("VTR disabled (baseline mode)")

    # ---- Properties required by lmms base class ----

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    # ---- Core evaluation method ----

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # ---- Merge generation kwargs ----
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")
            for k, v in DEFAULT_GEN_KWARGS.items():
                if k not in gen_kwargs:
                    gen_kwargs[k] = v

            # Remove unsupported kwargs
            pop_keys = [k for k in gen_kwargs if k not in DEFAULT_GEN_KWARGS]
            for k in pop_keys:
                gen_kwargs.pop(k)

            # ---- Process visuals ----
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)

            if visuals:
                pixel_values_list = [
                    load_image(visual, max_num=self._max_num).to(torch.bfloat16).cuda() for visual in visuals
                ]
                pixel_values = torch.cat(pixel_values_list, dim=0)
                num_patches_list = [pv.size(0) for pv in pixel_values_list]
                image_tokens = " ".join(["<image>"] * len(visuals))
                query = image_tokens + "\n" + contexts
            else:
                pixel_values = None
                num_patches_list = None
                query = contexts

            # ---- Generate response ----
            # model.chat() works for both baseline and VTR:
            #   - baseline: calls original generate -> standard LLM forward
            #   - VTR: calls _vtr_generate (monkeypatched) -> pruned LLM forward
            response, history = self.model.chat(
                self.tokenizer,
                pixel_values,
                query,
                gen_kwargs,
                num_patches_list=num_patches_list,
                history=None,
                return_history=True,
            )

            res.append(response)
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        assert False, "loglikelihood is not implemented for InternVLVTR."

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for InternVLVTR.")
