import os
import warnings
import shutil
import torch
from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig, AutoModelForCausalLM

from videollava.model import *
from videollava.constants import (
    DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
    DEFAULT_VIDEO_PATCH_TOKEN, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN,
)
# Import VTR classes
from .vtr_llava import VTRLlavaForCausalLM
from .fastv_llava import FastVLlava
from .infovtr_llava import FixedLayerInfoVTR, AdaptiveLayerInfoVTR

# Strategy to model class mapping
VTR_MODEL_CLASSES = {
    "vtr_base": VTRLlavaForCausalLM,
    "fastv": FastVLlava,
    "infovtr_fixed": FixedLayerInfoVTR,
    "infovtr_adaptive": AdaptiveLayerInfoVTR,
}

def load_vtr_model(model_path, model_base=None, model_name=None, model_type="fastv",
                   load_8bit=False, load_4bit=False, device_map="auto",
                   device="cuda", use_flash_attn=False, **kwargs):
    if model_name is None:
        from videollava.mm_utils import get_model_name_from_path
        model_name = get_model_name_from_path(model_path)

    # 1. Basic parameter setup
    kwargs = {"device_map": device_map, **kwargs}
    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    # Get the corresponding VTR model class
    vtr_class = VTR_MODEL_CLASSES.get(model_type, VTRLlavaForCausalLM)

    # 2. Core loading logic (following native LLaVA)
    if 'llava' in model_name.lower():
        # [Case A] Load LoRA model
        if 'lora' in model_name.lower() and model_base is not None:
            from videollava.model.language_model.llava_llama import LlavaConfig
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading VTR-LLaVA from base model...')

            # Use VTR class instead of native LlavaLlamaForCausalLM
            model = vtr_class.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)

            # Weight adjustment logic
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional weights (non-lora trainables)...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # Compatible with HF Hub
                from huggingface_hub import hf_hub_download
                non_lora_trainables = torch.load(hf_hub_download(repo_id=model_path, filename='non_lora_trainables.bin'), map_location='cpu')

            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')

        # [Case B] Load projector weights only (mm_projector.bin)
        elif model_base is not None:
            print('Loading VTR-LLaVA from base model with projector...')
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            cfg_pretrained = AutoConfig.from_pretrained(model_path)
            model = vtr_class.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)

        # [Case C] Standard full weight loading
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            model = vtr_class.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    else:
        # Non-LLaVA models (fallback to AutoModel)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    # 3. Post-processing (Vision Tower & Resize)
    processor = {"image": None, "video": None}

    if "llava" in model_name.lower():
        model._supports_cache_class = True

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            tokenizer.add_tokens([DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        if getattr(model.config, "mm_image_tower", None) is not None:
            image_tower = model.get_image_tower()
            if not image_tower.is_loaded:
                image_tower.load_model()
            image_tower.to(device=device, dtype=torch.float16)
            processor["image"] = image_tower.image_processor

        if getattr(model.config, "mm_video_tower", None) is not None:
            video_tower = model.get_video_tower()
            if not video_tower.is_loaded:
                video_tower.load_model()
            video_tower.to(device=device, dtype=torch.float16)
            processor["video"] = video_tower.video_processor

    # 4. Get Context Length
    context_len = getattr(model.config, "max_sequence_length", 2048)
    return tokenizer, model, processor, context_len
