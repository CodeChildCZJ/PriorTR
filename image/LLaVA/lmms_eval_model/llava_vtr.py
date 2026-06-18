import torch
import copy
import warnings
import logging
from datetime import timedelta
from typing import List, Optional, Tuple, Union
from dataclasses import fields, is_dataclass, asdict
from typing import get_type_hints

from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from packaging import version
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

warnings.filterwarnings("ignore")
eval_logger = logging.getLogger("lmms-eval")

# =========================================================
# 1. Import VTR components
# =========================================================
try:
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
    

    # [VTR] Import Builder and Config
    from llava.vtr.model import load_vtr_model
    from llava.vtr.config import VTRConfig
except ImportError as e:
    eval_logger.error(f"LLaVA or VTR modules not found. Error: {e}")

# =========================================================
# 2. VTR strategy mapping
# =========================================================
STRATEGY_TO_CONFIG = {
    "priortr": VTRConfig,
}

if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"

@register_model("llava_vtr")
class LlavaVTR(lmms):
    """
    Llava VTR Model Wrapper.
    Modified from lmms-eval simple/llava.py to support the VTR framework.
    """

    def __init__(
        self,
        pretrained: str = "liuhaotian/llava-v1.5-7b",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        model_name=None,
        attn_implementation=best_fit_attn_implementation,
        device_map="cuda:0",
        conv_template="vicuna_v1",
        use_cache=True,
        tie_weights: bool = True,
        truncate_context=False,
        customized_config=None,
        **kwargs,
    ) -> None:
        super().__init__()

        # ======================================================================
        # 1. Accelerator initialization (preserves original logic)
        # ======================================================================
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator
        
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        # ======================================================================
        # 2. VTR config parsing & adaptive model type
        # ======================================================================
        # Extract strategy name
        strategy_name = kwargs.pop("strategy", kwargs.pop("vtr_strategy", "priortr"))
        TargetConfigClass = STRATEGY_TO_CONFIG.get(strategy_name, VTRConfig)

        # Smart parse kwargs into VTR Config
        vtr_config_kwargs = self._smart_parse_kwargs(TargetConfigClass, kwargs)
        vtr_config_kwargs["strategy"] = strategy_name
        if "enabled" not in vtr_config_kwargs:
            vtr_config_kwargs["enabled"] = True

        # keep_tokens and query_aggregation default to None (auto)
        # setup_vtr() will auto-fill based on model version
        self.vtr_config = TargetConfigClass(**vtr_config_kwargs)

        # model_type inference
        model_type = kwargs.pop("model_type", "priortr")

        model_name = model_name if model_name is not None else get_model_name_from_path(pretrained)
        eval_logger.info(f"Loading VTR Model: {pretrained} | Strategy: {strategy_name} | Adaptive Type: {model_type}")

        # ======================================================================
        # 3. Build loading parameters & robust loading (following original try-except)
        # ======================================================================
        llava_model_args = {
            "multimodal": True, # Add by default, following original behavior
        }
        if customized_config is not None:
            llava_model_args["customized_config"] = customized_config
        if attn_implementation is not None:
            llava_model_args["attn_implementation"] = attn_implementation
        if "use_flash_attention_2" in kwargs:
            llava_model_args["use_flash_attention_2"] = kwargs["use_flash_attention_2"]

        # [Key modification] Follow original try-except mechanism for model loading
        # Ensures code works regardless of whether the Builder supports the multimodal parameter
        try:
            # Attempt 1: load with multimodal parameter
            self._tokenizer, self._model, self._image_processor, self._max_length = load_vtr_model(
                model_path=pretrained,
                model_type=model_type,       # Pass adaptively inferred type
                device_map=self.device_map,
                **llava_model_args           # includes multimodal=True
            )
        except TypeError:
            # Attempt 2: if error (unexpected keyword argument), remove multimodal and retry
            eval_logger.debug("load_vtr_model doesn't support 'multimodal' arg, removing it.")
            llava_model_args.pop("multimodal", None)
            self._tokenizer, self._model, self._image_processor, self._max_length = load_vtr_model(
                model_path=pretrained,
                model_type=model_type,
                device_map=self.device_map,
                **llava_model_args           # without multimodal
            )



        # ======================================================================
        # 4. Inject VTR state (setup_vtr auto-fills version-specific default hyperparams)
        # ======================================================================
        if self.vtr_config.enabled:
            self._model.setup_vtr(self.vtr_config)
        else:
            eval_logger.info("VTR is DISABLED.")

        # ======================================================================
        # 5. Print colored config panel (after setup_vtr, showing final values)
        # ======================================================================
        if self.accelerator.is_main_process:
            CYAN = "\033[96m"
            GREEN = "\033[92m"
            YELLOW = "\033[93m"
            RED = "\033[91m"
            BOLD = "\033[1m"
            RESET = "\033[0m"

            print(f"\n{CYAN}{'='*60}{RESET}")
            print(f"{GREEN}{BOLD} VTR Framework Configuration Panel{RESET}")
            print(f"{CYAN}{'='*60}{RESET}")
            print(f" {BOLD}Model Path{RESET}    : {YELLOW}{pretrained}{RESET}")
            print(f" {BOLD}Strategy{RESET}      : {YELLOW}{strategy_name.upper()}{RESET}")
            print(f" {BOLD}Model Type{RESET}    : {YELLOW}{model_type}{RESET}")
            print(f"{CYAN}{'-'*60}{RESET}")
            print(f"{BOLD} VTR Parameters:{RESET}")

            config_dict = asdict(self.vtr_config)
            for key, value in config_dict.items():
                color = YELLOW
                if key == "enabled":
                    color = GREEN if value else RED
                elif key == "prune_layer":
                    color = "\033[95m"
                elif key == "keep_ratio":
                    color = "\033[94m"
                print(f"    {key:<20}: {color}{value}{RESET}")

            print(f"{CYAN}{'='*60}{RESET}\n")

        # ======================================================================
        # 6. Standard follow-up flow (preserves original)
        # ======================================================================
        self._config = self._model.config
        self.model.eval()
        if tie_weights:
            self.model.tie_weights()

        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context

        # Accelerate Prepare
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type."
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs_ds = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs_ds)
            
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._rank = 0
            self._world_size = 1
        else:
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    # ----------------------------------------------------------------------
    # Helper: smart kwargs parsing
    # ----------------------------------------------------------------------
    def _smart_parse_kwargs(self, config_class, kwargs):
        """
        Extract parameters from kwargs (str) based on dataclass field definitions, with auto type conversion.
        [Fixed] Prioritizes List-type string inputs to resolve multi-layer prune_layer parsing issues.
        """
        if not is_dataclass(config_class):
            return {}

        valid_fields = {f.name: f for f in fields(config_class)}
        type_hints = get_type_hints(config_class)
        parsed_args = {}

        for k, v in kwargs.items():
            # 1. Key cleaning: allow vtr_ prefix
            clean_key = k
            if k not in valid_fields and k.startswith("vtr_") and k[4:] in valid_fields:
                clean_key = k[4:]
            
            if clean_key in valid_fields:
                target_type = type_hints.get(clean_key)
                
                # If input is not a string (direct code call), assign directly
                if not isinstance(v, str):
                    parsed_args[clean_key] = v
                    continue

                # =========================================================
                # [Key fix] Prioritize detecting and parsing List strings "[1,2,3]"
                # Regardless of target_type being int or list, parse if it looks like a list
                # =========================================================
                if v.strip().startswith('[') and v.strip().endswith(']'):
                    try:
                        v_clean = v.strip("[]")
                        if not v_clean:
                            parsed_args[clean_key] = []
                        else:
                            # Try to parse as int list
                            parsed_args[clean_key] = [int(x.strip()) for x in v_clean.split(',')]
                        # If parsing succeeds, skip subsequent logic
                        continue 
                    except ValueError:
                        # If int conversion fails (may be string list), fall through to regular logic
                        pass

                # =========================================================
                # Standard type hint parsing
                # =========================================================
                try:
                    if target_type == bool:
                        parsed_args[clean_key] = v.lower() in ('true', '1', 'yes', 'on')
                    elif target_type == int:
                        parsed_args[clean_key] = int(v)
                    elif target_type == float:
                        parsed_args[clean_key] = float(v)
                    # Handle Optional types
                    elif "NoneType" in str(target_type) and (v.lower() == 'none' or v == ''):
                            parsed_args[clean_key] = None
                    elif "float" in str(target_type):
                            parsed_args[clean_key] = float(v)
                    # Handle explicitly defined List types (fallback)
                    elif str(target_type).startswith("typing.List") or str(target_type).startswith("typing.Tuple"):
                        v_clean = v.strip("[]()")
                        if v_clean:
                            parsed_args[clean_key] = [int(x.strip()) for x in v_clean.split(',')]
                        else:
                            parsed_args[clean_key] = []
                    else:
                        parsed_args[clean_key] = v
                except Exception as e:
                    eval_logger.warning(f"Failed to parse argument {k}={v} as {target_type}: {e}. Keeping as string.")
                    parsed_args[clean_key] = v
        
        return parsed_args

    # ----------------------------------------------------------------------
    # Properties and methods below are copied directly from simple/llava.py
    # VTR does not change the input format, only the internal forward behavior
    # ----------------------------------------------------------------------

    @property
    def config(self):
        return self._config

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
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

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

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])
            
    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def flatten(self, input):
        if not input or any(i is None for i in input):
            return []
        new_list = []
        for i in input:
            if i:
                for j in i:
                    new_list.append(j)
        return new_list

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # loglikelihood implementation (from simple/llava.py)
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        
        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # Ensure images are on the correct device
            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            image_sizes = [[visual.size[0], visual.size[1]] for visual in visuals]
            if visuals:
                image = process_images(visuals, self._image_processor, self._config)
                if type(image) is list:
                    image = [_image.to(dtype=torch.float16, device=self.device) for _image in image]
                else:
                    image = image.to(dtype=torch.float16, device=self.device)
            else:
                image = None

            prompts_input = contexts[0] if isinstance(contexts, list) else contexts

            if image is not None and len(image) != 0 and DEFAULT_IMAGE_TOKEN not in prompts_input:
                image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visuals)
                image_tokens = " ".join(image_tokens)
                prompts_input = image_tokens + "\n" + (contexts[0] if isinstance(contexts, list) else contexts)

            if "llama_3" in self.conv_template:
                conv = copy.deepcopy(conv_templates[self.conv_template])
            else:
                conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], prompts_input)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            contxt_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            conv.messages[1][1] = continuation

            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            labels = input_ids.clone()
            labels[0, : contxt_id.shape[1]] = -100
            with torch.inference_mode():
                outputs = self.model(input_ids=input_ids, labels=labels, images=image, use_cache=True, image_sizes=image_sizes)
            loss = outputs["loss"]
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = input_ids[:, contxt_id.shape[1] :]
            greedy_tokens = greedy_tokens[:, contxt_id.shape[1] : input_ids.shape[1]]
            max_equal = (greedy_tokens == cont_toks).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)
        pbar.close()
        return res

    def generate_until(self, requests: List[Instance]) -> List[str]:
        # generate_until implementation (from simple/llava.py, VTR-compatible with HF generate interface)
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            batched_visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            flattened_visuals = self.flatten(batched_visuals)
            gen_kwargs = all_gen_kwargs[0]

            until = [self.tok_decode(self.eot_token_id)]
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            if "image_aspect_ratio" in gen_kwargs.keys() and "image_aspect_ratio" not in self._config.__dict__:
                self._config.image_aspect_ratio = gen_kwargs.pop("image_aspect_ratio")
                eval_logger.info(f"Setting image aspect ratio: {self._config.image_aspect_ratio}")
            
            if flattened_visuals:
                image_tensor = process_images(flattened_visuals, self._image_processor, self._config)
                if type(image_tensor) is list:
                    image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                else:
                    image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
            else:
                image_tensor = None

            question_input = []
            for visual, context in zip(batched_visuals, contexts):
                if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                    image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visual) if isinstance(visual, list) else [DEFAULT_IMAGE_TOKEN]
                    image_tokens = " ".join(image_tokens)
                    question = image_tokens + "\n" + context
                else:
                    question = context
                
                if "llama_3" in self.conv_template:
                    conv = copy.deepcopy(conv_templates[self.conv_template])
                else:
                    conv = conv_templates[self.conv_template].copy()
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()
                question_input.append(prompt_question)

            gen_kwargs["image_sizes"] = [flattened_visuals[idx].size for idx in range(len(flattened_visuals))]
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in question_input]
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
            attention_masks = input_ids.ne(pad_token_ids).to(self.device)
            
            try:
                cont = self.model.generate(
                    input_ids,
                    attention_mask=attention_masks,
                    pad_token_id=pad_token_ids,
                    images=image_tensor,
                    image_sizes=gen_kwargs["image_sizes"],
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                )
                text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
            except Exception as e:
                raise e
                eval_logger.error(f"Error {e} in generating")
                cont = ""
                text_outputs = [""]

            res.extend(text_outputs)
            pbar.update(1)
            
        res = re_ords.get_original(res)
        pbar.close()
        return res
    
    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for LLaVA")
    