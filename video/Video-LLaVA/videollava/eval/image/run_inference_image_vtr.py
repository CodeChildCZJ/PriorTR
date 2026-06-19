"""
Image evaluation script with VTR support for MME and GQA benchmarks.
Supports baseline, fastv, and infovtr_fixed strategies.
"""
import math
import os
import argparse
import json

import torch
from tqdm import tqdm
from PIL import Image

from videollava.conversation import conv_templates, SeparatorStyle
from videollava.constants import (
    DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN,
    IMAGE_TOKEN_INDEX
)
from videollava.mm_utils import (
    get_model_name_from_path, tokenizer_image_token,
    KeywordsStoppingCriteria, process_images
)
from videollava.vtr.model import load_vtr_model
from videollava.vtr import InfoVTRConfig, VTRConfig


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def parse_args():
    parser = argparse.ArgumentParser()

    # Model configuration
    parser.add_argument('--model_path', help='Path to model', required=True)
    parser.add_argument('--model_base', help='Base model path', default=None, type=str)
    parser.add_argument('--cache_dir', help='HuggingFace cache dir', default=None)

    # Data configuration
    parser.add_argument('--question_file', help='Path to question file (jsonl)', required=True)
    parser.add_argument('--image_folder', help='Directory containing images', required=True)
    parser.add_argument('--answers_file', help='Path to save answers', required=True)

    # Chunking for multi-GPU
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)

    # Generation configuration
    parser.add_argument("--conv_mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)

    # Device
    parser.add_argument("--device", type=str, default='cuda')

    # VTR configuration
    parser.add_argument('--vtr_enabled', action='store_true', help='Enable VTR pruning')
    parser.add_argument('--vtr_strategy', type=str, default='infovtr',
                       choices=['infovtr', 'fastv'], help='VTR strategy')
    parser.add_argument('--vtr_prune_layer', type=int, default=3, help='Layer to prune at')
    parser.add_argument('--vtr_keep_tokens', type=int, default=192, help='Number of tokens to keep')
    parser.add_argument('--vtr_query_aggregation', type=str, default='question',
                       choices=['question', 'last'], help='Query aggregation method')
    parser.add_argument('--vtr_head_aggregation', type=str, default='mean',
                       choices=['mean', 'max'], help='Head aggregation method')

    # Limit samples for quick testing
    parser.add_argument('--num_samples', type=int, default=-1, help='Limit number of samples (-1 for all)')

    return parser.parse_args()


def eval_model(args):
    # Initialize model with VTR support
    model_name = get_model_name_from_path(args.model_path)
    print(f"Loading model: {model_name}")

    tokenizer, model, processor, context_len = load_vtr_model(
        model_path=args.model_path,
        model_base=args.model_base,
        model_name=model_name,
        model_type="infovtr_fixed",
        device=args.device,
    )
    model = model.to(args.device)

    # Configure VTR
    if args.vtr_enabled:
        print(f"VTR enabled: strategy={args.vtr_strategy}, layer={args.vtr_prune_layer}, "
              f"keep_tokens={args.vtr_keep_tokens}, query_agg={args.vtr_query_aggregation}")
        cfg = InfoVTRConfig(
            enabled=True,
            strategy=args.vtr_strategy,
            prune_layer=args.vtr_prune_layer,
            keep_tokens=args.vtr_keep_tokens,
            prior_prompt="",
            query_aggregation=args.vtr_query_aggregation,
            head_aggregation=args.vtr_head_aggregation
        )
    else:
        print("VTR disabled (baseline mode)")
        cfg = VTRConfig(enabled=False)

    model.setup_vtr(cfg)

    # Get image processor
    image_processor = processor.get('image', None)
    if image_processor is None:
        raise ValueError("Image processor not found in model")

    # Load questions
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    if args.num_samples > 0:
        questions = questions[:args.num_samples]

    print(f"Evaluating {len(questions)} samples")

    # Prepare output file
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    # Check conv_mode
    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'Auto switching to {args.conv_mode}.')

    for line in tqdm(questions, desc="Evaluating"):
        idx = line.get("question_id", line.get("id"))
        image_file = line.get("image", line.get("image_path"))
        qs = line.get("text", line.get("question"))

        # Build prompt with image token
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Process image
        image_path = os.path.join(args.image_folder, image_file)
        image = Image.open(image_path).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]
        image_tensor = image_tensor.to(dtype=torch.float16, device=args.device)

        # Tokenize input
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        input_ids = input_ids.unsqueeze(0).to(args.device)

        # Generate
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=[image_tensor],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )

        # Decode output
        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        # Write result
        result = {
            "question_id": idx,
            "prompt": line.get("text", line.get("question")),
            "text": outputs,
            "model_id": model_name,
        }
        ans_file.write(json.dumps(result) + "\n")

    ans_file.close()
    print(f"Results saved to {answers_file}")


if __name__ == "__main__":
    args = parse_args()
    eval_model(args)
