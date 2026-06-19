import math
import os
import argparse
import json
import random

import numpy as np
import torch
import transformers
from tqdm import tqdm
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN
from videollava.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from videollava.model.builder import load_pretrained_model
from videollava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from videollava.train.train import smart_tokenizer_and_embedding_resize


from videollava.vtr.model import load_vtr_model
from videollava.vtr import PriorTR2FConfig, VTRConfig

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--model_path', help='', required=True)
    parser.add_argument('--cache_dir', help='', required=True)
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file_question', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--gt_file_answers', help='Path to the ground truth file containing answers.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument('--model_base', help='', default=None, type=str, required=False)
    parser.add_argument("--model_max_length", type=int, required=False, default=2048)

    # VTR configuration
    parser.add_argument('--vtr_enabled', action='store_true', help='Enable VTR pruning')
    parser.add_argument('--vtr_strategy', type=str, default='priortr_2f', choices=['priortr_2f', 'fastv'], help='VTR strategy')
    parser.add_argument('--vtr_prune_layer', type=int, default=3, help='Layer to prune at')
    parser.add_argument('--vtr_keep_tokens', type=int, default=194, help='Number of tokens to keep')
    parser.add_argument('--vtr_query_aggregation', type=str, default='question', choices=['question', 'last'], help='Query aggregation method')
    parser.add_argument('--vtr_head_aggregation', type=str, default='mean', choices=['mean', 'max'], help='Head aggregation method')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to evaluate')

    return parser.parse_args()

def get_model_output(model, video_processor, tokenizer, video, qs, args):
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_VID_START_TOKEN + ''.join([DEFAULT_IMAGE_TOKEN]*8) + DEFAULT_VID_END_TOKEN + '\n' + qs
    else:
        qs = ''.join([DEFAULT_IMAGE_TOKEN]*8) + '\n' + qs

    conv_mode = "llava_v1"
    args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()


    video_tensor = video_processor.preprocess(video, return_tensors='pt')['pixel_values'][0].half().to(args.device)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(args.device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[video_tensor],
            do_sample=False,
            temperature=0.0,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    print(outputs)
    return outputs


def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Output directory information before running
    print("=" * 80)
    print("Pre-run output directory info:")
    print(f"  Output dir: {args.output_dir}")
    if os.path.exists(args.output_dir):
        existing_files = [f for f in os.listdir(args.output_dir) if os.path.isfile(os.path.join(args.output_dir, f))]
        print(f"  Existing files: {len(existing_files)}")
        if existing_files:
            print(f"  File list: {', '.join(existing_files[:10])}" + ("..." if len(existing_files) > 10 else ""))
    else:
        print(f"  Output dir does not exist, will be created")
    print("=" * 80)

    # Set random seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Initialize the model
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_vtr_model(
        model_path=args.model_path,
        model_base=args.model_base,
        model_name=model_name,
        model_type="priortr_2f_fixed",
        device=args.device,
    )
    model = model.to(args.device)
    model.eval()

    # Configure VTR
    if args.vtr_enabled:
        print(f"VTR enabled: strategy={args.vtr_strategy}, prune_layer={args.vtr_prune_layer}, "
              f"keep_tokens={args.vtr_keep_tokens}, query_agg={args.vtr_query_aggregation}, "
              f"head_agg={args.vtr_head_aggregation}")
        # Use PriorTR2FConfig to support all strategies (priortr_2f, fastv, etc.)
        # because priortr_2f_fixed model requires prior_prompt attribute
        cfg = PriorTR2FConfig(
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

    # Load ground truth files containing questions and answers
    gt_questions = json.load(open(args.gt_file_question, "r"))
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)
    gt_questions = gt_questions[:args.num_samples]
    gt_answers = json.load(open(args.gt_file_answers, "r"))
    gt_answers = get_chunk(gt_answers, args.num_chunks, args.chunk_idx)
    print(f"Evaluating {len(gt_questions)} samples")

    answers_file = os.path.join(args.output_dir, f"{args.output_name}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    ans_file = open(answers_file, "w")

    output_list = []  # List to store the output results

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    # Iterate over each sample in the ground truth file
    index = 0
    for sample in tqdm(gt_questions):
        video_name = sample['video_name']
        question = sample['question']
        id = sample['question_id']
        answer = gt_answers[index]['answer']
        index += 1

        sample_set = {'id': id, 'question': question, 'answer': answer}

        # Load the video file
        for fmt in tqdm(video_formats):  # Added this line
            # Try plain name first, then with v_ prefix (ActivityNet uses v_ prefix)
            temp_path = os.path.join(args.video_dir, f"{video_name}{fmt}")
            if not os.path.exists(temp_path):
                temp_path = os.path.join(args.video_dir, f"v_{video_name}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                output = get_model_output(model, processor['video'], tokenizer, video_path, question, args)
                sample_set['pred'] = output
                output_list.append(sample_set)
                ans_file.write(json.dumps(sample_set) + "\n")
                break

    ans_file.close()

    # Post-run output directory info
    print("=" * 80)
    print("Post-run output directory info:")
    print(f"  Output dir: {args.output_dir}")
    if os.path.exists(args.output_dir):
        existing_files = [f for f in os.listdir(args.output_dir) if os.path.isfile(os.path.join(args.output_dir, f))]
        print(f"  Total files: {len(existing_files)}")
        if existing_files:
            print(f"  File list: {', '.join(existing_files[:10])}" + ("..." if len(existing_files) > 10 else ""))
        output_file = os.path.join(args.output_dir, f"{args.output_name}.json")
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"  Output file: {args.output_name}.json ({file_size} bytes)")
    print("=" * 80)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
