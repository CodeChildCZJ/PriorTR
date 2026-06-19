import openai
import os
import argparse
import json
import ast
import time
from multiprocessing.pool import Pool
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--pred_path", default=r'', help="The path to file containing prediction.")
    parser.add_argument("--output_dir", default=r'', help="The path to save annotation json files.")
    parser.add_argument("--output_json", default=r'', help="The path to save annotation final combined json file.")
    parser.add_argument("--api_key", default="", help="OpenAI API key.")
    parser.add_argument("--api_base", default="", type=str, help="OpenAI API base.")
    parser.add_argument("--num_tasks", default=4, type=int, help="Number of parallel tasks (lower = less concurrency).")
    parser.add_argument("--max_retries", default=3, type=int, help="Max retries per sample before skipping.")
    parser.add_argument("--retry_delay", default=2.0, type=float, help="Delay between retries in seconds.")
    args = parser.parse_args()
    return args


def annotate_single(qa_set, key, output_dir, args, max_retries=3, retry_delay=2.0):
    """
    Evaluates a single question-answer pair using GPT-3 with retry logic.
    Returns True if successful, False if skipped.
    """
    openai.api_key = args.api_key
    if args.api_base:
        openai.api_base = args.api_base

    question = qa_set['q']
    answer = qa_set['a']
    pred = qa_set['pred']

    for attempt in range(max_retries):
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content":
                            "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                            "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                            "------"
                            "##INSTRUCTIONS: "
                            "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                            "- Consider synonyms or paraphrases as valid matches.\n"
                            "- Evaluate the correctness of the prediction compared to the answer."
                    },
                    {
                        "role": "user",
                        "content":
                            "Please evaluate the following video-based question-answer pair:\n\n"
                            f"Question: {question}\n"
                            f"Correct Answer: {answer}\n"
                            f"Predicted Answer: {pred}\n\n"
                            "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                            "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                            "For example, your response should look like this: {'pred': 'yes', 'score': 4}."
                    }
                ],
                timeout=30  # Add timeout
            )

            response_message = completion["choices"][0]["message"]["content"]
            response_dict = ast.literal_eval(response_message)

            # Validate response format
            if 'pred' not in response_dict or 'score' not in response_dict:
                raise ValueError(f"Invalid response format: {response_message}")

            # Ensure score is integer
            response_dict['score'] = int(response_dict['score'])

            result_qa_pair = [response_dict, qa_set]

            with open(f"{output_dir}/{key}.json", "w") as f:
                json.dump(result_qa_pair, f)

            return True

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print(f"[SKIP] Failed after {max_retries} attempts for '{key}': {e}")
                return False

    return False


def annotate(prediction_set, caption_files, output_dir, args):
    """
    Evaluates question and answer pairs using GPT-3 with error handling.
    """
    skipped = []
    for file in caption_files:
        key = file[:-5]  # Strip file extension
        qa_set = prediction_set[key]

        success = annotate_single(
            qa_set, key, output_dir, args,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay
        )

        if not success:
            skipped.append(key)

        # Small delay between requests to avoid rate limiting
        time.sleep(0.1)

    return skipped


def main():
    """
    Main function to control the flow of the program.
    """
    args = parse_args()

    print(f"Configuration: num_tasks={args.num_tasks}, max_retries={args.max_retries}, retry_delay={args.retry_delay}s")

    file = open(args.pred_path)
    new_pred_contents = [eval(i.strip()) for i in file.readlines()]

    total_samples = len(new_pred_contents)
    print(f"Total samples to evaluate: {total_samples}")

    # Generating list of id's and corresponding files
    id_list = [x['id'] for x in new_pred_contents]
    caption_files = [f"{id}.json" for id in id_list]

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Preparing dictionary of question-answer sets
    prediction_set = {}
    for sample in new_pred_contents:
        id = sample['id']
        question = sample['question']
        answer = sample['answer']
        pred = sample['pred']
        qa_set = {"q": question, "a": answer, "pred": pred}
        prediction_set[id] = qa_set

    num_tasks = args.num_tasks
    all_skipped = []
    max_loops = 5  # Maximum number of retry loops to prevent infinite loops

    for loop in range(max_loops):
        try:
            completed_files = os.listdir(output_dir)
            print(f"Loop {loop + 1}: completed_files: {len(completed_files)}")

            incomplete_files = [f for f in caption_files if f not in completed_files]
            print(f"Loop {loop + 1}: incomplete_files: {len(incomplete_files)}")

            if len(incomplete_files) == 0:
                break

            # Skip files that have been tried too many times
            files_to_skip = [f for f in incomplete_files if f[:-5] in all_skipped]
            incomplete_files = [f for f in incomplete_files if f[:-5] not in all_skipped]

            if len(incomplete_files) == 0:
                print(f"All remaining {len(files_to_skip)} files have been skipped due to errors.")
                break

            if len(incomplete_files) <= num_tasks:
                num_tasks_actual = 1
            else:
                num_tasks_actual = num_tasks

            part_len = max(1, len(incomplete_files) // num_tasks_actual)
            all_parts = [incomplete_files[i:i + part_len] for i in range(0, len(incomplete_files), part_len)]
            task_args = [(prediction_set, part, args.output_dir, args) for part in all_parts]

            # Use a pool of workers to process the files in parallel
            with Pool(processes=num_tasks_actual) as pool:
                results = pool.starmap(annotate, task_args)

            # Collect skipped files
            for result in results:
                if result:
                    all_skipped.extend(result)

        except Exception as e:
            print(f"Error in loop {loop + 1}: {e}")
            time.sleep(5)

    # Combine all the processed files into one
    combined_contents = {}
    json_path = args.output_json

    for file_name in os.listdir(output_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(output_dir, file_name)
            try:
                with open(file_path, "r") as json_file:
                    content = json.load(json_file)
                    combined_contents[file_name[:-5]] = content
            except Exception as e:
                print(f"Error reading {file_name}: {e}")

    with open(json_path, "w") as json_file:
        json.dump(combined_contents, json_file)
    print("All evaluation completed!")

    # Calculate average score and accuracy
    score_sum = 0
    count = 0
    yes_count = 0
    no_count = 0
    error_count = 0

    for key, result in tqdm(combined_contents.items()):
        try:
            count += 1
            score_match = result[0]['score']
            score = int(score_match)
            score_sum += score

            pred = result[0]['pred']
            if "yes" in pred.lower():
                yes_count += 1
            elif "no" in pred.lower():
                no_count += 1
        except Exception as e:
            error_count += 1
            print(f"Error processing result for {key}: {result}")

    # Summary statistics
    valid_samples = yes_count + no_count
    skipped_samples = len(all_skipped)

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total samples: {total_samples}")
    print(f"Evaluated samples: {count}")
    print(f"Valid samples (yes+no): {valid_samples}")
    print(f"Skipped samples: {skipped_samples}")
    print(f"Parse errors: {error_count}")
    print("-" * 50)
    print(f"Yes count: {yes_count}")
    print(f"No count: {no_count}")

    if valid_samples > 0:
        accuracy = yes_count / valid_samples
        print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    else:
        accuracy = 0
        print("Accuracy: N/A (no valid samples)")

    if count > 0:
        average_score = score_sum / count
        print(f"Average score: {average_score:.3f}")
    else:
        average_score = 0
        print("Average score: N/A")
    print("=" * 50)

    # Save summary to file
    summary = {
        "total_samples": total_samples,
        "evaluated_samples": count,
        "valid_samples": valid_samples,
        "skipped_samples": skipped_samples,
        "parse_errors": error_count,
        "yes_count": yes_count,
        "no_count": no_count,
        "accuracy": accuracy,
        "average_score": average_score,
        "skipped_ids": all_skipped
    }

    summary_path = json_path.replace(".json", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
