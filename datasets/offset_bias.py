import sys
import os
import argparse
from datasets import load_dataset
from baseline.absolute_llm_judge_v2 import AbsoluteLLMJudge
from baseline.relative_llm_judge_v3 import RelativeLLMJudge
from transformers import AutoTokenizer
import re

dataset = load_dataset("aishwaryaxsahoo/offsetbias")
# dataset["test"] = dataset["test"].select(range(10))
# dataset["train"] = dataset["train"].select(range(10))

parser = argparse.ArgumentParser(description="Run Absolute LLM Judge on offsetbias Dataset")
parser.add_argument("--dataset_fold", type=str, default="test", help="Dataset split to use (e.g., test, validation)")
parser.add_argument("--output_file", type=str, required=True, help="Path to save the output CSV file")
parser.add_argument("--model_repo", type=str, required=True, help="Path to the model repository")
parser.add_argument("--baseline_type", type=str, required=True, help="Baseline type: absolute or relative")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_repo)

min_score = 1
max_score = 5


instruction_text = """You are a helpful and general purpose assistant. Answer the user query.
<user>{}</user>"""


rubrics = """
Does the model provide relevant and useful responses to the user's needs or questions?
"""


def get_llm_prompt_relative(instruction, response1, response2):
    if 'prometheus' in args.model_repo or 'llama' in args.model_repo:
        messages = [
        {"role": "system", "content": "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."},
        {"role": "user", "content": f"""Task Description:
        An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
        1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
        2. Make comparisons between Response A and Response B. Instead of examining Response A and Response B separately, go straight to the point and mention about the commonalities and differences between them.
        3. After writing the feedback, indicate the better response, either "A" or "B".
        4. The output format should look as follows: "Feedback: <Start_Feedback>write a feedback<End_Feedback> [RESULT] <Start_Better_response>Either "A" or "B"<End_better_response>"

        The instruction to evaluate:
        {instruction}

        Response A:
        {response1}

        Response B:
        {response2}

        Score Rubrics:
        {rubrics}

        Feedback: """}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)  



def get_llm_prompt_absolute(instruction, response):
    if 'prometheus' in args.model_repo or 'llama' in args.model_repo:
        messages = [
        {"role": "system", "content": "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."},
        {"role": "user", "content": f"""###Task Description:
        An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
        1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
        2. After writing a feedback, write a score that is an integer between {min_score} and {max_score}. You should refer to the score rubric.
        3. The output format should strictly be as follows: "(write a feedback) RESULT: (an overall score integer number between {min_score} and {max_score})"
        4. Please do not generate any other opening, closing, and explanations and strictly follow the format.

        ###The instruction to evaluate:
        {instruction}

        ###Response to evaluate:
        {response}

        ###Score Rubrics:
        {rubrics}

        ###Feedback: """}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)



def transform_data_relative(example):
    instruction = instruction_text.format(example["instruction"])
    response1 = example["output_1"]
    response2 = example["output_2"]
    return {
        "instruction": f"{example["instruction"]}",
        "prompt": get_llm_prompt_relative(instruction, response1, response2),
        "response1": example["output_1"],
        "response2": example["output_2"],
        "human_score": example["label"]-1
    }



def transform_data_absolute(example, index):
    instruction = f"{example['instruction']}"
    response1 = example["output_1"]
    response2 = example["output_2"]

    # Escape curly braces in the strings
    safe_instruction = instruction.replace("{", "{{").replace("}", "}}")
    safe_response1 = response1.replace("{", "{{").replace("}", "}}")
    safe_response2 = response2.replace("{", "{{").replace("}", "}}")

    model1 = "Model_A"  # Placeholder for model 1
    model2 = "Model_B"  # Placeholder for model 2

    return {
        "instruction": instruction,
        "index": index,  # Add the unique index here
        "model1": model1,
        "model2": model2,
        "response1": response1,
        "response2": response2,
        "human_score": example["label"] - 1,
        "prompt1": get_llm_prompt_absolute(safe_instruction, safe_response1),
        "prompt2": get_llm_prompt_absolute(safe_instruction, safe_response2)
    }



def _parse_feedback_and_score_relative(text):
    result_match = re.search(r"\[RESULT\]\s*(?:<Start_Better_response>([AB])</End_better_response>|([AB]))", text)
    if result_match:
        # score = result_match.group(1)
        score = result_match.group(1) or result_match.group(2)
        feedback = text[:result_match.start()].strip()
    else:
        score = None
        feedback = text.strip()
    if score in ['A', 'B']:
        score = 0 if score == 'A' else 1
    else:
        score = None
    
    return feedback, score



def _parse_feedback_and_score_absolute(text):
    result_match = re.search(r"RESULT:\s*(\d+)", text)
    if result_match:
        print(f"Result found: {result_match.group(1)}")
        score = int(result_match.group(1))
        feedback = text[:result_match.start()].strip()
    else:
        score = None
        feedback = text.strip()
    return feedback, score



def get_parser():
    if 'prometheus' in args.model_repo or 'llama' in args.model_repo:
        if 'relative' in args.baseline_type:
            return _parse_feedback_and_score_relative
        elif 'absolute' in args.baseline_type:
            return _parse_feedback_and_score_absolute


parse_feedback_and_score_func = get_parser()


offset_bias_transformed_dataset_relative = dataset[args.dataset_fold].map(transform_data_relative)
offset_bias_transformed_dataset_absolute = dataset[args.dataset_fold].map(
    lambda example, index: transform_data_absolute(example, index), 
    with_indices=True
)


transformed_dataset = [
    transform_data_absolute(example, index) for index, example in enumerate(offset_bias_transformed_dataset_absolute)
]


if args.baseline_type == "absolute":
    absolute_llm_judge = AbsoluteLLMJudge(
        dataset=transformed_dataset, 
        rubrics=rubrics, 
        output_file=args.output_file, 
        repo_name=args.model_repo, 
        min_score=1, 
        max_score=5,
        processed_indices={},
        parse_feedback_and_score=parse_feedback_and_score_func
    )
    absolute_llm_judge.generate_inference_file_pair()

else: 
    relative_llm_judge = RelativeLLMJudge(
        dataset=offset_bias_transformed_dataset_relative, 
        rubrics=rubrics, 
        output_file=args.output_file, 
        repo_name=args.model_repo,
        processed_indices={},
        parse_feedback_and_score= parse_feedback_and_score_func
    )
    relative_llm_judge.generate_inference_file() 