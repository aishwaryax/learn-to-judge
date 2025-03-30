import sys
import os
import argparse
from datasets import load_dataset
from baseline.absolute_llm_judge_v2 import AbsoluteLLMJudge
from transformers import AutoTokenizer
import re

#for later add a parameter for axis

parser = argparse.ArgumentParser(description="Run Absolute LLM Judge on Helpsteer Dataset")
parser.add_argument("--dataset_fold", type=str, default="test", help="Dataset split to use (e.g., test, validation)")
parser.add_argument("--output_file", type=str, required=True, help="Path to save the output CSV file")
parser.add_argument("--model_repo", type=str, required=True, help="Path to the model repository")
args = parser.parse_args()

dataset = load_dataset("nvidia/HelpSteer2")
# dataset["train"] = dataset["train"].select(range(10))
# dataset["validation"] = dataset["validation"].select(range(10))

rubrics = """
[Helpfulness can be measured by how useful and helpful the overall response is.
While giving score, you can refer the following scoring rubrics. You can only give a single value for the resulting score.]
Score of 0: The response is not useful or helpful at all. The response completely missed the essence of what the user wanted.
Score of 1: The response is borderline unhelpful and mostly does not capture what the user was looking for, but is still usable and helpful in a small way.
Score of 2: The response is partially helpful but misses the overall goal of the user's query/input in some way. The response did not fully satisfy what the user was looking for.
Score of 3: The response is mostly helpful and mainly aligned with what the user was looking for, but there is still some room for improvement.
Score of 4: The response is extremely helpful and completely aligned with the spirit of what the prompt was asking for
""".strip()

instruction_text = """You are a helpful and general purpose assistant. Answer the user query.
<user>{}</user>"""

min_score = 0
max_score = 4
tokenizer = AutoTokenizer.from_pretrained(args.model_repo)

def get_llm_prompt(instruction, response):
    if 'prometheus' in args.model_repo or 'llama' in args.model_repo:
        messages = [
        {"role": "system", "content": "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."},
        {"role": "user", "content": f"""###Task Description:
        An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
        1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
        2. After writing a feedback, write a score that is an integer between {min_score} and {max_score}. You should refer to the score rubric.
        3. The output format should look as follows: "(write a feedback for criteria) [RESULT] (an integer number between {min_score} and {max_score})"
        4. Please do not generate any other opening, closing, and explanations.

        ###The instruction to evaluate:
        {instruction}

        ###Response to evaluate:
        {response}

        ###Score Rubrics:
        {rubrics}

        ###Feedback: """}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)  

def transform_data(example):
    instruction = instruction_text.format(example["prompt"])
    response = example["response"]
    return {
        "instruction": instruction,
        "response": response,
        "prompt": get_llm_prompt(instruction, response),
        "human_score": example["helpfulness"]
    }

def parse_feedback_and_score_prometheus(text):
    result_match = re.search(r"\[RESULT\]\s*(\d+)", text)

    if result_match:
        score = int(result_match.group(1))
        feedback = text[:result_match.start()].strip()
    else:
        score = None
        feedback = text.strip()
    return feedback, score

def get_parser():
    if 'prometheus' in args.model_repo or 'llama' in args.model_repo:
        return parse_feedback_and_score_prometheus

parse_feedback_and_score_func = get_parser()

steerlm_transformed_dataset = dataset[args.dataset_fold].map(transform_data, remove_columns=[col for col in dataset[args.dataset_fold].column_names if col not in ["text", "resp", "human_score", "article"]])
absolute_llm_judge = AbsoluteLLMJudge(
    dataset=steerlm_transformed_dataset, 
    rubrics=rubrics, 
    output_file=args.output_file, 
    repo_name=args.model_repo, 
    min_score=0, 
    max_score=4,
    parse_feedback_and_score=parse_feedback_and_score_func
)
absolute_llm_judge.generate_inference_file()
