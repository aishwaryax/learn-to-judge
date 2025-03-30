import sys
import os
import argparse
from datasets import load_dataset
from baseline.absolute_llm_judge_v2 import AbsoluteLLMJudge
from transformers import AutoTokenizer
import re

parser = argparse.ArgumentParser(description="Run Absolute LLM Judge on Summarization Dataset")
parser.add_argument("--dataset_fold", type=str, default="test", help="Dataset split to use (e.g., test, validation)")
parser.add_argument("--output_file", type=str, required=True, help="Path to save the output CSV file")
parser.add_argument("--model_repo", type=str, required=True, help="Path to the model repository")
args = parser.parse_args()

dataset = load_dataset("openai/summarize_from_feedback", "axis")
# dataset["test"] = dataset["test"].select(range(10))
# dataset["validation"] = dataset["validation"].select(range(10))
min_score = 1
max_score = 7

rubrics = """
[How good is the summary overall at representing the post? If it's hard to find ways to make the summary better, give the summary a high score. If there are lots of different ways the summary can be made better, give the summary a low score. 
Judge on the following criteria while giving the feedback:

Essence: is the summary a good representation of the post?,
Clarity: is the summary reader-friendly? Does it express ideas clearly?
Accuracy: does the summary contain the same information as the longer post?
Purpose: does the summary serve the same purpose as the original post?
Concise: is the summary short and to-the-point?
Style: is the summary written in the same style as the original post?

While giving score, you can refer the following scoring rubrics. Try to interpolate to scores of 2, 3, 5 and 6 as those are not mentioned. You can only give a single value for overall score.
Score of 1: The summary is terrible.
Score of 4: The summary is an okay representation of the post, but could be significantly improved.
Score of 7: The summary is an excellent representation of the post.
"""

instruction_text = "Read the given text and provide a summary.\nText: "
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
    instruction = f"{instruction_text}{example['info']['post'] if example['info']['post'] is not None else example['info']['article']}"
    response = example["summary"]["text"]
    safe_instruction = instruction.replace("{", "{{").replace("}", "}}")
    safe_response = response.replace("{", "{{").replace("}", "}}")
    return {
        "instruction": instruction,
        "prompt": get_llm_prompt(safe_instruction, safe_response),
        "response": response,
        "human_score": example["summary"]["axes"]["overall"]
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
    
def parse_feedback_and_score_llama(text):
    score_match = re.search(r"Overall Score:\s*(\d+)", text)
    score = int(score_match.group(1)) if score_match else None
    reasoning_match = re.search(r"(.*?)Overall Score:\s*\d+\s*(.*)", text, re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip() + "\n" + reasoning_match.group(2).strip()
    else:
        reasoning = text.strip()
    return reasoning, score


def get_parser():
    if 'prometheus' in args.model_repo:
        return parse_feedback_and_score_prometheus
    elif 'llama' in args.model_repo:
        return parse_feedback_and_score_llama

summarization_transformed_dataset = dataset[args.dataset_fold].map(
    transform_data, 
    remove_columns=[col for col in dataset[args.dataset_fold].column_names if col not in ["text", "resp", "human_score", "article"]]
)

parse_feedback_and_score_func = get_parser()

absolute_llm_judge = AbsoluteLLMJudge(
    dataset=summarization_transformed_dataset, 
    rubrics=rubrics, 
    output_file=args.output_file, 
    repo_name=args.model_repo, 
    min_score=1, 
    max_score=7,
    parse_feedback_and_score=parse_feedback_and_score_func
)
absolute_llm_judge.generate_inference_file()
