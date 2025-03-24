import sys
import os
import argparse
from datasets import load_dataset
from baseline.absolute_llm_judge import AbsoluteLLMJudge

parser = argparse.ArgumentParser(description="Run Absolute LLM Judge on Summarization Dataset")
parser.add_argument("--dataset_fold", type=str, default="test", help="Dataset split to use (e.g., test, validation)")
parser.add_argument("--output_file", type=str, required=True, help="Path to save the output CSV file")
parser.add_argument("--model_repo", type=str, required=True, help="Path to the model repository")
args = parser.parse_args()

dataset = load_dataset("openai/summarize_from_feedback", "axis")
# dataset["test"] = dataset["test"].select(range(10))
# dataset["validation"] = dataset["validation"].select(range(10))

rubrics = """
[How good is the summary overall at representing the post? If it's hard to find ways to make the summary better, give the summary a high score. If there are lots of different ways the summary can be made better, give the summary a low score.
These are just reference rubrics, and please interpolate the score with respect to these rubrics.
Score of 1: The summary is terrible.
Score of 4: The summary is an okay representation of the post, but could be significantly improved.
Score of 7: The summary is an excellent representation of the post.
"""

instruction_text = "Read the given text and provide a summary.\nText: "

def transform_data(example):
    return {
        "instruction": f"{instruction_text}{example['info']['post'] if example['info']['post'] is not None else example['info']['article']}",
        "response": example["summary"]["text"],
        "human_score": example["summary"]["axes"]["overall"]
    }

summarization_transformed_dataset = dataset[args.dataset_fold].map(
    transform_data, 
    remove_columns=[col for col in dataset[args.dataset_fold].column_names if col not in ["text", "resp", "human_score", "article"]]
)

absolute_llm_judge = AbsoluteLLMJudge(
    dataset=summarization_transformed_dataset, 
    rubrics=rubrics, 
    output_file=args.output_file, 
    repo_name=args.model_repo, 
    min_score=1, 
    max_score=7
)
absolute_llm_judge.generate_inference_file()
