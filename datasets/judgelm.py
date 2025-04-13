import sys
import os
import argparse
from datasets import load_dataset
from baseline.absolute_llm_judge import AbsoluteLLMJudge
from baseline.relative_llm_judge import RelativeLLMJudge

dataset = load_dataset("tusharbudhwani/JudgeLM")
# dataset["test"] = dataset["test"].select(range(10))
# dataset["train"] = dataset["train"].select(range(10))

parser = argparse.ArgumentParser(description="Run Absolute LLM Judge on Helpsteer Dataset")
# parser.add_argument("--dataset_fold", type=str, default="test", help="Dataset split to use (e.g., test, validation)")
parser.add_argument("--output_file", type=str, required=True, help="Path to save the output CSV file")
parser.add_argument("--model_repo", type=str, required=True, help="Path to the model repository")
parser.add_argument("--baseline_type", type=str, required=True, help="Baseline type: absolute or relative")
args = parser.parse_args()

rubrics = """
Does the model provide relevant and useful responses to the user's needs or questions?
"""

def transform_data(example):
    return {
        "instruction": f"{example["instruction"]}",
        "response1": example["response1"],
        "response2": example["response2"],
        "human_score": example["human_score"]
    }

transformed_dataset = dataset["train"].map(transform_data)

if args.baseline_type == "absolute":
    absolute_llm_judge = AbsoluteLLMJudge(
        dataset=transformed_dataset, 
        rubrics=rubrics, 
        output_file=args.output_file, 
        repo_name=args.model_repo, 
        min_score=1, 
        max_score=10
    )
    absolute_llm_judge.generate_inference_file_pair()
else:
    relative_llm_judge = RelativeLLMJudge(
        dataset=transformed_dataset, 
        rubrics=rubrics, 
        output_file=args.output_file, 
        repo_name=args.model_repo
    )
    relative_llm_judge.generate_inference_file()
    


