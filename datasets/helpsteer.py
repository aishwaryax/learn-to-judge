import sys
import os
import argparse
from datasets import load_dataset
from baseline.absolute_llm_judge import AbsoluteLLMJudge

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
[How useful and helpful the overall response is?]
Score of 0: The response is not useful or helpful at all. The response completely missed the essence of what the user wanted.
Score of 1: The response is borderline unhelpful and mostly does not capture what the user was looking for, but is still usable and helpful in a small way.
Score of 2: The response is partially helpful but misses the overall goal of the user's query/input in some way. The response did not fully satisfy what the user was looking for.
Score of 3: The response is mostly helpful and mainly aligned with what the user was looking for, but there is still some room for improvement.
Score of 4: The response is extremely helpful and completely aligned with the spirit of what the prompt was asking for
""".strip()

instruction_text = """You are a helpful and general purpose assistant. Answer the user query.
<user>{}</user>"""

def transform_data(example):
    return {
        "instruction": instruction_text.format(example["prompt"]),
        "response": example["response"],
        "human_score": example["helpfulness"]
    }

steerlm_transformed_dataset = dataset[args.dataset_fold].map(transform_data, remove_columns=[col for col in dataset[args.dataset_fold].column_names if col not in ["text", "resp", "human_score", "article"]])
absolute_llm_judge = AbsoluteLLMJudge(
    dataset=steerlm_transformed_dataset, 
    rubrics=rubrics, 
    output_file=args.output_file, 
    repo_name=args.model_repo, 
    min_score=0, 
    max_score=4
)
absolute_llm_judge.generate_inference_file()
