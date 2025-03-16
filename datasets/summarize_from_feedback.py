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
These are just reference rubrics, and please interpolate the score with respect to these rubrics.]
Score of 1: The summary is extremely vague or irrelevant, failing to capture key aspects of the original text. Important information is missing, or the summary is disconnected from the main content.
Score of 2: The summary includes some relevant points but omits crucial details or misrepresents the central themes of the original text. It may be overly generalized or inaccurate.
Score of 3: The summary captures the main idea but lacks depth or omits significant context. Some details may be misrepresented or simplified, leading to a partial understanding of the original text.
Score of 4: The summary is mostly accurate, capturing the main points and themes with only minor omissions or inaccuracies. It reflects a general understanding of the text but may lack clarity or completeness in certain areas.
Score of 5: The summary is clear, accurate, and captures the key elements of the text. It includes the central ideas, though some minor details may be left out. Overall, it demonstrates a good understanding of the content.
Score of 6: The summary is detailed, well-organized, and accurately reflects the main points and supporting details of the text. It may omit very minor information but conveys the essence of the content effectively.
Score of 7: The summary is comprehensive and insightful, capturing all major ideas and details in a concise and coherent manner. It accurately reflects the structure and tone of the original text while maintaining clarity and precision.
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
