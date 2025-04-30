# run_inference_with_metrics.py
# Usage:
# python run_inference_with_metrics.py \
#   --model_dir ./final_model \
#   --test_csv test_data.csv \
#   --output_csv predictions.csv \
#   --batch_size 16

import argparse
import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from scipy.stats import pearsonr, spearmanr, kendalltau

# Prompt template and rubric (same as training)
rubrics = """
[Helpfulness can be measured by how useful and helpful the overall response is.
Score of 0: The response is not useful or helpful at all. The response completely missed the essence of what the user wanted.
Score of 1: The response is borderline unhelpful and mostly does not capture what the user was looking for, but is still usable and helpful in a small way.
Score of 2: The response is partially helpful but misses the overall goal of the user's query/input in some way. The response did not fully satisfy what the user was looking for.
Score of 3: The response is mostly helpful and mainly aligned with what the user was looking for, but there is still some room for improvement.
Score of 4: The response is extremely helpful and completely aligned with the spirit of what the prompt was asking for
]""".strip()

ABSOLUTE_PROMPT = """###Task Description:
An instruction, a response, and a score rubric are given.
1. Write detailed feedback based on the rubric.
2. Then write a score between {min_score} and {max_score}.
3. Output format: an integer between {min_score} and {max_score}.

###Instruction:
{instruction}

###Response:
{response}

###Score Rubrics:
{rubrics}

###Score:"""


def build_prompt(instruction: str, response: str, min_score=0, max_score=4):
    prompt = ABSOLUTE_PROMPT.format(
        instruction=instruction,
        response=response,
        rubrics=rubrics,
        min_score=min_score,
        max_score=max_score,
    )
    return prompt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with fine-tuned scoring model"
    )
    parser.add_argument(
        "--model_dir", type=str, required=True,
        help="Path to the merged fine-tuned model directory"
    )
    parser.add_argument(
        "--test_csv", type=str, required=True,
        help="CSV file with columns: instruction,response[,human_score]"
    )
    parser.add_argument(
        "--output_csv", type=str, required=True,
        help="Where to write the results (with predicted_score and metrics)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for inference"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    # Read test data
    df = pd.read_csv(args.test_csv)
    assert "instruction" in df.columns and "response" in df.columns, \
        "CSV must have 'instruction' and 'response' columns"

    # Build prompts
    prompts = [
        build_prompt(ins, rsp) for ins, rsp in zip(df["instruction"], df["response"])
    ]

    # Inference
    preds = []
    batch_size = args.batch_size
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i: i + batch_size]
            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(model.device)

            out = model.generate(
                **enc,
                max_new_tokens=2,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

            for prompt, seq in zip(batch_prompts, out):
                text = tokenizer.decode(seq, skip_special_tokens=True)
                completion = text[len(prompt):].strip().split()[0]
                try:
                    score = int(completion)
                except ValueError:
                    score = None
                preds.append(score)

    df["predicted_score"] = preds

    # Evaluation metrics
    if "human_score" in df.columns:
        df_eval = df.dropna(subset=["predicted_score", "human_score"]).copy()
        y_true = df_eval["human_score"].astype(int)
        y_pred = df_eval["predicted_score"].astype(int)

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        pearson_corr, _ = pearsonr(y_true, y_pred)
        spearman_corr, _ = spearmanr(y_true, y_pred)
        kendall_corr, _ = kendalltau(y_true, y_pred)

        print("=== Evaluation Metrics ===")
        print(f"MSE            : {mse:.4f}")
        print(f"MAE            : {mae:.4f}")
        print(f"Accuracy       : {acc:.2%}")
        print(f"Pearson r      : {pearson_corr:.4f}")
        print(f"Spearman rho   : {spearman_corr:.4f}")
        print(f"Kendall tau    : {kendall_corr:.4f}")
    
    # Save results
    df.to_csv(args.output_csv, index=False)
    print(f"Saved predictions and metrics to {args.output_csv}")


if __name__ == "__main__":
    main()
