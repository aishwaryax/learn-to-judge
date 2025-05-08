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
from rubric_config import RUBRIC_CONFIG
import re
import random

# fix the seed if you want reproducible “random” fills
random.seed(42)


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

def get_rubric_and_scores(dataset_path):
    if "helpsteer" in dataset_path.lower():
        return RUBRIC_CONFIG["helpsteer"]
    elif "summarize_from_feedback" in dataset_path.lower():
        return RUBRIC_CONFIG["summarize_from_feedback"]
    else:
        raise ValueError("Unknown dataset type in path.")


def build_prompt(instruction: str, response: str, rubric_config: dict):
    prompt = ABSOLUTE_PROMPT.format(
        instruction=instruction,
        response=response,
        rubrics=rubric_config["rubric"],
        min_score=rubric_config["min_score"],
        max_score=rubric_config["max_score"],
    )
    # print(f"Prompt: {prompt}")
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
        "--batch_size", type=int, default=64,
        help="Batch size for inference"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Load tokenizer and model

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, padding_side="left")
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
    
    rubric_config = get_rubric_and_scores(args.test_csv)

    # Build prompts
    prompts = [
        build_prompt(ins, rsp, rubric_config) for ins, rsp in zip(df["instruction"], df["response"])
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

            
            for j, seq in enumerate(out):
                input_len = int(enc["attention_mask"][j].sum().item())
                gen_text  = tokenizer.decode(
                    seq, skip_special_tokens=True).strip()

                m = re.search(r"(?:\#{1,3}\s*)?Score\s*[：:]\s*(\d+)\b", gen_text, re.IGNORECASE)
                if not m:
                    # Fallback to looser pattern
                    m = re.search(r"Score\D*(\d+)", gen_text, re.IGNORECASE)
                if m:                                    # only if we got a match
                    score = int(m.group(1))               # group() → matched text → int
                    # print(score)
                else:
                    print("no number found:", gen_text)   # handle missing scores
                    print("score = ", score)
                preds.append(int(m.group(1)) if m else None)

    df["predicted_score"] = preds

    # Evaluation metrics
    if "human_score" in df.columns:
        df['predicted_score'] = df['predicted_score'].apply(
        lambda x: x if pd.notnull(x) else random.randint(rubric_config["min_score"], rubric_config["max_score"]))
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
