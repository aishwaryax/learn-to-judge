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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.stats import pearsonr, spearmanr, kendalltau
import re

# Prompt template and rubric (same as training)
rubrics = """
Does the model provide relevant and useful responses to the user's needs or questions?
""".strip()

RELATIVE_PROMPT_TEMPLATE = """###Task Description:
An instruction (might include an Input inside it), two responses to evaluate (denoted as Response A and Response B), and an evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given evaluation criteria,, not evaluating in general.
2. Make comparisons between Response A and Response B. Instead of examining Response A and Response B separately, go straight to the point and mention about the commonalities and differences between them.
3. After writing the feedback, indicate the better response, either "A" or "B".
4. Please do not generate any other opening, closing, and explanations.

###Instruction:
{instruction}

###Response A:
{response1}

###Response B:
{response2}

###Score Rubric:
{rubrics}

###Result: """


def build_prompt(instruction: str, response1: str,  response2: str):
    prompt = RELATIVE_PROMPT_TEMPLATE.format(
        instruction=instruction,
        response1=response1,
        response2=response2,
        rubrics=rubrics
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
    # tokenizer = AutoTokenizer.from_pretrained(args.model_dir, padding_side="left")
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
    assert "instruction" in df.columns and "response1" in df.columns and "response2" in df.columns, \
        "CSV must have 'instruction' and 'response' columns"


    # Build prompts
    prompts = [
        build_prompt(ins, rsp1, rsp2) for ins, rsp1, rsp2 in zip(df["instruction"], df["response1"], df["response2"])
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
                result_match = re.search(r"Result:\s*([A-Za-z])", gen_text)
                score = None
                if result_match:
                    score_char = result_match.group(1).upper()
                    if score_char in ['A', 'B']:
                        score = 0 if score_char == 'A' else 1
                    else:
                        print("Unexpected result letter:", score_char)
                else:
                    print("No result found:")

                preds.append(score)

    df["predicted_score"] = preds

    # Evaluation metrics
    if "human_score" in df.columns:
        df_eval = df.dropna(subset=["predicted_score", "human_score"]).copy()
        y_true = df_eval["human_score"].astype(int)
        y_pred = df_eval["predicted_score"].astype(int)

        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        pearson_corr, _ = pearsonr(y_true, y_pred)
        spearman_corr, _ = spearmanr(y_true, y_pred)
        kendall_corr, _ = kendalltau(y_true, y_pred)

        print("=== Evaluation Metrics ===")
        print(f"Accuracy       : {acc:.4%}")
        print(f"Precision      : {precision:.4%}")
        print(f"Recall         : {recall:.4%}")
        print(f"F1 Score       : {f1:.4%}")
        print(f"Pearson r      : {pearson_corr:.4f}")
        print(f"Spearman rho   : {spearman_corr:.4f}")
        print(f"Kendall tau    : {kendall_corr:.4f}")
    
    # Save results
    # df.to_csv(args.output_csv, index=False)
    # print(f"Saved predictions and metrics to {args.output_csv}")


if __name__ == "__main__":
    main()
