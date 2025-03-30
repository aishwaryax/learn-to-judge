import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import numpy as np
import argparse
import os

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()
    return tokenizer, model

def get_embeddings(texts, tokenizer, model, batch_size=8):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        tokens = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)

        last_hidden_state = outputs.hidden_states[-1]
        batch_embeddings = last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(batch_embeddings)

    return np.vstack(embeddings)

def compute_scores(question, response, targets, tokenizer, model, top_k=100):
    input_text = f"{question}{response}"
    input_tokens = tokenizer(input_text, return_tensors="pt").to(device)
    input_ids = input_tokens["input_ids"]
    
    question_tokens = tokenizer(question, return_tensors="pt")["input_ids"]
    response_tokens = tokenizer(response, return_tensors="pt")["input_ids"]
    
    question_length = question_tokens.shape[1]
    response_start = question_length
    response_end = input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model(**input_tokens)
        logits = outputs.logits
        
    shifted_logits = logits[:, :-1, :]
    shifted_input_ids = input_ids[:, 1:]

    response_logits = shifted_logits[0, response_start:response_end, :]
    response_token_ids = shifted_input_ids[0, response_start:response_end]
    
    response_log_probs = -torch.nn.functional.cross_entropy(
        response_logits, response_token_ids, reduction='none'
    )

    self_consistency_score = torch.exp(response_log_probs.mean()).item()
    
    target_probabilities = {}

    for target in targets:
        target_token_id = tokenizer(target, add_special_tokens=False)["input_ids"][-1]
        target_token_index = (response_token_ids == target_token_id).nonzero(as_tuple=True)

        if target_token_index[0].numel() > 0:
            target_index = target_token_index[0][-1].item()
            target_logit = response_logits[target_index, :]
            target_prob = torch.nn.functional.softmax(target_logit, dim=-1)[target_token_id].item()
            target_probabilities[target] = target_prob
        else:
            target_probabilities[target] = 0.0

    return target_probabilities, self_consistency_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute embeddings and logits for critique-response pairs using a transformer model.")
    parser.add_argument("--model_repo", type=str, required=True, help="Path to the pre-trained model.")
    parser.add_argument("--input_file", type=str, required=True, help="Input CSV file.")
    parser.add_argument("--output_prefix", type=str, required=True, help="Prefix for output files.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing inputs.")
    args = parser.parse_args()

    input_dir = os.path.dirname(os.path.abspath(args.input_file))
    output_prefix = os.path.join(input_dir, args.output_prefix)
    tokenizer, model = load_model(args.model_repo)
    
    df = pd.read_csv(args.input_file)
    df = df[
        df["llm_critique1"].notna() & (df["llm_critique1"].str.strip() != "") &
        df["llm_critique2"].notna() & (df["llm_critique2"].str.strip() != "") &
        df["llm_response1"].notna() & (df["llm_response1"].str.strip() != "") &
        df["llm_response2"].notna() & (df["llm_response2"].str.strip() != "")
        pd.to_numeric(df["llm_score"], errors='coerce').notna()
    ]

    embeddings_critique1 = get_embeddings(df["llm_critique1"].tolist(), tokenizer, model, args.batch_size)
    embeddings_critique2 = get_embeddings(df["llm_critique2"].tolist(), tokenizer, model, args.batch_size)

    df["target_probability1"], df["self_consistency_score1"] = zip(*df.apply(
            lambda row: compute_scores(
                row["llm_prompt1"], row["llm_response1"], targets, tokenizer, model
            ), axis=1
        ))
    df["target_probability2"], df["self_consistency_score2"] = zip(*df.apply(
            lambda row: compute_scores(
                row["llm_prompt2"], row["llm_response2"], targets, tokenizer, model
            ), axis=1
        ))
    np.save(f"{args.output_prefix}_critique1.npy", embeddings_critique1)
    np.save(f"{args.output_prefix}_critique2.npy", embeddings_critique2)

    df["embedding_index_critique1"] = np.arange(len(df))
    df["embedding_index_critique2"] = np.arange(len(df))

    df.to_csv(f"{args.output_prefix}_with_scores_embeddings.csv", index=False)

    print(f"Generated embeddings saved to {args.output_prefix}_critique1.npy and {args.output_prefix}_critique2.npy")
    print(f"Generated logits saved to {args.output_prefix}_response1_logits.npy and {args.output_prefix}_response2_logits.npy")
    print(f"Updated CSV saved to {args.output_prefix}.csv")
