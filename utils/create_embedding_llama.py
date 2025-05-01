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


def compute_scores(question, response, actual_target, targets, tokenizer, model, top_k=100, dataset_type="absolute"):
    question_ids = tokenizer(question, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)
    response_ids = tokenizer(response, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)

    input_ids = torch.cat([question_ids, response_ids]).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    shifted_logits = logits[:, :-1, :]
    shifted_input_ids = input_ids[:, 1:]

    response_start = question_ids.shape[0] - 1 
    response_logits = shifted_logits[0, response_start:, :]
    response_token_ids = shifted_input_ids[0, response_start:]

    log_probs = -torch.nn.functional.cross_entropy(response_logits, response_token_ids, reduction='none')
    self_consistency_score = torch.exp(log_probs.mean()).item()

    target_token_ids = {
        target: tokenizer(target, add_special_tokens=False)["input_ids"][-1]
        for target in targets
    }

    target_to_index_mapping = {target: i for i, target in enumerate(targets)}
    if dataset_type == "relative":
        if actual_target == "0":
            actual_target = " A"
        elif actual_target == "1":
            actual_target = " B"
    actual_token_id = target_token_ids.get(actual_target, None)
    if actual_token_id is None:
        return {i: 0.0 for i, k in enumerate(targets)}, self_consistency_score

    match_pos = (response_token_ids == actual_token_id).nonzero(as_tuple=True)
    if match_pos[0].numel() == 0:
        return {i: 0.0 for i, k in enumerate(targets)}, self_consistency_score

    target_index = match_pos[0][-1].item()

    target_logit = response_logits[target_index, :]
    probs = torch.nn.functional.softmax(target_logit, dim=-1)

    target_probabilities = {
        target_to_index_mapping[target]: probs[token_id].item()
        for target, token_id in target_token_ids.items()
    }

    total_prob = sum(target_probabilities.values())

    if total_prob > 0:
        target_probabilities = {k: v / total_prob for k, v in target_probabilities.items()}
    else:
        target_probabilities = {k: 0.0 for k in targets}
    return target_probabilities, self_consistency_score




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute embeddings, logits, self-reflection, and self-consistency scores for LLM responses.")
    parser.add_argument("--model_repo", type=str, required=True, help="Path to the pre-trained model.")
    parser.add_argument("--input_file", type=str, required=True, help="Input CSV file.")
    parser.add_argument("--output_prefix", type=str, required=True, help="Prefix for output files.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing inputs.")
    parser.add_argument("--dataset_type", type=str, choices=["absolute", "relative"], default="absolute", help="Type of dataset: 'absolute' or 'relative'.")
    args = parser.parse_args()

    input_dir = os.path.dirname(os.path.abspath(args.input_file))
    output_prefix = os.path.join(input_dir, args.output_prefix)
    tokenizer, model = load_model(args.model_repo)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token else tokenizer.eos_token

    df = pd.read_csv(args.input_file)
    df = df[:10]
    df = df[
        df["llm_response"].notna() & 
        (df["llm_response"].str.strip() != "") & 
        df["llm_critique"].notna() & 
        pd.to_numeric(df["llm_score"], errors='coerce').notna()
    ]
    targets = None
    
    if args.dataset_type == "relative":
        targets = [" A", " B"]
    else:
        targets = sorted(df['llm_score'].unique().astype(int).astype(str))
        
    embeddings_critique = get_embeddings(df["llm_critique"].tolist(), tokenizer, model, args.batch_size)

    df["target_probability"], df["self_consistency_score"] = zip(*df.apply(
        lambda row: compute_scores(
            row["llm_prompt"], row["llm_response"], str(row['llm_score']), targets, tokenizer, model, dataset_type=args.dataset_type
        ), axis=1
    ))

    df["embedding_index_critique"] = np.arange(len(df))
    np.save(f"{output_prefix}_critique_embeddings.npy", embeddings_critique)
    df.to_csv(f"{output_prefix}_with_scores_embeddings.csv", index=False)

    print(f"Embeddings saved to {output_prefix}_critique_embeddings.npy")
    print(f"Updated CSV with scores saved to {output_prefix}_with_scores_embeddings.csv")
