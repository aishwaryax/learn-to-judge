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

def get_logits(texts, tokenizer, model, batch_size=8):
    logits = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        tokens = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = model(**tokens)

        attention_mask = tokens['attention_mask'].unsqueeze(-1).to(device)
        batch_logits = (outputs.logits * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        batch_logits = batch_logits.cpu().numpy()
        logits.append(batch_logits)

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
    ]

    embeddings_critique1 = get_embeddings(df["llm_critique1"].tolist(), tokenizer, model, args.batch_size)
    embeddings_critique2 = get_embeddings(df["llm_critique2"].tolist(), tokenizer, model, args.batch_size)

    logits_response1 = get_logits(df["llm_response1"].tolist(), tokenizer, model, args.batch_size)
    logits_response2 = get_logits(df["llm_response2"].tolist(), tokenizer, model, args.batch_size)

    np.save(f"{args.output_prefix}_critique1.npy", embeddings_critique1)
    np.save(f"{args.output_prefix}_critique2.npy", embeddings_critique2)
    np.save(f"{args.output_prefix}_response1_logits.npy", logits_response1)
    np.save(f"{args.output_prefix}_response2_logits.npy", logits_response2)

    df["embedding_index_critique1"] = np.arange(len(df))
    df["embedding_index_critique2"] = np.arange(len(df))
    df["logits_index_response1"] = np.arange(len(df))
    df["logits_index_response2"] = np.arange(len(df))

    df.to_csv(f"{args.output_prefix}.csv", index=False)

    print(f"Generated embeddings saved to {args.output_prefix}_critique1.npy and {args.output_prefix}_critique2.npy")
    print(f"Generated logits saved to {args.output_prefix}_response1_logits.npy and {args.output_prefix}_response2_logits.npy")
    print(f"Updated CSV saved to {args.output_prefix}.csv")
