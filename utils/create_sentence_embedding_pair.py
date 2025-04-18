import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
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

def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute sentence-transformer embeddings and logits for critique-response pairs.")
    parser.add_argument("--model_repo", type=str, required=True, help="Path to the pre-trained model.")
    parser.add_argument("--input_file", type=str, required=True, help="Input CSV file.")
    parser.add_argument("--output_prefix", type=str, required=True, help="Prefix for output files.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for sentence embeddings.")
    args = parser.parse_args()

    input_dir = os.path.dirname(os.path.abspath(args.input_file))
    output_prefix = os.path.join(input_dir, args.output_prefix)

    # Load models
    # tokenizer, model = load_model(args.model_repo)
    embedding_model = load_embedding_model()

    # Load and clean data
    df = pd.read_csv(args.input_file)
    df = df[
        df["llm_critique1"].notna() & (df["llm_critique1"].str.strip() != "") &
        df["llm_critique2"].notna() & (df["llm_critique2"].str.strip() != "") &
        df["llm_response1"].notna() & (df["llm_response1"].str.strip() != "") &
        df["llm_response2"].notna() & (df["llm_response2"].str.strip() != "") &
        pd.to_numeric(df["llm_score1"], errors='coerce').astype('Int64').notna() &
        pd.to_numeric(df["llm_score2"], errors='coerce').astype('Int64').notna()
    ]

    # Compute sentence embeddings using sentence-transformers
    print("Encoding critique1...")
    embeddings_critique1 = embedding_model.encode(
        df["llm_critique1"].tolist(),
        batch_size=args.batch_size,
        show_progress_bar=True
    )

    print("Encoding critique2...")
    embeddings_critique2 = embedding_model.encode(
        df["llm_critique2"].tolist(),
        batch_size=args.batch_size,
        show_progress_bar=True
    )


    # Save everything
    np.save(f"{output_prefix}_critique1_sentence_embeddings.npy", embeddings_critique1)
    np.save(f"{output_prefix}_critique2_sentence_embeddings.npy", embeddings_critique2)

    df["embedding_index_critique1"] = np.arange(len(df))
    df["embedding_index_critique2"] = np.arange(len(df))

    df.to_csv(f"{output_prefix}_with_scores_sentence_embeddings.csv", index=False)

    print(f"Embeddings saved to {output_prefix}_critique1_sentence_embeddings.npy and {output_prefix}_critique2_sentence_embeddings.npy")
    print(f"Updated CSV saved to {output_prefix}_with_scores_sentence_embeddings.csv")
