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

# def load_model(model_path):
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
#     model.eval()
#     return tokenizer, model


def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

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

    embedding_model = load_embedding_model()

    # Read and filter input
    df = pd.read_csv(args.input_file)
    df = df[
        df["llm_response"].notna() & 
        (df["llm_response"].str.strip() != "") & 
        df["llm_critique"].notna() & 
        pd.to_numeric(df["llm_score"], errors='coerce').notna()
    ]

    # Compute critique embeddings using all-MiniLM
    embeddings_critique = embedding_model.encode(
        df["llm_critique"].tolist(),
        batch_size=args.batch_size,
        show_progress_bar=True
    )

    # Save embeddings and updated CSV
    df["embedding_index_critique"] = np.arange(len(df))
    np.save(f"{output_prefix}_critique_sentence_embeddings.npy", embeddings_critique)
    df.to_csv(f"{output_prefix}_with_scores_sentence_embeddings.csv", index=False)

    print(f"Embeddings saved to {output_prefix}_critique_sentence_embeddings.npy")
    print(f"Updated CSV with scores saved to {output_prefix}_with_scores_sentence_embeddings.csv")
