import argparse
import subprocess
import os
import pandas as pd
import re
from pathlib import Path
import csv
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from scipy.stats import pearsonr, spearmanr, kendalltau



def compute_metrics_from_csv(csv_path: str) -> dict:
    """
    Reads inference output CSV and computes evaluation metrics.
    Assumes CSV has columns 'human_score' and 'predicted_score'.
    """
    df = pd.read_csv(csv_path)
    # Fill missing predicted_score if any
    df['predicted_score'] = df['predicted_score'].fillna(0).astype(int)
    # Only evaluate where human_score exists
    df = df.dropna(subset=['human_score'])
    y_true = df['human_score'].astype(int)
    y_pred = df['predicted_score'].astype(int)

    metrics = {
        'MSE': mean_squared_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'Accuracy': accuracy_score(y_true, y_pred),
    }
    # correlations
    if len(y_true) > 1:
        metrics['Pearson r'], _ = pearsonr(y_true, y_pred)
        metrics['Spearman ρ'], _ = spearmanr(y_true, y_pred)
        metrics['Kendall τ'], _ = kendalltau(y_true, y_pred)
    else:
        metrics['Pearson r'] = metrics['Spearman ρ'] = metrics['Kendall τ'] = float('nan')
    return metrics


def run_for_seed(args_tuple):
    args, seed = args_tuple

    # Load full dataset
    df = pd.read_csv(args.dataset_path)
    total = len(df)

    # Define percentages to sample
    dataset_sizes = [1, 2, 5, 10]

    # Prepare metrics directory
    metrics_dir = Path(args.output_base) / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)

    for size in dataset_sizes:
        # Sample subset
        n = max(1, int(total * size / 100.0))
        subset = df.sample(n, random_state=seed)

        # Save subset
        subset_dir = Path(args.output_base) / f"subset_{size}_seed_{seed}"
        subset_dir.mkdir(parents=True, exist_ok=True)
        subset_path = subset_dir / 'data.csv'
        subset.to_csv(subset_path, index=False)

        # Prepare model output dir
        save_dir = Path(args.output_base) / f"model_{size}_seed_{seed}"
        save_dir.mkdir(parents=True, exist_ok=True)

        # 1) Train
        train_cmd = [
            'python', args.train_script,
            '--model_repo', args.model_repo,
            '--dataset_path', str(subset_path),
            '--save_path', str(save_dir),
            '--epochs', str(args.epochs),
            '--batch_size', str(args.batch_size),
            '--lr', str(args.lr),
            '--reg_lambda', str(args.reg_lambda),
        ]
        print(f"[Seed {seed}] Training with {size}% data...")
        subprocess.run(train_cmd, check=True)

        # 2) Inference
        tmp_pred_csv = metrics_dir / f'preds_{size}_{seed}.csv'
        infer_cmd = [
            'python', args.inference_script,
            '--model_dir', str(save_dir) + '/sft',
            '--test_csv', args.testset_path,
            '--output_csv', tmp_pred_csv,
            '--batch_size', str(args.batch_size*4)
        ]
        print(f"[Seed {seed}] Inference for {size}% data...")
        proc = subprocess.run(infer_cmd, capture_output=True, text=True)
        print(proc.stdout)
        # 3) Compute metrics from CSV
        metrics = compute_metrics_from_csv(tmp_pred_csv)

        # 3) Save metrics
        metrics_file = metrics_dir / f'metrics.csv'
        write_header = not metrics_file.exists()
        with open(metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['Model'] + ['Dataset Size'] + ['Seed'] + list(metrics.keys()))
            writer.writerow([seed] + list(metrics.values()))


def main():
    parser = argparse.ArgumentParser(
        description="Run training and inference ablation over dataset sizes and seeds."
    )
    parser.add_argument('--model_repo', type=str, required=True,
                        help='Path to pretrained model repo')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Full CSV dataset path')
    parser.add_argument('--testset_path', type=str, required=True,
                        help='CSV file for inference/testing')
    parser.add_argument('--output_base', type=str, required=True,
                        help='Base dir for subsets, models, and metrics')
    parser.add_argument('--train_script', type=str, default='train/sft_quantized_absolute.py',
                        help='Path to training script')
    parser.add_argument('--inference_script', type=str, default='train/inference_absolute.py',
                        help='Path to inference script')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for both train/infer')
    parser.add_argument('--lr', type=float, default=5e-6,
                        help='Learning rate')
    parser.add_argument('--reg_lambda', type=float, default=1e-4,
                        help='Regularization lambda')
    args = parser.parse_args()

    # Setup seeds
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 0–9
    args_tuples = [(args, seed) for seed in seeds]

    # Run sequentially
    for args_tuple in args_tuples:
        run_for_seed(args_tuple)


if __name__ == '__main__':
    main()
