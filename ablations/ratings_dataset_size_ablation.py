import argparse
import os
import csv
from multiprocessing import Pool
from pathlib import Path
from regressors.delta_multinomial import DeltaMultinomial
from regressors.delta_ls import DeltaLS
from regressors.llm import LLMRegressor
import multiprocessing

def save_results(results, model_name, experiment_folder, dataset_size):
    experiment_folder = Path(experiment_folder)
    experiment_folder.mkdir(parents=True, exist_ok=True)
    csv_path = os.path.join(experiment_folder, "dataset_size_experiment_results.csv")
    
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Model", "Dataset Size"] + list(results.keys()))
        writer.writerow([model_name, dataset_size] + list(results.values()))

def run_experiment_for_size(args_tuple):
    args, dataset_size = args_tuple
    
    models = {
        "LS": DeltaLS,
        "Multinomial": DeltaMultinomial,
        "Delta LS": DeltaLS,
        "Delta Multinomial": DeltaMultinomial,
    }

    experiment_folder = Path(args.experiment_folder)
    experiment_folder.mkdir(parents=True, exist_ok=True)
    csv_path = os.path.join(experiment_folder, "dataset_size_experiment_results.csv")

    for model_name, model_class in models.items():
        model = model_class(
            args.train_path,
            args.test_path,
            args.train_emb_path,
            args.test_emb_path,
            size=dataset_size,
            use_external_bias=True if 'Delta' in model_name else False,
            seed=args.seed
        )
        results = model.experiment()
        
        print(f"{model_name} (Dataset Size: {dataset_size}%) Experiment Results:")
        for metric, value in results.items():
            if isinstance(value, (int, float)):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
        
        save_results(results, model_name, args.experiment_folder, dataset_size)

def main():
    parser = argparse.ArgumentParser(description="Run Ratings Dataset experiment.")
    parser.add_argument("--train_path", type=str, help="Path to the training data CSV file.")
    parser.add_argument("--test_path", type=str, help="Path to the testing data CSV file.")
    parser.add_argument("--train_emb_path", type=str, help="Path to the training embeddings .npy file.")
    parser.add_argument("--test_emb_path", type=str, help="Path to the testing embeddings .npy file.")
    parser.add_argument("--experiment_folder", type=str, help="Folder to save experiment results.")
    parser.add_argument("--dataset_size", type=int, default=None, help="Percentage of dataset to use (10–100).")
    parser.add_argument("--seed", type=int, default=2, help="Seed.")

    args = parser.parse_args()

    sizes_to_run = [args.dataset_size] if args.dataset_size else list(range(10, 101, 10))
    
    args_tuples = [(args, size) for size in sizes_to_run]
    num_workers = multiprocessing.cpu_count()-1
    with Pool(processes=num_workers) as pool:
        pool.map(run_experiment_for_size, args_tuples)

if __name__ == "__main__":
    main()
