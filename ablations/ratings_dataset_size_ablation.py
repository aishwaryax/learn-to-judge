import argparse
import os
import csv
from multiprocessing import Pool
from pathlib import Path
import multiprocessing
from regressors.delta_multinomial import DeltaMultinomial
from regressors.delta_ls import DeltaLS
from regressors.llm import LLMRegressor


def is_result_already_present(csv_path, model_name, size, seed):
    if not os.path.exists(csv_path):
        return False
    with open(csv_path, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if (
                row["Model"] == model_name
                and int(row["Dataset Size"]) == size
                and int(row["Seed"]) == seed
            ):
                return True
    return False

def save_results(results, model_name, experiment_folder, dataset_size, seed):
    experiment_folder = Path(experiment_folder)
    experiment_folder.mkdir(parents=True, exist_ok=True)
    csv_path = os.path.join(experiment_folder, "dataset_size_experiment_results.csv")

    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Model", "Dataset Size", "Seed"] + list(results.keys()))
        writer.writerow([model_name, dataset_size, seed] + list(results.values()))

def run_all_models_for_seed(args_tuple):
    args, seed = args_tuple
    csv_path = os.path.join(args.experiment_folder, "dataset_size_experiment_results.csv")
    models = {
        "LS": DeltaLS,
        "Multinomial": DeltaMultinomial,
        "Delta LS": DeltaLS,
        "Delta Multinomial": DeltaMultinomial,
    }

    dataset_sizes = list(range(1, 10))
    dataset_sizes = dataset_sizes + list(range(10, 101, 10))

    for dataset_size in dataset_sizes:
        for model_name, model_class in models.items():
            if is_result_already_present(csv_path, model_name, dataset_size, seed):
                print(f"Skipping {model_name} | Seed: {seed} | Size: {dataset_size}% (already present)")
                continue
            model = model_class(
                args.train_path,
                args.test_path,
                args.train_emb_path,
                args.test_emb_path,
                size=dataset_size,
                use_external_bias=True if 'Delta' in model_name else False,
                seed=seed
            )
            results = model.experiment()
            print(f"{model_name} | Seed: {seed} | Size: {dataset_size}%")
            for metric, value in results.items():
                if isinstance(value, (int, float)):
                    print(f"{metric}: {value:.4f}")
                else:
                    print(f"{metric}: {value}")
            save_results(results, model_name, args.experiment_folder, dataset_size, seed)

def main():
    parser = argparse.ArgumentParser(description="Run all models across multiple seeds.")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--train_emb_path", type=str, required=True)
    parser.add_argument("--test_emb_path", type=str, required=True)
    parser.add_argument("--experiment_folder", type=str, required=True)

    args = parser.parse_args()

    seeds = list(range(10))  # Seeds: 0 to 9
    args_tuples = [(args, seed) for seed in seeds]

    num_workers = min(len(seeds), multiprocessing.cpu_count() - 1)
    with Pool(processes=num_workers) as pool:
        pool.map(run_all_models_for_seed, args_tuples)

if __name__ == "__main__":
    main()
