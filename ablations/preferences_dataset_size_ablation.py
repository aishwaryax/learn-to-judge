import argparse
import os
import csv
from pathlib import Path
from multiprocessing import Pool
from regressors.delta_btl2 import DeltaBTL2
from regressors.delta_multinomial import DeltaMultinomial
from regressors.llm import LLMRegressor
from regressors.llm2 import LLM2Regressor
import multiprocessing


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

    relative_models = {
        "BTL": DeltaMultinomial,
        "Delta BTL": DeltaMultinomial,
    }

    absolute_models = {
        "BTL2": DeltaBTL2,
        "Delta BTL2": DeltaBTL2,
    }

    csv_path = os.path.join(args.experiment_folder, "dataset_size_experiment_results.csv")

    dataset_sizes = list(range(1, 10))
    dataset_sizes = dataset_sizes + list(range(10, 101, 10))

    for size in dataset_sizes:

        for model_name, model_class in relative_models.items():
            if is_result_already_present(csv_path, model_name, size, seed):
                print(f"Skipping {model_name} | Seed: {seed} | Size: {size}% (already present)")
                continue
            model = model_class(
                args.relative_train_path,
                args.relative_test_path,
                args.relative_train_emb_path,
                args.relative_test_emb_path,
                size=size,
                use_external_bias=True if 'Delta' in model_name else False,
                seed=seed)
            results = model.experiment()
            print(f"{model_name} | Seed: {seed} | Size: {size}%")
            for k, v in results.items():
                print(f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}")
            save_results(results, model_name, args.experiment_folder, size, seed)

        for model_name, model_class in absolute_models.items():
            model = model_class(
                args.absolute_train_path,
                args.absolute_test_path,
                args.absolute_train_emb1_path,
                args.absolute_train_emb2_path,
                args.absolute_test_emb1_path,
                args.absolute_test_emb2_path,
                size=size,
                use_external_bias=True if 'Delta' in model_name else False,
                seed=seed
            )
            results = model.experiment()
            print(f"{model_name} | Seed: {seed} | Size: {size}%")
            for k, v in results.items():
                print(f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}")
            save_results(results, model_name, args.experiment_folder, size, seed)

def main():
    parser = argparse.ArgumentParser(description="Run Ratings Dataset experiment.")
    parser.add_argument("--relative_train_path", type=str)
    parser.add_argument("--relative_test_path", type=str)
    parser.add_argument("--relative_train_emb_path", type=str)
    parser.add_argument("--relative_test_emb_path", type=str)
    parser.add_argument("--absolute_train_path", type=str)
    parser.add_argument("--absolute_test_path", type=str)
    parser.add_argument("--absolute_train_emb1_path", type=str)
    parser.add_argument("--absolute_train_emb2_path", type=str)
    parser.add_argument("--absolute_test_emb1_path", type=str)
    parser.add_argument("--absolute_test_emb2_path", type=str)
    parser.add_argument("--experiment_folder", type=str)

    args = parser.parse_args()

    seeds = list(range(10))  # Seeds: 0 to 9
    args_tuples = [(args, seed) for seed in seeds]

    num_workers = min(len(seeds), multiprocessing.cpu_count() - 1)
    with Pool(processes=num_workers) as pool:
        pool.map(run_all_models_for_seed, args_tuples)

if __name__ == "__main__":
    main()
