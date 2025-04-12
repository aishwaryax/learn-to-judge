import argparse
import os
import csv
from regressors.delta_multinomial import DeltaMultinomial
from regressors.delta_ls import DeltaLS
from pathlib import Path
from regressors.llm import LLMRegressor

def save_results(results, model_name, experiment_folder):
    experiment_folder = Path(experiment_folder)
    experiment_folder.mkdir(parents=True, exist_ok=True)
    csv_path = os.path.join(experiment_folder, "experiment_results.csv")
    
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Model"] + list(results.keys()))
        writer.writerow([model_name] + list(results.values()))

def main():
    parser = argparse.ArgumentParser(description="Run Ratings Dataset experiment.")
    parser.add_argument("--train_path", type=str, help="Path to the training data CSV file.")
    parser.add_argument("--test_path", type=str, help="Path to the testing data CSV file.")
    parser.add_argument("--train_emb_path", type=str, help="Path to the training embeddings .npy file.")
    parser.add_argument("--test_emb_path", type=str, help="Path to the testing embeddings .npy file.")
    parser.add_argument("--experiment_folder", type=str, help="Folder to save experiment results.")
    args = parser.parse_args()
    
    models = {
        "LS": DeltaLS,
        "Multinomial": DeltaMultinomial,
        "Delta LS": DeltaLS,
        "Delta Multinomial": DeltaMultinomial,
        "LLMRegressor": LLMRegressor,
    }
    
    for model_name, model_class in models.items():
        if model_name == "LLMRegressor":
            model = model_class(args.test_path)
        else:
            model = model_class(args.train_path, args.test_path, args.train_emb_path, args.test_emb_path, use_external_bias=True if "Delta" in model_name else False)
        results = model.experiment()
        
        print(f"{model_name} Experiment Results:")
        for metric, value in results.items():
            if isinstance(value, (int, float)):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
        print("================================================")        
        save_results(results, model_name, args.experiment_folder)

if __name__ == "__main__":
    main()
