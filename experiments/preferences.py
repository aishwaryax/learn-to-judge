import argparse
import os
import csv
from regressors.delta_multinomial import DeltaMultinomial
from regressors.delta_btl2 import DeltaBTL2
from regressors.llm import LLMRegressor
from regressors.llm2 import LLM2Regressor
from pathlib import Path
import sys

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
    parser.add_argument("--relative_train_path", type=str, help="Path to the training data CSV file.")
    parser.add_argument("--relative_test_path", type=str, help="Path to the testing data CSV file.")
    parser.add_argument("--relative_train_emb_path", type=str, help="Path to the training embeddings .npy file.")
    parser.add_argument("--relative_test_emb_path", type=str, help="Path to the training embeddings .npy file.")
    parser.add_argument("--absolute_train_path", type=str, help="Path to the training data CSV file.")
    parser.add_argument("--absolute_test_path", type=str, help="Path to the testing data CSV file.")
    parser.add_argument("--absolute_train_emb1_path", type=str, help="Path to the training embeddings .npy file.")
    parser.add_argument("--absolute_train_emb2_path", type=str, help="Path to the training embeddings .npy file.")
    parser.add_argument("--absolute_test_emb1_path", type=str, help="Path to the training embeddings .npy file.")
    parser.add_argument("--absolute_test_emb2_path", type=str, help="Path to the training embeddings .npy file.")
    parser.add_argument("--experiment_folder", type=str, help="Folder to save experiment results.")

    args = parser.parse_args()    
    
    absolute_models = {
        "BTL2": DeltaBTL2,
        "Delta BTL2": DeltaBTL2,
        "LLM2Regressor": LLM2Regressor
    }
    
    relative_models = {
        "BTL": DeltaMultinomial,
        "Delta BTL": DeltaMultinomial,
        "LLMRegressor": LLMRegressor
    }
    
    for model_name, model_class in relative_models.items():
        if model_name == "LLMRegressor":
            model = model_class(args.relative_test_path)
        else:
            model = model_class(args.relative_train_path, args.relative_test_path, args.relative_train_emb_path, args.relative_test_emb_path, use_external_bias=True if 'Delta' in model_name else False)
        results = model.experiment()
        
        print(f"{model_name} Experiment Results:")
        for metric, value in results.items():
            if isinstance(value, (int, float)):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
        
        save_results(results, model_name, args.experiment_folder)
        
    for model_name, model_class in absolute_models.items():
        if model_name == "LLM2Regressor":
            model = model_class(args.absolute_test_path)
        else:
            model = model_class(args.absolute_train_path, args.absolute_test_path, args.absolute_train_emb1_path, args.absolute_train_emb2_path, args.absolute_test_emb1_path, args.absolute_test_emb2_path, use_external_bias=True if 'Delta' in model_name else False)
        results = model.experiment()
        
        print(f"{model_name} Experiment Results:")
        for metric, value in results.items():
            if isinstance(value, (int, float)):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
        
        save_results(results, model_name, args.experiment_folder)

if __name__ == "__main__":
    main()
