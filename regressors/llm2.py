import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_recall_fscore_support, r2_score
import os
import random
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

class LLM2Regressor:
    def __init__(self, test_path):
        self.test_path = test_path
    
    def preprocess(self):
        test_df = pd.read_csv(self.test_path)

        for df in [test_df]:
            df["human_score"] = pd.to_numeric(df["human_score"], errors="coerce")
            df["llm_score1"] = pd.to_numeric(df["llm_score1"], errors="coerce")
            df["llm_score2"] = pd.to_numeric(df["llm_score2"], errors="coerce")
            df.dropna(subset=["human_score", "llm_score1", "llm_score2"], inplace=True)

        # Prepare X_test (features: llm_score1 and llm_score2)
        X_test = test_df[['llm_score1', 'llm_score2']].values

        # Use human_score as y_test (target)
        y_test = test_df['human_score'].values
        
        return X_test, y_test
    
    def tune_hyperparameters(self, *args, **kwargs):
        pass
    
    def train(self, *args, **kwargs):
        pass
    
    def predict(self, X_test):
        def assign_y_pred(row):
            if row[0] > row[1]:
                return 0
            elif row[0] < row[1]:
                return 1
            else:
                return np.random.choice([0, 1])
        return np.array([assign_y_pred(row) for row in X_test])
    
    def eval(self):
        X_test, y_test = self.preprocess()
        y_pred = self.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        y_test_rounded = np.round(y_test)
        y_pred_rounded = np.round(np.clip(y_pred, 0, 1))

        accuracy = accuracy_score(y_test_rounded, y_pred_rounded)
        average_val = "weighted"
        if set(y_test) == 2:
            average_val = "binary"
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average=average_val, zero_division=1
        )
        return {
            "MSE": mse,
            "MAE": mae,
            "R2 Score": r2,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        }
        
    def experiment(self):
        return self.eval()