import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_recall_fscore_support, r2_score, confusion_matrix
import os
import random
import torch
from scipy.stats import pearsonr, spearmanr, kendalltau

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

class LLMRegressor:
    def __init__(self, test_path):
        self.test_path = test_path
    
    def preprocess(self):
        test_df = pd.read_csv(self.test_path)
        for df in [test_df]:
            df["human_score"] = pd.to_numeric(df["human_score"], errors="coerce")
            df["llm_score"] = pd.to_numeric(df["llm_score"], errors="coerce")
            df.dropna(subset=["human_score", "llm_score"], inplace=True)
        X_test = test_df['llm_score'].values
        y_test = test_df['human_score'].values
        
        return X_test, y_test
    
    def tune_hyperparameters(self, *args, **kwargs):
        pass
    
    def train(self, *args, **kwargs):
        pass
    
    def predict(self, X_test):
        return X_test
    
    def eval(self):
        X_test, y_test = self.preprocess()
        y_pred = self.predict(X_test)
        min_pred, max_pred = np.min(y_pred), np.max(y_pred)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        y_test_rounded = np.round(y_test)
        y_pred_rounded = np.round(np.clip(y_pred, min_pred, max_pred))

        accuracy = accuracy_score(y_test_rounded, y_pred_rounded)

        average_val = "weighted"
        if set(y_test) == 2:
            average_val = "binary"
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average=average_val, zero_division=1
        )
        
        classes = np.unique(y_test).astype(int)
        cm      = confusion_matrix(y_test, y_pred, labels=classes)

        pearson_r,  _ = pearsonr(y_test, y_pred)
        spearman_rho, _ = spearmanr(y_test, y_pred)
        kendall_tau, _ = kendalltau(y_test, y_pred)
        
        return {
            "MSE": mse,
            "MAE": mae,
            "R2 Score": r2,
            "Min Prediction": min_pred,
            "Max Prediction": max_pred,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Confusion Matrix": cm,
            "Class Labels": classes.tolist(),
            "Pearson r": pearson_r,
            "Spearman ρ": spearman_rho,
            "Kendall τ": kendall_tau
        }
        
    def experiment(self):
        return self.eval()
