import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_recall_fscore_support, r2_score

class LLMRegressor:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
    
    def preprocess(self):
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        for df in [train_df, test_df]:
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
        precision, recall, f1, support = precision_recall_fscore_support(y_test_rounded, y_pred_rounded, average='weighted', zero_division=1)

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
            "Support": support
        }
        
    def experiment(self):
        return self.eval()
