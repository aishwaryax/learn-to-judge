import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)
import torch, random, os
from scipy.stats import pearsonr, spearmanr, kendalltau

# --------------------------------------------------------------------------- #
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)
# --------------------------------------------------------------------------- #

class MeanHumanRegressor:
    """
    Baseline that always predicts the (sample) mean of human_score.
    Structure intentionally parallels LLMRegressor.
    """
    def __init__(self, test_path):
        self.test_path = test_path
        self._mean_human = None        # cached after preprocess

    # --------------------------- data handling ------------------------------ #
    def preprocess(self):
        df = pd.read_csv(self.test_path)
        df["human_score"] = pd.to_numeric(df["human_score"], errors="coerce")
        df.dropna(subset=["human_score"], inplace=True)

        # cache the mean so .predict() can access it later
        self._mean_human = df["human_score"].mean()

        X_test = np.zeros(len(df))            # dummy feature vector (not used)
        y_test = df["human_score"].values
        return X_test, y_test

    # ---------------------- placeholder API hooks --------------------------- #
    def tune_hyperparameters(self, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        pass

    # ----------------------------- inference -------------------------------- #
    def predict(self, X_test):
        if self._mean_human is None:
            raise RuntimeError("Call preprocess() before predict().")
        return np.full_like(X_test, fill_value=self._mean_human, dtype=float)

    # ------------------------------ metrics --------------------------------- #
    def eval(self):
        X_test, y_test = self.preprocess()
        y_pred = self.predict(X_test)

        min_pred, max_pred = y_pred.min(), y_pred.max()

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2  = r2_score(y_test, y_pred)

        y_true_round = np.rint(y_test)
        y_pred_round = np.rint(np.clip(y_pred, min_pred, max_pred))
        
        classes = np.unique(y_true_round).astype(int)
        cm = confusion_matrix(y_true_round, y_pred_round, labels=classes)

        pearson_r,  _ = pearsonr(y_test, y_pred)
        spearman_rho, _ = spearmanr(y_test, y_pred)
        kendall_tau, _  = kendalltau(y_test, y_pred)

        acc = accuracy_score(y_true_round, y_pred_round)
        avg = "binary" if len(np.unique(y_test)) == 2 else "weighted"
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_round, y_pred_round, average=avg, zero_division=1
        )

        return {
            "MSE": mse,
            "MAE": mae,
            "R2 Score": r2,
            "Min Prediction": min_pred,
            "Max Prediction": max_pred,
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Confusion Matrix": cm,
            "Class Labels": classes.tolist(),
            "Pearson r": pearson_r,
            "Spearman ρ": spearman_rho,
            "Kendall τ": kendall_tau
        }

    # convenience wrapper
    def experiment(self):
        return self.eval()
