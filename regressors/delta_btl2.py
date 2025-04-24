import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr, kendalltau


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class DeltaBTL2Judge(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(input_dim))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, X_diff: torch.Tensor, ext_bias: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(X_diff @ self.theta + self.bias + self.alpha * ext_bias)


class DeltaBTL2:
    def __init__(self, train_path: str, test_path: str,
                 emb1_path_train: str, emb2_path_train: str,
                 emb1_path_test: str, emb2_path_test: str,
                 size: int = -1, lr: float = 1e-3, epochs: int = 10000,
                 lambda_l2: float = 0.0, early_stopping_patience: int = 10,
                 use_external_bias: bool = True, device: torch.device = None,
                 seed: int = 42):
        set_seed(seed)
        self.train_path = train_path
        self.test_path = test_path
        self.emb1_path_train = emb1_path_train
        self.emb2_path_train = emb2_path_train
        self.emb1_path_test = emb1_path_test
        self.emb2_path_test = emb2_path_test
        self.size = size
        self.lr = lr
        self.epochs = epochs
        self.lambda_l2 = lambda_l2
        self.early_stopping_patience = early_stopping_patience
        self.use_external_bias = use_external_bias
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_epochs = 200
        self.model = None

    def preprocess(self):
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        emb_train1 = np.load(self.emb1_path_train)
        emb_train2 = np.load(self.emb2_path_train)
        emb_test1 = np.load(self.emb1_path_test)
        emb_test2 = np.load(self.emb2_path_test)

        for df in [train_df, test_df]:
            df.dropna(subset=["human_score", "llm_score1", "llm_score2",
                              "embedding_index_critique1", "embedding_index_critique2"], inplace=True)
            df["human_score"] = df["human_score"].astype(int)

        if self.size != -1:
            sample_size = min(self.size, len(train_df))
            selected_indices = np.random.choice(train_df.index, size=sample_size, replace=False)
            train_df = train_df.loc[selected_indices].reset_index(drop=True)

        X_train = emb_train1[train_df["embedding_index_critique1"]] - emb_train2[train_df["embedding_index_critique2"]]
        y_train = train_df["human_score"].values

        X_test = emb_test1[test_df["embedding_index_critique1"]] - emb_test2[test_df["embedding_index_critique2"]]
        y_test = test_df["human_score"].values

        def compute_log_odds(df):
            p = df["llm_score1"].replace(0, 1e-8).astype(np.float32) / (
                df["llm_score1"].replace(0, 1e-8).astype(np.float32) +
                df["llm_score2"].replace(0, 1e-8).astype(np.float32))
            return np.log(p / np.clip(1 - p, 1e-8, None))

        log_odds_train = compute_log_odds(train_df)
        log_odds_test = compute_log_odds(test_df)

        X_train, X_val, y_train, y_val, log_odds_train, log_odds_val = train_test_split(
            X_train, y_train, log_odds_train, test_size=0.2, random_state=42)

        return X_train, y_train, log_odds_train, X_val, y_val, log_odds_val, X_test, y_test, log_odds_test

    def train_model(self, X_train, y_train, log_odds_train,
                    X_val=None, y_val=None, log_odds_val=None):
        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        ext_bias_train = torch.tensor(log_odds_train.to_numpy(), dtype=torch.float32, device=self.device)

        self.model = DeltaBTL2Judge(X_train.shape[1]).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        best_val_loss, best_state_dict = float('inf'), None
        epochs_without_improvement = 0

        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            preds = self.model(X_train_t, ext_bias_train).squeeze()
            loss = F.binary_cross_entropy(preds, y_train_t)
            if self.lambda_l2 > 0:
                loss += self.lambda_l2 * (torch.sum(self.model.theta ** 2) + self.model.alpha ** 2)
            loss.backward()
            optimizer.step()

            if X_val is not None and y_val is not None and log_odds_val is not None:
                if epoch < self.min_epochs:
                    continue

                self.model.eval()
                with torch.no_grad():
                    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=self.device)
                    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=self.device)
                    ext_bias_val = torch.tensor(log_odds_val.to_numpy(), dtype=torch.float32, device=self.device)
                    val_preds = self.model(X_val_t, ext_bias_val).squeeze()
                    val_loss = F.binary_cross_entropy(val_preds, y_val_t)
                    if self.lambda_l2 > 0:
                        val_loss += self.lambda_l2 * (torch.sum(self.model.theta ** 2) + self.model.alpha ** 2)

                if val_loss.item() < best_val_loss - 1e-4:
                    best_val_loss = val_loss.item()
                    best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= self.early_stopping_patience:
                        print(f"Early stopping at epoch {epoch}.")
                        break

        if best_state_dict:
            self.model.load_state_dict(best_state_dict)
        return self.model

    def tune_hyperparameters(self, X_train, y_train, log_odds_train,
                             X_val, y_val, log_odds_val, lambdas):
        best_acc, best_lambda, best_state_dict = 0, None, None
        self.reg_data = {}

        for lambda_l2 in lambdas:
            self.lambda_l2 = lambda_l2
            self.train_model(X_train, y_train, log_odds_train, X_val, y_val, log_odds_val)

            self.model.eval()
            with torch.no_grad():
                X_val_t = torch.tensor(X_val, dtype=torch.float32, device=self.device)
                ext_bias_val = torch.tensor(log_odds_val.to_numpy(), dtype=torch.float32, device=self.device)
                preds = self.model(X_val_t, ext_bias_val).cpu().numpy()
                y_pred = (preds > 0.5).astype(int)
                val_acc = accuracy_score(y_val, y_pred)

            self.reg_data[lambda_l2] = val_acc
            print(f"Validation Accuracy: {val_acc:.4f} @ λ₂: {lambda_l2}")

            if val_acc > best_acc:
                best_acc, best_lambda = val_acc, lambda_l2
                best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        print(f"Best λ₂={best_lambda} with validation accuracy {best_acc:.4f}")
        self.lambda_l2 = best_lambda
        if best_state_dict:
            self.model.load_state_dict(best_state_dict)

    def evaluate(self, X_test, y_test, log_odds_test):
        self.model.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test, dtype=torch.float32, device=self.device)
            ext_bias_test = torch.tensor(log_odds_test.to_numpy(), dtype=torch.float32, device=self.device)
            preds = self.model(X_test_t, ext_bias_test).cpu().numpy()
            y_pred = (preds > 0.5).astype(int)

        return {
            "MSE": mean_squared_error(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "R2 Score": r2_score(y_test, y_pred),
            "Min Prediction": np.min(y_pred),
            "Max Prediction": np.max(y_pred),
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=1)[0],
            "Recall": precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=1)[1],
            "F1 Score": precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=1)[2],
            "Confusion Matrix": confusion_matrix(y_test, y_pred),
            "Class Labels": np.unique(y_test).astype(int).tolist(),
            "Pearson r": pearsonr(y_test, preds)[0],
            "Spearman ρ": spearmanr(y_test, preds)[0],
            "Kendall τ": kendalltau(y_test, preds)[0],
        }

    def experiment(self):
        X_train, y_train, log_odds_train, X_val, y_val, log_odds_val, X_test, y_test, log_odds_test = self.preprocess()
        lambdas = [10.0 ** x for x in range(-5, 6)]
        self.tune_hyperparameters(X_train, y_train, log_odds_train, X_val, y_val, log_odds_val, lambdas)
        self.train_model(X_train, y_train, log_odds_train, X_val, y_val, log_odds_val)
        return self.evaluate(X_test, y_test, log_odds_test)