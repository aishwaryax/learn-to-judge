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
from sklearn.preprocessing import StandardScaler

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

    def regularization_loss(self, lambda_theta: float) -> torch.Tensor:
        return lambda_theta * torch.sum(self.theta ** 2 + self.alpha ** 2)


class DeltaBTL2:
    def __init__(self, train_path: str, test_path: str,
                 emb1_path_train: str, emb2_path_train: str,
                 emb1_path_test: str, emb2_path_test: str,
                 size: int = -1, lr: float = 1e-3, epochs: int = 10000,
                 lambda_theta: float = 0.0, early_stopping_patience: int = 10,
                 use_external_bias: bool = True, device: torch.device = None,
                 seed: int = 42, standardize: bool = True):
        set_seed(seed)
        self.seed = seed
        self.train_path = train_path
        self.test_path = test_path
        self.emb1_path_train = emb1_path_train
        self.emb2_path_train = emb2_path_train
        self.emb1_path_test = emb1_path_test
        self.emb2_path_test = emb2_path_test
        self.size = size
        self.lr = lr
        self.epochs = epochs
        self.lambda_theta = lambda_theta
        self.early_stopping_patience = early_stopping_patience
        self.use_external_bias = use_external_bias
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_epochs = 200
        self.model = None
        self.reg_data = {}
        self.standardize = standardize 

    def _load_data(self):
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        emb_train1 = np.load(self.emb1_path_train)
        emb_train2 = np.load(self.emb2_path_train)
        emb_test1 = np.load(self.emb1_path_test)
        emb_test2 = np.load(self.emb2_path_test)
        return train_df, test_df, emb_train1, emb_train2, emb_test1, emb_test2

    def _filter_and_sample(self, df, emb1, emb2):
        df.dropna(subset=["human_score", "llm_score1", "llm_score2",
                         "embedding_index_critique1", "embedding_index_critique2"], inplace=True)
        df["human_score"] = df["human_score"].astype(int)
        if self.size != -1:
            df = df.sample(n=min(self.size, len(df)), random_state=42).reset_index(drop=True)
        return df

    def _compute_log_odds(self, df):
        p = df["llm_score1"].replace(0, 1e-8).astype(np.float32) / (
            df["llm_score1"].replace(0, 1e-8).astype(np.float32) +
            df["llm_score2"].replace(0, 1e-8).astype(np.float32))
        return np.log(p / np.clip(1 - p, 1e-8, None))

    def preprocess(self):
        train_df, test_df, emb_train1, emb_train2, emb_test1, emb_test2 = self._load_data()
        train_df = self._filter_and_sample(train_df, emb_train1, emb_train2)
        test_df = self._filter_and_sample(test_df, emb_test1, emb_test2)

        X_train = emb_train1[train_df["embedding_index_critique1"]] - emb_train2[train_df["embedding_index_critique2"]]
        y_train = train_df["human_score"].values
        log_odds_train = self._compute_log_odds(train_df).values

        X_test = emb_test1[test_df["embedding_index_critique1"]] - emb_test2[test_df["embedding_index_critique2"]]
        y_test = test_df["human_score"].values
        log_odds_test = self._compute_log_odds(test_df).values

        X_train, X_val, y_train, y_val, log_odds_train, log_odds_val = train_test_split(
            X_train, y_train, log_odds_train, test_size=0.2, random_state=42)
                        
        if self.standardize:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
            X_test = self.scaler.transform(X_test)

        X_train = torch.tensor(X_train, dtype=torch.float, device=self.device)
        y_train = torch.tensor(y_train, dtype=torch.float, device=self.device)
        log_odds_train = torch.tensor(log_odds_train, dtype=torch.float, device=self.device)

        X_val = torch.tensor(X_val, dtype=torch.float, device=self.device)
        y_val = torch.tensor(y_val, dtype=torch.float, device=self.device)
        log_odds_val = torch.tensor(log_odds_val, dtype=torch.float, device=self.device)

        X_test = torch.tensor(X_test, dtype=torch.float, device=self.device)
        y_test = torch.tensor(y_test, dtype=torch.float, device=self.device)
        log_odds_test = torch.tensor(log_odds_test, dtype=torch.float, device=self.device)

        return X_train, y_train, log_odds_train, X_val, y_val, log_odds_val, X_test, y_test, log_odds_test


    def train_model(self, X_train, y_train, log_odds_train, X_val=None, y_val=None, log_odds_val=None):
        self.model = DeltaBTL2Judge(X_train.shape[1]).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        best_val_loss, best_state_dict = float('inf'), None
        epochs_without_improvement = 0

        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            preds = self.model(X_train, log_odds_train).squeeze()
            loss = F.binary_cross_entropy(preds, y_train)
            loss += self.model.regularization_loss(self.lambda_theta)
            loss.backward()
            optimizer.step()

            if X_val is not None and y_val is not None and log_odds_val is not None:
                if epoch < self.min_epochs:
                    continue
                self.model.eval()
                with torch.no_grad():
                    val_preds = self.model(X_val, log_odds_val).squeeze()
                    val_loss = F.binary_cross_entropy(val_preds, y_val)
                    val_loss += self.model.regularization_loss(self.lambda_theta)
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

    def tune_hyperparameters(self, X_train, y_train, log_odds_train, X_val, y_val, log_odds_val, lambda_values):
        best_acc, best_lambda_theta, best_state_dict = 0, None, None
        self.reg_data = {}

        for lambda_theta in lambda_values:
            self.lambda_theta = lambda_theta
            self.train_model(X_train, y_train, log_odds_train, X_val, y_val, log_odds_val)

            self.model.eval()
            with torch.no_grad():
                preds = self.model(X_val, log_odds_val).cpu().numpy()
                y_pred = (preds > 0.5).astype(int)
                val_acc = accuracy_score(y_val.cpu().numpy(), y_pred)

            self.reg_data[(lambda_theta)] = val_acc
            print(f"Validation Accuracy: {val_acc:.4f} @ (λ_theta={lambda_theta})")
            if val_acc > best_acc:
                best_acc, best_lambda_theta = val_acc, lambda_theta
                best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        print(f"Best λ_theta={best_lambda_theta} with validation accuracy {best_acc:.4f}")
        self.lambda_theta = best_lambda_theta
        if best_state_dict:
            self.model.load_state_dict(best_state_dict)
            
    def predict(self, X, log_odds):
        self.model.eval()
        with torch.no_grad():
            probs = self.model(X, log_odds).squeeze()
        return (probs > 0.5).cpu().numpy().astype(int), probs.cpu().numpy()

    def evaluate(self, X_test, y_test, log_odds_test):
        y_pred, probs = self.predict(X_test, log_odds_test)
        y_test = y_test.cpu().numpy()
        return {
            "MSE": mean_squared_error(y_test, probs),
            "MAE": mean_absolute_error(y_test, probs),
            "R2 Score": r2_score(y_test, probs),
            "Min Prediction": np.min(y_pred),
            "Max Prediction": np.max(y_pred),
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=1)[0],
            "Recall": precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=1)[1],
            "F1 Score": precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=1)[2],
            "Confusion Matrix": confusion_matrix(y_test, y_pred),
            "Class Labels": np.unique(y_test).astype(int).tolist(),
            "Pearson r": pearsonr(y_test, probs)[0],
            "Spearman ρ": spearmanr(y_test, probs)[0],
            "Kendall τ": kendalltau(y_test, probs)[0],
        }

    def experiment(self):
        X_train, y_train, log_odds_train, X_val, y_val, log_odds_val, X_test, y_test, log_odds_test = self.preprocess()
        lambda_values = [10.0 ** x for x in range(-5, 6)]
        self.tune_hyperparameters(X_train, y_train, log_odds_train, X_val, y_val, log_odds_val, lambda_values)
        self.train_model(X_train, y_train, log_odds_train, X_val, y_val, log_odds_val)
        return self.evaluate(X_test, y_test, log_odds_test)