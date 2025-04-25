import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)
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


class DeltaLSJudge(nn.Module):
    def __init__(self, input_dim: int, use_external_bias: bool, lambda_theta: float = 0.0):
        super().__init__()
        self.use_external_bias = use_external_bias
        self.lambda_theta = lambda_theta
        self.theta = nn.Parameter(torch.zeros(input_dim - 1 if use_external_bias else input_dim))
        self.alpha = nn.Parameter(torch.tensor(0.0)) if use_external_bias else None
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_external_bias:
            emb, ext = x[:, :-1], x[:, -1]
            z = emb @ self.theta + self.alpha * ext + self.bias
        else:
            z = x @ self.theta + self.bias
        return z.squeeze(-1)

    def l2_regularization(self) -> torch.Tensor:
        reg = self.lambda_theta * torch.sum(self.theta ** 2)
        if self.alpha is not None:
            reg += self.lambda_theta * self.alpha ** 2
        return reg


class DeltaLS:
    def __init__(self, train_path: str, test_path: str,
                 train_emb_path: str, test_emb_path: str,
                 size: int = -1, use_external_bias: bool = True,
                 seed: int = 42, standardize: bool = True):
        set_seed(seed)
        self.train_path = train_path
        self.test_path = test_path
        self.train_emb_path = train_emb_path
        self.test_emb_path = test_emb_path
        self.size = size
        self.use_external_bias = use_external_bias
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.best_lambda_theta = None
        self.epochs = 10000
        self.min_epochs = 200
        self.early_stopping_patience = 10
        self.lr = 1e-3
        
        self.standardize = standardize 

    def preprocess(self):
        train_df = pd.read_csv(self.train_path).dropna(subset=["human_score"])
        test_df = pd.read_csv(self.test_path).dropna(subset=["human_score"])
        train_df["human_score"] = train_df["human_score"].astype(int)
        test_df["human_score"] = test_df["human_score"].astype(int)

        train_embeddings = np.load(self.train_emb_path)
        test_embeddings = np.load(self.test_emb_path)

        if self.size != -1:
            selected_indices = np.random.choice(train_df.index, size=min(self.size, len(train_df)), replace=False)
            train_df = train_df.loc[selected_indices].reset_index(drop=True)

        def prepare_features(df, embeddings):
            X = embeddings[df["embedding_index_critique"].values]
            if self.use_external_bias:
                bias = df[["llm_score"]].values
                X = np.hstack([X, bias])
            y = df["human_score"].values
            return X, y

        X_train_full, y_train_full = prepare_features(train_df, train_embeddings)
        X_test, y_test = prepare_features(test_df, test_embeddings)
        
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

        if self.standardize:
            mean = X_train.mean(axis=0)
            std = X_train.std(axis=0)
            std[std == 0] = 1e-8

            X_train = (X_train - mean) / std
            X_val = (X_val - mean) / std
            X_test = (X_test - mean) / std

        return map(
            lambda x, dtype: torch.tensor(x, dtype=dtype, device=self.device),
            (X_train, X_val, y_train, y_val, X_test, y_test),
            (torch.float, torch.float, torch.float, torch.float, torch.float, torch.long)
        )

    def tune_hyperparameters(self, X_train, X_val, y_train, y_val, lambda_values):
        best_loss = float('inf')
        input_dim = X_train.shape[1]
        self.reg_data = {}

        for lambda_theta in lambda_values:
            model = DeltaLSJudge(input_dim, self.use_external_bias, lambda_theta=lambda_theta).to(self.device)
            model = self.train(X_train, X_val, y_train, y_val, lambda_theta=lambda_theta)
            with torch.no_grad():
                preds = model(X_val)
                loss = F.mse_loss(preds, y_val) + model.l2_regularization()
            self.reg_data[(lambda_theta)] = loss.item()
            print(f"Validation MSE: {loss:.4f} @ (λ_theta={lambda_theta})")
            if loss < best_loss:
                best_loss = loss
                self.best_lambda_theta = lambda_theta
                best_model_state = model.state_dict()

        print(f"Best regularization: λ_theta={self.best_lambda_theta} with MSE {best_loss:.4f}")

    def train(self, X_train, X_val, y_train, y_val, lambda_theta=None):
        model = DeltaLSJudge(X_train.shape[1], self.use_external_bias, lambda_theta=lambda_theta).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        best_loss = float('inf')
        best_model_state = None
        early_stop_counter = 0

        for epoch in range(self.epochs):
            model.train()
            optimizer.zero_grad()
            preds = model(X_train)
            loss = F.mse_loss(preds, y_train) + model.l2_regularization()
            loss.backward()
            optimizer.step()

            if epoch < self.min_epochs:
                continue

            loss_val = loss.item()
            if loss_val < best_loss - 1e-4:
                best_loss = loss_val
                best_model_state = model.state_dict()
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= self.early_stopping_patience:
                    print(f"Early stopping during training at epoch {epoch}")
                    break

        if best_model_state:
            model.load_state_dict(best_model_state)
        return model

    def predict(self, X: torch.Tensor) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            return self.model(X).cpu().numpy()

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> dict:
        y = y.detach().cpu().numpy()
        y_pred = self.predict(X)
        min_pred, max_pred = y_pred.min(), y_pred.max()

        metrics = {
            "MSE": mean_squared_error(y, y_pred),
            "MAE": mean_absolute_error(y, y_pred),
            "R2 Score": r2_score(y, y_pred),
            "Min Prediction": min_pred,
            "Max Prediction": max_pred
        }

        y_true_rounded = np.round(y)
        y_pred_rounded = np.round(np.clip(y_pred, min_pred, max_pred))

        metrics.update({
            "Accuracy": accuracy_score(y_true_rounded, y_pred_rounded),
            "Precision": precision_recall_fscore_support(y_true_rounded, y_pred_rounded, average='weighted', zero_division=0)[0],
            "Recall": precision_recall_fscore_support(y_true_rounded, y_pred_rounded, average='weighted', zero_division=0)[1],
            "F1 Score": precision_recall_fscore_support(y_true_rounded, y_pred_rounded, average='weighted', zero_division=0)[2],
            "Confusion Matrix": confusion_matrix(y_true_rounded, y_pred_rounded),
            "Class Labels": np.unique(y_true_rounded).astype(int).tolist(),
            "Pearson r": pearsonr(y, y_pred)[0],
            "Spearman ρ": spearmanr(y, y_pred)[0],
            "Kendall τ": kendalltau(y, y_pred)[0]
        })
        return metrics

    def experiment(self) -> dict:
        X_train, X_val, y_train, y_val, X_test, y_test = self.preprocess()
        X_train, X_val, X_test = X_train.float(), X_val.float(), X_test.float()
        y_train, y_val, y_test = y_train.float(), y_val.float(), y_test.float()

        lambda_values = [10.0 ** x for x in range(-5, 6)]
        self.tune_hyperparameters(X_train, X_val, y_train, y_val, lambda_values)
        self.model = self.train(X_train, X_val, y_train, y_val,
                                lambda_theta=self.best_lambda_theta)
        return self.evaluate(X_test, y_test)
