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


class TorchDeltaLS(nn.Module):
    def __init__(self, input_dim: int, use_external_bias: bool, lambda_l2: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)
        self.lambda_l2 = lambda_l2
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
        reg = self.lambda_l2 * torch.sum(self.theta ** 2)
        if self.alpha is not None:
            reg += self.lambda_l2 * self.alpha ** 2
        return reg


class DeltaLS:
    def __init__(self, train_path: str, test_path: str,
                 train_emb_path: str, test_emb_path: str,
                 size: int = -1, use_external_bias: bool = True,
                 seed: int = 42):
        set_seed(seed)
        self.train_path = train_path
        self.test_path = test_path
        self.train_emb_path = train_emb_path
        self.test_emb_path = test_emb_path
        self.size = size
        self.use_external_bias = use_external_bias
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.best_lambda_l2 = None
        self.epochs = 10000
        self.min_epochs = 200
        self.early_stopping_patience = 10
        self.lr = 1e-3

    def preprocess(self):
        train_df = pd.read_csv(self.train_path).dropna(subset=["human_score"])
        test_df = pd.read_csv(self.test_path).dropna(subset=["human_score"])
        train_df["human_score"] = train_df["human_score"].astype(int)
        test_df["human_score"] = test_df["human_score"].astype(int)

        mean_score = train_df["llm_score"].mean()
        std_score = train_df["llm_score"].std()
        train_df["llm_score"] = (train_df["llm_score"] - mean_score) / std_score
        test_df["llm_score"] = (test_df["llm_score"] - mean_score) / std_score

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

        return map(torch.tensor, (X_train, X_val, y_train, y_val, X_test, y_test))

    def tune_hyperparameters(self, X_train, X_val, y_train, y_val, lambda_l2_values):
        best_loss = float('inf')
        input_dim = X_train.shape[1]
        self.reg_data = {}

        for l2 in lambda_l2_values:
            model = TorchDeltaLS(input_dim, self.use_external_bias, lambda_l2=l2).to(self.device)
            model = self.train(X_train, X_val, y_train, y_val, lambda_l2=l2)
            with torch.no_grad():
                preds = model(X_val.to(self.device))
                loss = F.mse_loss(preds, y_val.to(self.device)) + model.l2_regularization()
            self.reg_data[l2] = loss.item()
            print(f"Validation MSE: {loss:.4f} @ λ₂: {l2}")
            if loss < best_loss:
                best_loss = loss
                self.best_lambda_l2 = l2
                best_model_state = model.state_dict()

        print(f"Best regularization: λ₂={self.best_lambda_l2} with MSE {best_loss:.4f}")

    def train(self, X_train, X_val, y_train, y_val, lambda_l2=None):
        model = TorchDeltaLS(X_train.shape[1], self.use_external_bias, lambda_l2=lambda_l2).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        best_loss = float('inf')
        best_model_state = None
        early_stop_counter = 0

        for epoch in range(self.epochs):
            model.train()
            optimizer.zero_grad()
            preds = model(X_train.to(self.device))
            loss = F.mse_loss(preds, y_train.to(self.device)) + model.l2_regularization()
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
            return self.model(X.to(self.device)).cpu().numpy()

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> dict:
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

        lambda_l2_values = [10.0 ** x for x in range(-5, 6)]
        self.tune_hyperparameters(X_train, X_val, y_train, y_val, lambda_l2_values)
        l2 = self.best_lambda_l2
        self.model = self.train(X_train, X_val, y_train, y_val, lambda_l2=l2)

        return self.evaluate(X_test, y_test)