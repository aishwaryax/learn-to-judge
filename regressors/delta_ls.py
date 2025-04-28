import os
import random
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)
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


class DeltaLSJudge(nn.Module):
    def __init__(self, input_dim: int, lambda_theta: float = 0.0):
        super().__init__()
        self.lambda_theta = lambda_theta
        self.Theta = nn.Parameter(torch.zeros(input_dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x @ self.Theta + self.bias
        return z.view(-1)

    def l2_regularization(self) -> torch.Tensor:
        return self.lambda_theta * torch.sum(self.Theta ** 2)


class DeltaLS:
    def __init__(self, train_path: str, test_path: str,
                 train_emb_path: str, test_emb_path: str,
                 size: int = 100, use_external_bias: bool = True,
                 seed: int = 42, standardize: bool = True, is_percent: bool = True):
        self.seed = seed
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
        self.standardize = standardize
        self.lr = 1
        self.max_iter = 10000
        self.tol = 1e-9
        self.is_percent = is_percent
        
    def _clean_data(self, df):
        df.dropna(subset=["human_score", "llm_score"], inplace=True)
        df["human_score"] = df["human_score"].astype(int)
        df["llm_score"] = df["llm_score"].astype(int)
        df["target_probability"] = df["target_probability"].apply(ast.literal_eval)

    def preprocess(self):
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        self._clean_data(train_df)
        self._clean_data(test_df)

        train_embeddings = np.load(self.train_emb_path)
        test_embeddings = np.load(self.test_emb_path)

        if not self.is_percent or (self.is_percent and self.size != 100):
            if self.is_percent:
                n = min(int(len(train_df) * self.size/100), len(train_df))
            else:
                n = self.size
            selected_indices = np.random.choice(train_df.index, size=n, replace=False)
            train_df = train_df.loc[selected_indices].reset_index(drop=True)

        def prepare_features(df, embeddings):
            X = embeddings[df["embedding_index_critique"].values]
            if self.use_external_bias:
                bias = df[["llm_score"]].values
                X = np.hstack([X, bias])
            y = df["human_score"].values
            return X, y

        X_train, y_train = prepare_features(train_df, train_embeddings)
        X_test, y_test = prepare_features(test_df, test_embeddings)

        if self.standardize:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        return map(
            lambda x: torch.tensor(x, dtype=torch.float32, device=self.device),
            (X_train, y_train, X_test, y_test)
        )

    def tune_hyperparameters(self, X_train, y_train, k_folds=5, lambda_values=None):
        best_loss = float('inf')
        input_dim = X_train.shape[1]
        self.reg_data = {}

        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=self.seed)

        for lambda_theta in lambda_values:
            fold_losses = []
            for train_idx, val_idx in kfold.split(X_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                model = self.train(X_tr, y_tr, lambda_theta=lambda_theta)
                model.eval()
                with torch.no_grad():
                    preds = model(X_val)
                    loss = F.mse_loss(preds, y_val) + model.l2_regularization()
                fold_losses.append(loss.item())

            avg_loss = np.mean(fold_losses)
            self.reg_data[lambda_theta] = avg_loss
            print(f"Average Validation MSE: {avg_loss:.4f} @ (λ_theta={lambda_theta})")
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.best_lambda_theta = lambda_theta

        print(f"Best regularization: λ_theta={self.best_lambda_theta} with Avg MSE {best_loss:.4f}")

    def train(self, X_train, y_train, lambda_theta=None, n_steps=10000, batch_size=512):
        model = DeltaLSJudge(X_train.shape[1], lambda_theta=lambda_theta).to(self.device)
        learning_rate = 1.0 / np.sqrt(n_steps)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        model.train()
        dataset_size = X_train.shape[0]

        for step in range(n_steps):
            indices = torch.randint(0, dataset_size, (batch_size,), device=self.device)
            X_batch = X_train[indices]
            y_batch = y_train[indices]

            optimizer.zero_grad()
            preds = model(X_batch).squeeze()
            loss = F.mse_loss(preds, y_batch) + model.l2_regularization()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        return model

    def predict(self, X: torch.Tensor) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            return self.model(X).cpu().numpy()

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        expected_score = y_pred
        if hasattr(y, "cpu"):
            y = y.cpu().numpy()
        if np.issubdtype(y.dtype, np.floating):
            y_rounded = np.round(y).astype(int)
        else:
            y_rounded = y.astype(int)

        y_pred_rounded = np.round(y_pred).astype(int)
        return {
            "MSE": mean_squared_error(y, y_pred),
            "MAE": mean_absolute_error(y, y_pred),
            "R2 Score": r2_score(y, y_pred),
            "Min Prediction": np.min(y_pred),
            "Max Prediction": np.max(y_pred),
            "Accuracy": accuracy_score(y_rounded, y_pred_rounded),
            "Precision": precision_recall_fscore_support(y_rounded, y_pred_rounded, average="weighted", zero_division=1)[0],
            "Recall": precision_recall_fscore_support(y_rounded, y_pred_rounded, average="weighted", zero_division=1)[1],
            "F1 Score": precision_recall_fscore_support(y_rounded, y_pred_rounded, average="weighted", zero_division=1)[2],
            "Confusion Matrix": confusion_matrix(y_rounded, y_pred_rounded, labels=np.arange(y_rounded.max() + 1)),
            "Class Labels": list(range(y_rounded.max() + 1)),
            "Pearson r": pearsonr(y, y_pred)[0],
            "Spearman ρ": spearmanr(y, y_pred)[0],
            "Kendall τ": kendalltau(y, y_pred)[0],
        }


    def experiment(self) -> dict:
        X_train, y_train, X_test, y_test = self.preprocess()
        lambda_values = [10.0 ** x for x in range(-5, 6)]
        self.tune_hyperparameters(X_train, y_train, k_folds=5, lambda_values=lambda_values)
        self.model = self.train(X_train, y_train, lambda_theta=self.best_lambda_theta)
        return self.evaluate(X_test, y_test)