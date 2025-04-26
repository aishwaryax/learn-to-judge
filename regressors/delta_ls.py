import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, input_dim: int, use_external_bias: bool, lambda_theta: float = 0.0):
        super().__init__()
        self.use_external_bias = use_external_bias
        self.lambda_theta = lambda_theta
        self.theta = nn.Parameter(torch.zeros(input_dim - 1 if use_external_bias else input_dim))
        self.alpha = nn.Parameter(torch.tensor(0.0)) if use_external_bias else None
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.double()

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
        self.epochs = 2000
        self.min_epochs = 200
        self.early_stopping_patience = 10
        self.standardize = standardize
        self.lr = 1
        self.max_iter = 500
        self.tol = 1e-9
        
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

        if self.standardize:
            self.scaler = StandardScaler()
            X_train_full = self.scaler.fit_transform(X_train_full)
            X_test = self.scaler.transform(X_test)

        return map(
            lambda x: torch.tensor(x, dtype=torch.float64, device=self.device),
            (X_train_full, y_train_full, X_test, y_test)
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

                model = DeltaLSJudge(input_dim, self.use_external_bias, lambda_theta=lambda_theta).to(self.device)
                model = self.train(X_tr, y_tr, lambda_theta=lambda_theta, epochs=500)
                with torch.no_grad():
                    preds = model(X_val)
                    loss = F.mse_loss(preds, y_val)
                fold_losses.append(loss.item())

            avg_loss = np.mean(fold_losses)
            self.reg_data[lambda_theta] = avg_loss
            print(f"Average Validation MSE: {avg_loss:.4f} @ (λ_theta={lambda_theta})")
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.best_lambda_theta = lambda_theta

        print(f"Best regularization: λ_theta={self.best_lambda_theta} with Avg MSE {best_loss:.4f}")

    def train(self, X_train, y_train, lambda_theta=None, epochs=None):
        model = DeltaLSJudge(X_train.shape[1], self.use_external_bias, lambda_theta=lambda_theta).to(self.device)
        epochs = self.epochs if epochs is None else epochs
        learning_rate = 1.0 / np.sqrt(epochs)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        batch_size = 128
        dataset_size = X_train.shape[0]
        steps_per_epoch = (dataset_size + batch_size - 1) // batch_size

        model.train()

        for epoch in range(epochs):
            permutation = torch.randperm(dataset_size)

            for step in range(steps_per_epoch):
                start_idx = step * batch_size
                end_idx = min((step + 1) * batch_size, dataset_size)
                indices = permutation[start_idx:end_idx]

                X_batch = X_train[indices]
                y_batch = y_train[indices]

                optimizer.zero_grad()
                preds = model(X_batch)
                loss = F.mse_loss(preds, y_batch) + model.l2_regularization()
                loss.backward()
                optimizer.step()

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
            "Max Prediction": max_pred,
            "Pearson r": pearsonr(y, y_pred)[0],
            "Spearman ρ": spearmanr(y, y_pred)[0],
            "Kendall τ": kendalltau(y, y_pred)[0]
        }

        return metrics

    def experiment(self) -> dict:
        X_train_full, y_train_full, X_test, y_test = self.preprocess()

        lambda_values = [10.0 ** x for x in range(-5, 6)]
        self.tune_hyperparameters(X_train_full, y_train_full, k_folds=5, lambda_values=lambda_values)
        self.model = self.train(X_train_full, y_train_full, lambda_theta=self.best_lambda_theta)

        return self.evaluate(X_test, y_test)