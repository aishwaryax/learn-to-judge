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
from sklearn.model_selection import train_test_split, KFold
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
        self.Theta = nn.Parameter(torch.zeros(input_dim, dtype=torch.float64))
        self.bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, X_diff: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(X_diff @ self.Theta + self.bias)

    def regularization_loss(self, lambda_theta: float) -> torch.Tensor:
        return lambda_theta * torch.sum(self.Theta ** 2)


class DeltaBTL2:
    def __init__(self, train_path: str, test_path: str,
                 emb1_path_train: str, emb2_path_train: str,
                 emb1_path_test: str, emb2_path_test: str,
                 size: int = 100, lr: float = 1e-3, epochs: int = 10000,
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
        if self.size != 100:
            n = min(int(len(train_df) * self.size/100), len(train_df))
            train_df = train_df.sample(n=n, random_state=42).reset_index(drop=True)
        X_train = emb_train1[train_df["embedding_index_critique1"]] - emb_train2[train_df["embedding_index_critique2"]]
        y_train = train_df["human_score"].values
        log_odds_train = self._compute_log_odds(train_df).values

        X_test = emb_test1[test_df["embedding_index_critique1"]] - emb_test2[test_df["embedding_index_critique2"]]
        y_test = test_df["human_score"].values
        log_odds_test = self._compute_log_odds(test_df).values

        if self.use_external_bias:
            X_train = np.hstack([X_train, log_odds_train[:, None]])
            X_test = np.hstack([X_test, log_odds_test[:, None]])
                        
        if self.standardize:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        X_train = torch.tensor(X_train, dtype=torch.float64, device=self.device)
        y_train = torch.tensor(y_train, dtype=torch.float64, device=self.device)

        X_test = torch.tensor(X_test, dtype=torch.float64, device=self.device)
        y_test = torch.tensor(y_test, dtype=torch.float64, device=self.device)

        return X_train, y_train, X_test, y_test


    def train(self, X_train, y_train, lambda_theta=None, n_steps=10000, batch_size=512):
        model = DeltaBTL2Judge(input_dim=X_train.shape[1]).to(self.device)
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
            loss = F.binary_cross_entropy(preds, y_batch)
            loss += model.regularization_loss(lambda_theta if lambda_theta is not None else self.lambda_theta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        self.model = model
        return model


    def tune_hyperparameters(self, X_train, y_train, lambda_values=None, k_folds=5):
        best_acc = -float('inf')
        best_lambda_theta = None
        best_state_dict = None
        self.reg_data = {}

        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=self.seed)

        for lambda_theta in lambda_values:
            fold_accuracies = []

            for train_idx, val_idx in kfold.split(X_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                model = self.train(X_tr, y_tr, lambda_theta=lambda_theta)

                model.eval()
                with torch.no_grad():
                    preds = model(X_val).squeeze()
                    preds = (preds > 0.5).cpu().numpy().astype(int)
                    acc = accuracy_score(y_val.cpu().numpy(), preds)
                    fold_accuracies.append(acc)

            avg_acc = np.mean(fold_accuracies)
            self.reg_data[lambda_theta] = avg_acc
            print(f"Avg Validation Accuracy: {avg_acc:.4f} @ (λ_theta={lambda_theta})")

            if avg_acc > best_acc:
                best_acc = avg_acc
                best_lambda_theta = lambda_theta

        print(f"Best λ_theta={best_lambda_theta} with avg validation accuracy {best_acc:.4f}")
        self.lambda_theta = best_lambda_theta

            
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            probs = self.model(X).squeeze()
        return (probs > 0.5).cpu().numpy().astype(int), probs.cpu().numpy()

    def evaluate(self, X_test, y_test):
        y_pred, probs = self.predict(X_test)
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
        X_train, y_train, X_test, y_test = self.preprocess()
        lambda_values = [10.0 ** x for x in range(-5, 6)]
        self.tune_hyperparameters(X_train, y_train, lambda_values)
        self.train(X_train, y_train)
        return self.evaluate(X_test, y_test)