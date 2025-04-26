import os
import ast
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, mean_squared_error, mean_absolute_error, r2_score,
    precision_recall_fscore_support, confusion_matrix
)
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
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


class MNJudge(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.Theta = nn.Parameter(torch.zeros(input_dim, num_classes))
        self.bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X @ self.Theta + self.bias

    def regularization_loss(self, lambda_theta: float) -> torch.Tensor:
        return lambda_theta * torch.sum(self.Theta ** 2)


class DeltaMultinomial:
    def __init__(self, train_path, test_path, train_emb_path, test_emb_path,
                 size=-1, lr=1, epochs=2000, lambda_theta=0.0,
                 device=None, seed=42, standardize=True):
        set_seed(seed)
        self.train_path = train_path
        self.test_path = test_path
        self.train_emb_path = train_emb_path
        self.test_emb_path = test_emb_path
        self.size = size
        self.lr = lr
        self.epochs = epochs
        self.lambda_theta = lambda_theta
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_encoder = LabelEncoder()
        self.model = None
        self.reg_data = {}
        self.standardize = standardize
        self.scaler = None

    def _load_data(self):
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        train_embeddings = np.load(self.train_emb_path)
        test_embeddings = np.load(self.test_emb_path)
        return train_df, test_df, train_embeddings, test_embeddings

    def _clean_data(self, df):
        df.dropna(subset=["human_score", "llm_score"], inplace=True)
        df["human_score"] = df["human_score"].astype(int)
        df["llm_score"] = df["llm_score"].astype(int)
        df["target_probability"] = df["target_probability"].apply(ast.literal_eval)

    def _encode_labels(self, train_df, test_df):
        all_labels = np.concatenate([train_df["human_score"], test_df["human_score"]])
        self.label_encoder.fit(all_labels)

    def compute_log_p(self, df):
        label_classes = self.label_encoder.classes_
        reverse_label_dict = {label: idx for idx, label in enumerate(label_classes)}
        log_p_list = []
        for _, row in df.iterrows():
            tp = row["target_probability"]
            p_vec = np.zeros(len(label_classes))
            for key in tp:
                if int(key) in reverse_label_dict:
                    p_vec[reverse_label_dict[int(key)]] = tp[key]
            log_p_list.append(np.log(np.clip(p_vec, 1e-8, 1.0 - 1e-8)))
        return np.stack(log_p_list)

    def preprocess(self):
        train_df, test_df, train_embeddings, test_embeddings = self._load_data()
        self._clean_data(train_df)
        self._clean_data(test_df)
        self._encode_labels(train_df, test_df)

        if self.size != -1:
            selected_indices = np.random.choice(train_df.index, size=min(self.size, len(train_df)), replace=False)
            train_df = train_df.loc[selected_indices].reset_index(drop=True)

        X = train_embeddings[train_df["embedding_index_critique"].values]
        y_train = self.label_encoder.transform(train_df["human_score"])
        X_train, X_val, y_train, y_val = train_test_split(X, y_train, test_size=0.2, random_state=42)

        X_test = test_embeddings[test_df["embedding_index_critique"].values]
        y_test = self.label_encoder.transform(test_df["human_score"])

        log_p_all_train = self.compute_log_p(train_df)
        log_p_train, log_p_val = train_test_split(log_p_all_train, test_size=0.2, random_state=42)
        log_p_test = self.compute_log_p(test_df)

        X_train = np.hstack([X_train, log_p_train])
        X_val = np.hstack([X_val, log_p_val])
        X_test = np.hstack([X_test, log_p_test])

        if self.standardize:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
            X_test = self.scaler.transform(X_test)

        return map(
            lambda x, dtype: torch.tensor(x, dtype=dtype, device=self.device),
            (X_train, X_val, y_train, y_val, X_test, y_test),
            (torch.float, torch.float, torch.long, torch.long, torch.float, torch.long)
        )

    def train_model(self, X_train, y_train, epochs=None):
        num_classes = len(self.label_encoder.classes_)
        input_dim = X_train.shape[1]
        self.model = MNJudge(input_dim=input_dim, num_classes=num_classes).to(self.device)

        epochs = self.epochs if epochs is None else epochs
        learning_rate = 1.0 / np.sqrt(epochs)
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        batch_size = 128
        dataset_size = X_train.shape[0]

        self.model.train()
        steps_per_epoch = (dataset_size + batch_size - 1) // batch_size

        for epoch in range(epochs):
            permutation = torch.randperm(dataset_size)

            for step in range(steps_per_epoch):
                start_idx = step * batch_size
                end_idx = min((step + 1) * batch_size, dataset_size)
                indices = permutation[start_idx:end_idx]

                X_batch = X_train[indices]
                y_batch = y_train[indices]

                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = F.cross_entropy(logits, y_batch)
                loss += self.model.regularization_loss(self.lambda_theta)
                loss.backward()
                optimizer.step()

        return self.model

    def tune_hyperparameters(self, X_train, y_train, k_folds=5, lambda_values=None):
        best_val_acc = -float('inf')
        best_lambda_theta = self.lambda_theta
        best_state_dict = None
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        for l2_theta in lambda_values:
            fold_accuracies = []

            for train_idx, val_idx in kf.split(X_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                self.lambda_theta = l2_theta
                model = self.train_model(X_tr, y_tr, epochs=500)

                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(X_val)
                    preds = torch.argmax(val_logits, dim=1)
                    acc = accuracy_score(y_val.cpu().numpy(), preds.cpu().numpy())
                    fold_accuracies.append(acc)

            avg_val_acc = np.mean(fold_accuracies)
            self.reg_data[l2_theta] = avg_val_acc
            print(f"Avg Val Accuracy: {avg_val_acc:.4f} @ lambda_theta={l2_theta}")

            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                best_lambda_theta = l2_theta
                best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        print(f"Best lambda_theta={best_lambda_theta} with avg accuracy={best_val_acc:.4f}")
        self.lambda_theta = best_lambda_theta
        self.model.load_state_dict(best_state_dict)

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X)
            probs = F.softmax(logits, dim=1)
        return np.argmax(probs.cpu().numpy(), axis=1), probs.cpu().numpy()

    def evaluate(self, X, y):
        y_pred, proba = self.predict(X)
        expected_score = proba[:, 1] if proba.shape[1] == 2 else np.dot(proba, self.label_encoder.inverse_transform(np.arange(proba.shape[1])).astype(float))
        y = y.cpu().numpy()
        return {
            "MSE": mean_squared_error(y, expected_score),
            "MAE": mean_absolute_error(y, expected_score),
            "R2 Score": r2_score(y, expected_score),
            "Min Prediction": np.min(y_pred),
            "Max Prediction": np.max(y_pred),
            "Accuracy": accuracy_score(y, y_pred),
            "Precision": precision_recall_fscore_support(y, y_pred, average="weighted", zero_division=1)[0],
            "Recall": precision_recall_fscore_support(y, y_pred, average="weighted", zero_division=1)[1],
            "F1 Score": precision_recall_fscore_support(y, y_pred, average="weighted", zero_division=1)[2],
            "Confusion Matrix": confusion_matrix(y, y_pred, labels=np.arange(len(self.label_encoder.classes_))),
            "Class Labels": self.label_encoder.classes_.tolist(),
            "Pearson r": pearsonr(y, expected_score)[0],
            "Spearman ρ": spearmanr(y, expected_score)[0],
            "Kendall τ": kendalltau(y, expected_score)[0],
        }

    def experiment(self):
        X_train, X_val, y_train, y_val, X_test, y_test = self.preprocess()
        lambda_values = [10.0 ** x for x in range(-5, 6)]
        self.tune_hyperparameters(X_train, y_train, k_folds=5, lambda_values=lambda_values)
        X_full = torch.cat([X_train, X_val], dim=0)
        y_full = torch.cat([y_train, y_val], dim=0)
        self.train_model(X_full, y_full)
        return self.evaluate(X_test, y_test)
