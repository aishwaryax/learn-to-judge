import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import os
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

class TorchDeltaLS(nn.Module):
    def __init__(self, input_dim, lambda_l2=0.1):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)
        self.lambda_l2 = lambda_l2
    def forward(self, x):
        return self.linear(x).squeeze(-1)

    def l2_regularization(self):
        return self.lambda_l2 * torch.sum(self.linear.weight ** 2)

class DeltaLS:
    def __init__(self, train_path, test_path, train_emb_path, test_emb_path, size=100, use_external_bias=True, seed=42):
        set_seed(seed)
        self.train_path = train_path
        self.test_path = test_path
        self.train_emb_path = train_emb_path
        self.test_emb_path = test_emb_path
        self.size = size
        self.use_external_bias = use_external_bias
        self.model = None
        self.best_lambda_l2 = None
        self.epochs = 10000
        self.min_epochs = 200
        self.early_stopping_patience = 10
        self.lr = 1e-3

    def preprocess(self):
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        
        train_embeddings = np.load(self.train_emb_path)
        test_embeddings = np.load(self.test_emb_path)

        for df in [train_df, test_df]:
            df.dropna(subset=["human_score"], inplace=True)
            df["human_score"] = df["human_score"].astype(int)
            df["llm_score"] = df["llm_score"].astype(int)

        total_rows = len(train_df)
        sample_size = int((self.size / 100.0) * total_rows)
        selected_indices = np.random.choice(train_df.index, size=sample_size, replace=False)
        train_df = train_df.loc[selected_indices].reset_index(drop=True)

        def prepare_features(df, embeddings):
            X = embeddings[df["embedding_index_critique"].values]
            if self.use_external_bias:
                bias = df[["llm_score"]].values
                X = np.hstack([X, bias])
            y = df["human_score"].values
            return X, y

        X, y = prepare_features(train_df, train_embeddings)
        X_test, y_test = prepare_features(test_df, test_embeddings)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        return map(torch.tensor, (X_train, X_val, y_train, y_val, X_test, y_test))

    def tune_hyperparameters(self, X_train, X_val, y_train, y_val, lambda_l2_values):
        best_loss = float('inf')
        input_dim = X_train.shape[1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        best_lambda_l2 = None
        best_model_state = None

        for lambda_l2 in lambda_l2_values:
            model = TorchDeltaLS(input_dim, lambda_l2).to(device)

            model = self.train(X_train, X_val, y_train, y_val, lambda_l2=lambda_l2)  # One step training to evaluate

            with torch.no_grad():
                preds = model(X_val.to(device))
                loss = F.mse_loss(preds, y_val.to(device)) + model.l2_regularization()
            print(f"Validation MSE: {loss:.4f} @ λ₂: {lambda_l2}")
            if loss < best_loss:
                best_loss = loss
                best_lambda_l2 = lambda_l2
                best_model_state = model.state_dict()
        print(f"Best regularization: λ₂={best_lambda_l2} with MSE {best_loss:.4f}")
        self.best_lambda_l2 = best_lambda_l2
        self.model = TorchDeltaLS(input_dim, self.best_lambda_l2).to(device)
        self.model.load_state_dict(best_model_state)

    def train(self, X_train, X_val, y_train, y_val, lambda_l2=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = X_train.shape[1]
        self.model = TorchDeltaLS(input_dim, lambda_l2).to(device)
        if lambda_l2 is not None:
            self.model.lambda_l2 = lambda_l2

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        best_loss = float('inf')
        early_stop_counter = 0
        best_model_state = None

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            preds = self.model(X_train.to(device))
            loss = F.mse_loss(preds, y_train.to(device)) + self.model.l2_regularization()
            loss.backward()
            optimizer.step()

            if epoch < self.min_epochs:
                continue

            loss_val = loss.item()
            if loss_val < best_loss - 1e-4:
                best_loss = loss_val
                best_model_state = self.model.state_dict()
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= self.early_stopping_patience:
                    print(f"Early stopping during training at epoch {epoch}")
                    break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return self.model

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(X.to(self.model.linear.weight.device)).cpu().numpy()

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        min_pred, max_pred = np.min(y_pred), np.max(y_pred)

        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        y_true_rounded = np.round(y)
        y_pred_rounded = np.round(np.clip(y_pred, min_pred, max_pred))

        accuracy = accuracy_score(y_true_rounded, y_pred_rounded)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_rounded, y_pred_rounded, average='weighted', zero_division=0
        )

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
        }

    def experiment(self):
        X_train, X_val, y_train, y_val, X_test, y_test = self.preprocess()

        X_train, X_val, X_test = X_train.float(), X_val.float(), X_test.float()
        y_train, y_val, y_test = y_train.float(), y_val.float(), y_test.float()

        lambda_l2_values = [10.0 ** x for x in range(-5, 6)]

        self.tune_hyperparameters(X_train, X_val, y_train, y_val, lambda_l2_values)

        self.model = self.train(X_train, X_val, y_train, y_val, lambda_l2=self.best_lambda_l2)

        results = self.evaluate(X_test, y_test)
        return results
