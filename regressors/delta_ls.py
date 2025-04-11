import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

class TorchDeltaLS(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)  # Always use learned bias

    def forward(self, x):
        return self.linear(x).squeeze(-1)

class DeltaLS:
    def __init__(self, train_path, test_path, train_emb_path, test_emb_path, size=100, use_external_bias=True):
        self.train_path = train_path
        self.test_path = test_path
        self.train_emb_path = train_emb_path
        self.test_emb_path = test_emb_path
        self.size = size
        self.use_external_bias = use_external_bias
        self.model = None
        self.best_alpha = None

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

    def tune_hyperparameters(self, X_train, X_val, y_train, y_val, alphas, epochs=100, lr=1e-3, patience=10):
        best_loss = float('inf')
        input_dim = X_train.shape[1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for alpha in alphas:
            model = TorchDeltaLS(input_dim).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            early_stop_counter = 0
            best_model_state = None

            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                preds = model(X_train.to(device))
                loss = F.mse_loss(preds, y_train.to(device)) + alpha * torch.norm(model.linear.weight, p=2) ** 2
                loss.backward()
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_preds = model(X_val.to(device))
                    val_loss = F.mse_loss(val_preds, y_val.to(device)).item()

                if val_loss < best_loss:
                    best_loss = val_loss
                    self.model = model
                    self.best_alpha = alpha
                    best_model_state = model.state_dict()
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= patience:
                        print(f"Early stopping at epoch {epoch} for alpha={alpha}")
                        break

            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)

        print(f"Best alpha: {self.best_alpha}")

    def train(self, X, y, epochs=100, lr=1e-3, patience=10):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        best_loss = float('inf')
        early_stop_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            optimizer.zero_grad()
            preds = self.model(X.to(device))
            loss = F.mse_loss(preds, y.to(device)) + self.best_alpha * torch.norm(self.model.linear.weight, p=2) ** 2
            loss.backward()
            optimizer.step()

            # Track loss for early stopping
            loss_val = loss.item()
            if loss_val < best_loss:
                best_loss = loss_val
                best_model_state = self.model.state_dict()
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f"Early stopping during final training at epoch {epoch}")
                    break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)


    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(X.to(self.model.linear.weight.device)).cpu().numpy()

    def eval(self, X, y):
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

        alphas = [10.0 ** i for i in np.arange(-3, 4)]
        self.tune_hyperparameters(X_train, X_val, y_train, y_val, alphas)

        X_full_train = torch.cat([X_train, X_val], dim=0)
        y_full_train = torch.cat([y_train, y_val], dim=0)
        self.train(X_full_train, y_full_train)

        return self.eval(X_test, y_test)
