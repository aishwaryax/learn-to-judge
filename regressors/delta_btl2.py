import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
import random
import os
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.stats import pearsonr, spearmanr, kendalltau

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

class DeltaBTL2Judge(nn.Module):
    def __init__(self, input_dim):
        super(DeltaBTL2Judge, self).__init__()
        self.theta = nn.Parameter(torch.zeros(input_dim))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, X_diff, ext_bias):
        z = X_diff @ self.theta + self.bias + ext_bias
        return torch.sigmoid(z)

class DeltaBTL2:
    def __init__(self, train_path, test_path,
                 emb1_path_train, emb2_path_train,
                 emb1_path_test, emb2_path_test,
                 size=100, lr=1e-3, epochs=10000, lambda_l2=0.0,
                 early_stopping_patience=10, use_external_bias=True,
                 device=None, seed=42):
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
        self.model = None
        self.min_epochs = 200

    def preprocess(self):
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        emb_train1 = np.load(self.emb1_path_train)
        emb_train2 = np.load(self.emb2_path_train)
        emb_test1 = np.load(self.emb1_path_test)
        emb_test2 = np.load(self.emb2_path_test)

        for df in [train_df, test_df]:
            df.dropna(subset=["human_score", "llm_score1", "llm_score2", "embedding_index_critique1", "embedding_index_critique2"], inplace=True)
            df["human_score"] = df["human_score"].astype(int)

        total_rows = min(len(train_df), emb_train1.shape[0], emb_train2.shape[0])
        sample_size = int((self.size / 100.0) * total_rows)
        selected_indices = np.random.choice(train_df.index, size=sample_size, replace=False)
        train_df = train_df.loc[selected_indices].reset_index(drop=True)

        X_train = emb_train1[train_df["embedding_index_critique1"].values] - emb_train2[train_df["embedding_index_critique2"].values]
        y_train = train_df["human_score"].values

        X_test = emb_test1[test_df["embedding_index_critique1"].values] - emb_test2[test_df["embedding_index_critique2"].values]
        y_test = test_df["human_score"].values

        p_train = (train_df["llm_score1"].replace(0, 1e-8).values.astype(np.float32) /
                   (train_df["llm_score1"].replace(0, 1e-8).values.astype(np.float32) +
                    train_df["llm_score2"].replace(0, 1e-8).values.astype(np.float32)))
        p_test = (test_df["llm_score1"].replace(0, 1e-8).values.astype(np.float32) /
                  (test_df["llm_score1"].replace(0, 1e-8).values.astype(np.float32) +
                   test_df["llm_score2"].replace(0, 1e-8).values.astype(np.float32)))

        log_odds_train = np.log(p_train / np.clip(1 - p_train, 1e-8, None))
        log_odds_test = np.log(p_test / np.clip(1 - p_test, 1e-8, None))

        X_train, X_val, y_train, y_val, log_odds_train, log_odds_val = train_test_split(
            X_train, y_train, log_odds_train, test_size=0.2, random_state=42)

        return X_train, y_train, log_odds_train, X_val, y_val, log_odds_val, X_test, y_test, log_odds_test

    def tune_hyperparameters(self, X_train, y_train, log_odds_train,
                            X_val, y_val, log_odds_val, lambdas):
        best_acc_score = 0
        best_lambda = None
        best_state_dict = None

        for lambda_l2 in lambdas:
            self.lambda_l2 = lambda_l2
            self.train_model(X_train, y_train, log_odds_train,
                            X_val, y_val, log_odds_val)

            self.model.eval()
            with torch.no_grad():
                X_val_t = torch.tensor(X_val, dtype=torch.float32, device=self.device)
                y_val_t = torch.tensor(y_val, dtype=torch.float32, device=self.device)
                ext_bias_val = torch.tensor(log_odds_val, dtype=torch.float32, device=self.device) if self.use_external_bias else torch.zeros(len(X_val), device=self.device)
                preds = self.model(X_val_t, ext_bias_val).cpu().numpy()
                y_pred = (preds > 0.5).astype(int)
                val_acc_score = accuracy_score(y_val, y_pred)
            print(f"Validation Accuracy: {best_acc_score:.4f} @ λ₂: {lambda_l2}")
            if val_acc_score > best_acc_score:
                best_acc_score = val_acc_score
                best_lambda = lambda_l2
                best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        self.lambda_l2 = best_lambda
        print(f"Best regularization: λ₂={best_lambda} with validation accuracy {best_acc_score:.4f}")
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

    def train_model(self, X_train, y_train, log_odds_train,
                    X_val=None, y_val=None, log_odds_val=None):
        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        log_odds_train_t = torch.tensor(log_odds_train, dtype=torch.float32, device=self.device)

        n, d = X_train.shape

        self.model = DeltaBTL2Judge(input_dim=d).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        best_val_loss = float('inf')
        best_state_dict = None
        epochs_without_improvement = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            optimizer.zero_grad()
            ext_bias_train = log_odds_train_t if self.use_external_bias else torch.zeros(n, device=self.device)
            preds = self.model(X_train_t, ext_bias_train)
            loss = F.binary_cross_entropy(preds.squeeze(), y_train_t)
            if self.lambda_l2 > 0:
                loss += self.lambda_l2 * torch.sum(self.model.theta ** 2)
            loss.backward()
            optimizer.step()

            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=self.device)
                    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=self.device)
                    n_val = X_val.shape[0]
                    ext_bias_val = torch.tensor(log_odds_val, dtype=torch.float32, device=self.device) if self.use_external_bias else torch.zeros(n_val, device=self.device)
                    val_preds = self.model(X_val_t, ext_bias_val)
                    val_loss = F.binary_cross_entropy(val_preds.squeeze(), y_val_t)
                    if self.lambda_l2 > 0:
                        val_loss += self.lambda_l2 * torch.sum(self.model.theta ** 2)

                if epoch < self.min_epochs:
                    continue

                if val_loss.item() < best_val_loss - 1e-4:
                    best_val_loss = val_loss.item()
                    best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= self.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

        self.best_model = self.model
        self.best_lambda_l2 = self.lambda_l2

        return self.model

    def evaluate(self, X_test, y_test, log_odds_test):
        self.model.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test, dtype=torch.float32, device=self.device)
            y_test_t = torch.tensor(y_test, dtype=torch.float32, device=self.device)
            n_test = X_test.shape[0]
            ext_bias_test = torch.tensor(log_odds_test, dtype=torch.float32, device=self.device) if self.use_external_bias else torch.zeros(n_test, device=self.device)
            
            preds = self.model(X_test_t, ext_bias_test).cpu().numpy()
            
            y_pred = (preds > 0.5).astype(int)
            
            acc = accuracy_score(y_test, y_pred)            
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            min_pred, max_pred = np.min(y_pred), np.max(y_pred)
            
            average_val = "weighted" if len(np.unique(y_test)) != 2 else "binary"
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average=average_val, zero_division=1
            )
            
            classes = np.unique(y_test).astype(int)
            cm = confusion_matrix(y_test, y_pred, labels=classes)
            
            pearson_r,  _ = pearsonr(y_test, preds)
            spearman_rho, _ = spearmanr(y_test, preds)
            kendall_tau, _ = kendalltau(y_test, preds)

            return {
                "MSE": mse,
                "MAE": mae,
                "R2 Score": r2,
                "Min Prediction": min_pred,
                "Max Prediction": max_pred,
                "Accuracy": acc,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "Confusion Matrix": cm,
                "Class Labels": classes.tolist(),
                "Pearson r": pearson_r,
                "Spearman ρ": spearman_rho,
                "Kendall τ": kendall_tau,
            }

    def experiment(self, use_external_bias=True):
        X_train, y_train, log_odds_train, \
        X_val, y_val, log_odds_val, \
        X_test, y_test, log_odds_test = self.preprocess()

        lambdas = [10.0 ** x for x in range(-5, 6)]
        
        self.tune_hyperparameters(
            X_train, y_train, log_odds_train,
            X_val, y_val, log_odds_val, lambdas=lambdas
        )

        self.train_model(X_train, y_train, log_odds_train, X_val, y_val, log_odds_val)

        results = self.evaluate(X_test, y_test, log_odds_test)
        return results