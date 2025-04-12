import numpy as np
import pandas as pd
import ast
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    accuracy_score, precision_recall_fscore_support
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MNJudge(nn.Module):
    """
    PyTorch implementation of the Multinomial Judge (Δ-MN judge).

    Implements the model:
        π(s | e, p; Θ) = softmax( [φ(e) 1] · Θ + log p )
    where:
      - [φ(e) 1] is the augmented embedding (with a bias term),
      - Θ (Theta) is a learned parameter matrix of shape [(d+1), |S|],
      - log p (log_p) is the log probability vector from the original judge.
    
    When Θ is zero, the predicted distribution becomes the original judge's probability p.
    """
    def __init__(self, input_dim, num_classes):
        super(MNJudge, self).__init__()
        # Θ has shape ((input_dim + 1), num_classes) with the last row learning a bias.
        self.Theta = nn.Parameter(torch.zeros(input_dim + 1, num_classes))
    
    def forward(self, X, log_p):
        # X: tensor of shape (n, input_dim+1) with bias column already appended.
        # log_p: tensor of shape (n, num_classes)
        logits = X @ self.Theta + log_p
        return F.softmax(logits, dim=1)

class DeltaMultinomial:
    def __init__(self, train_path, test_path, train_emb_path, test_emb_path, size=100, 
                 lr=1e-3, epochs=10000, lambda_l2=0.0, lambda_l1=0.0,
                 use_external_bias=True, device=None):
        """
        Wrapper for training and evaluating the Δ-MN Judge.

        Parameters:
          - train_path, test_path: Paths to CSV files containing the data.
          - train_emb_path, test_emb_path: Paths to NumPy files with the embeddings (φ(e)).
          - size: Percentage of the training dataset to use.
          - lr: Learning rate for the Adam optimizer.
          - epochs: Maximum number of training epochs.
          - lambda_l2, lambda_l1: Regularization strengths.
          - use_external_bias: If False, the external bias (log p) is disabled.
          - device: Torch device (CPU or CUDA).
        """
        self.train_path = train_path
        self.test_path = test_path
        self.train_emb_path = train_emb_path
        self.test_emb_path = test_emb_path
        self.size = size
        self.lr = lr
        self.epochs = epochs
        self.lambda_l2 = lambda_l2
        self.lambda_l1 = lambda_l1
        self.use_external_bias = use_external_bias
        self.label_encoder = LabelEncoder()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def compute_log_p(self, df):
        """
        Computes log p for each sample from the original judge's probability distribution.
        For each row, it extracts the probability vector (field "target_probability"),
        clips it to avoid numerical issues, and returns its log.
        """
        log_p_list = []
        label_classes = self.label_encoder.classes_
        # Get the original string labels from indices.
        indices = self.label_encoder.inverse_transform(np.arange(len(label_classes)))
        for _, row in df.iterrows():
            tp = row["target_probability"]
            p_vec = np.zeros(len(label_classes))
            for idx, key in enumerate(indices):
                if key in tp:
                    p_vec[idx] = tp[key]
            p_vec = np.clip(p_vec, 1e-8, 1.0 - 1e-8)
            log_p_list.append(np.log(p_vec))
        return np.stack(log_p_list, axis=0)

    def augment_with_bias(self, X):
        """
        Append a column of ones to X to account for the bias in φ(e).
        """
        n = X.shape[0]
        bias_col = np.ones((n, 1))
        return np.hstack([X, bias_col])

    def preprocess(self):
        # Load CSVs and embedding arrays.
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        train_embeddings = np.load(self.train_emb_path)
        test_embeddings = np.load(self.test_emb_path)

        for df in [train_df, test_df]:
            df.dropna(subset=["human_score"], inplace=True)
            df.dropna(subset=["llm_score"], inplace=True)
            df["human_score"] = df["human_score"].astype(int)
            df["llm_score"] = df["llm_score"].astype(int)
            df["target_probability"] = df["target_probability"].apply(ast.literal_eval)

        # Sample a subset of training data based on the given size.
        total_rows = len(train_df)
        sample_size = int((self.size / 100.0) * total_rows)
        selected_indices = np.random.choice(train_df.index, size=sample_size, replace=False)
        train_df = train_df.loc[selected_indices].reset_index(drop=True)

        # Fit label encoder with human scores from both train and test data.
        all_labels = np.concatenate([train_df["human_score"].values, test_df["human_score"].values])
        self.label_encoder.fit(all_labels)

        # Get embeddings and augment features with a bias column.
        X = train_embeddings[train_df["embedding_index_critique"].values]
        X = self.augment_with_bias(X)
        y_train = self.label_encoder.transform(train_df["human_score"].values)
        y_test = self.label_encoder.transform(test_df["human_score"].values)

        X_train, X_val, y_train, y_val = train_test_split(X, y_train, test_size=0.2, random_state=42)

        X_test = test_embeddings[test_df["embedding_index_critique"].values]
        X_test = self.augment_with_bias(X_test)

        log_p_all_train = self.compute_log_p(train_df)
        log_p_train, log_p_val = train_test_split(log_p_all_train, test_size=0.2, random_state=42)
        log_p_test = self.compute_log_p(test_df)

        return X_train, X_val, y_train, y_val, X_test, y_test, log_p_train, log_p_val, log_p_test

    def train_model(self, X_train, y_train, log_p_train, 
                    X_val=None, y_val=None, log_p_val=None, 
                    early_stopping_patience=10, verbose=True):
        """
        Train the model using the Adam optimizer and optionally perform early stopping.
        If X_val, y_val, and log_p_val are provided, the training loop monitors validation loss.
        Early stopping is triggered if no improvement is seen for 'early_stopping_patience' epochs.
        """
        # Convert NumPy arrays into PyTorch tensors.
        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        # If external bias is disabled, replace log_p with zeros.
        if self.use_external_bias:
            log_p_train_t = torch.tensor(log_p_train, dtype=torch.float32, device=self.device)
        else:
            log_p_train_t = torch.zeros((X_train.shape[0], self.label_encoder.classes_.shape[0]), 
                                          dtype=torch.float32, device=self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.long, device=self.device)

        n, input_dim_aug = X_train.shape  # input_dim_aug = original dimension d + 1 for bias.
        num_classes = log_p_train.shape[1]

        # Initialize the MN judge model (pass original input dimension without bias).
        self.model = MNJudge(input_dim=input_dim_aug - 1, num_classes=num_classes).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        best_val_loss = float('inf')
        best_state_dict = None
        epochs_without_improvement = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            optimizer.zero_grad()

            logits = X_train_t @ self.model.Theta + log_p_train_t
            log_probs = F.log_softmax(logits, dim=1)
            loss = F.nll_loss(log_probs, y_train_t)

            if self.lambda_l2 > 0:
                l2_reg = torch.sum(self.model.Theta ** 2)
                loss += self.lambda_l2 * l2_reg
            if self.lambda_l1 > 0:
                l1_reg = torch.sum(torch.abs(self.model.Theta))
                loss += self.lambda_l1 * l1_reg

            loss.backward()
            optimizer.step()

            if verbose and epoch % (self.epochs // 10) == 0:
                print(f"Epoch {epoch}/{self.epochs}, Training Loss: {loss.item():.4f}")

            if X_val is not None and y_val is not None and log_p_val is not None:
                self.model.eval()
                with torch.no_grad():
                    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=self.device)
                    if self.use_external_bias:
                        log_p_val_t = torch.tensor(log_p_val, dtype=torch.float32, device=self.device)
                    else:
                        log_p_val_t = torch.zeros((X_val.shape[0], num_classes),
                                                  dtype=torch.float32, device=self.device)
                    y_val_t = torch.tensor(y_val, dtype=torch.long, device=self.device)
                    val_logits = X_val_t @ self.model.Theta + log_p_val_t
                    val_log_probs = F.log_softmax(val_logits, dim=1)
                    val_loss = F.nll_loss(val_log_probs, y_val_t)
                    if self.lambda_l2 > 0:
                        l2_reg = torch.sum(self.model.Theta ** 2)
                        val_loss += self.lambda_l2 * l2_reg
                    if self.lambda_l1 > 0:
                        l1_reg = torch.sum(torch.abs(self.model.Theta))
                        val_loss += self.lambda_l1 * l1_reg

                if verbose and epoch % (self.epochs // 10) == 0:
                    print(f"Epoch {epoch}/{self.epochs}, Validation Loss: {val_loss.item():.4f}")

                if val_loss.item() < best_val_loss - 1e-4:
                    best_val_loss = val_loss.item()
                    best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    if best_state_dict is not None:
                        self.model.load_state_dict(best_state_dict)
                    break

    def tune_hyperparameters(self, X_train, y_train, log_p_train, X_val, y_val, log_p_val,
                             lambda_l2_values, lambda_l1_values, patience=10):
        """
        Tunes the regularization parameters by evaluating validation accuracy.
        For each candidate pair (lambda_l2, lambda_l1), train a model with early stopping and evaluate on validation data.
        Stores the best regularization parameters and reloads the corresponding model weights.
        """
        best_val_acc = 0
        best_lambda_l2 = self.lambda_l2
        best_lambda_l1 = self.lambda_l1
        best_state_dict = None

        original_epochs = self.epochs
        self.epochs = max(100, original_epochs // 10)

        for l2 in lambda_l2_values:
            for l1 in lambda_l1_values:
                self.lambda_l2 = l2
                self.lambda_l1 = l1

                self.train_model(X_train, y_train, log_p_train, 
                                 X_val, y_val, log_p_val, 
                                 early_stopping_patience=patience,
                                 verbose=False)
                val_preds, val_probs = self.predict(X_val, log_p_val)
                val_acc = accuracy_score(y_val, val_preds)
                print(f"Validation Acc: {val_acc:.4f} @ λ₂: {l2}, λ₁: {l1}")
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_lambda_l2 = l2
                    best_lambda_l1 = l1
                    best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        print(f"Best regularization: λ₂={best_lambda_l2}, λ₁={best_lambda_l1} with validation accuracy {best_val_acc:.4f}")
        self.lambda_l2 = best_lambda_l2
        self.lambda_l1 = best_lambda_l1
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
        self.epochs = original_epochs

    def _predict_internal(self, X, log_p):
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        if self.use_external_bias:
            log_p_t = torch.tensor(log_p, dtype=torch.float32, device=self.device)
        else:
            num_classes = self.label_encoder.classes_.shape[0]
            log_p_t = torch.zeros((X.shape[0], num_classes), dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            logits = X_t @ self.model.Theta + log_p_t
            probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def predict(self, X_test, log_p_test):
        probs = self._predict_internal(X_test, log_p_test)
        return np.argmax(probs, axis=1), probs

    def eval(self, X_test, y_test, log_p_test):
        y_pred, proba = self.predict(X_test, log_p_test)
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        min_pred, max_pred = np.min(y_pred), np.max(y_pred)
        
        average_val = "weighted" if len(np.unique(y_test)) != 2 else "binary"
        accuracy_val = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average=average_val, zero_division=1
        )
        
        return {
            "MSE": mse,
            "MAE": mae,
            "R2 Score": r2,
            "Min Prediction": min_pred,
            "Max Prediction": max_pred,
            "Accuracy": accuracy_val,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        }
        
    def experiment(self):
        X_train, X_val, y_train, y_val, X_test, y_test, log_p_train, log_p_val, log_p_test = self.preprocess()
        X_train_full = np.vstack([X_train, X_val])
        y_train_full = np.hstack([y_train, y_val])
        log_p_train_full = np.vstack([log_p_train, log_p_val])
        
        lambda_l2_values = [10**i for i in np.arange(-3, 4, 1, dtype=float)]
        lambda_l1_values = [0]
        self.tune_hyperparameters(X_train, y_train, log_p_train, 
                                  X_val, y_val, log_p_val,
                                  lambda_l2_values, lambda_l1_values, patience=10)
        
        self.train_model(X_train_full, y_train_full, log_p_train_full,
                         X_val, y_val, log_p_val, early_stopping_patience=10, verbose=True)
        results = self.eval(X_test, y_test, log_p_test)
        return results

# Example usage:
# model_wrapper = DeltaMultinomialV2(
#     train_path='train.csv', test_path='test.csv',
#     train_emb_path='train_emb.npy', test_emb_path='test_emb.npy',
#     size=100, lr=1e-3, epochs=10000, lambda_l2=0.0, lambda_l1=0.0,
#     use_external_bias=False  # Set to False to disable the external bias.
# )
# experiment_results = model_wrapper.experiment()
# print("Evaluation Results:", experiment_results)
