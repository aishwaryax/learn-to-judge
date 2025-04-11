import numpy as np
import pandas as pd
import ast
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import numpy as np
#############################################
# Two-Headed BTL (BTL2) Judge with Learned Bias and External Bias Toggle
#############################################

class DeltaBTL2Judge(nn.Module):
    """
    Two-headed Bradley-Terry-Luce (BTL2) Judge.
    
    Given embeddings for two responses, e₁ and e₂, we compute 
      φ(e) = φ(e₁) - φ(e₂)
    and use the external bias 
      log_odds = log(p/(1-p))
    derived from the original judge's scores.
    
    The predicted probability is given by:
    
        π(e, p; θ) = σ( (φ(e₁) - φ(e₂))^T θ + b + ext_bias )
    
    where b is a learned bias. When θ = 0 and b = 0, if ext_bias is present,
    then the output equals the original probability p.
    """
    def __init__(self, input_dim):
        super(DeltaBTL2Judge, self).__init__()
        self.theta = nn.Parameter(torch.zeros(input_dim))
        self.bias = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, X_diff, ext_bias):
        # X_diff: tensor of shape (n, d)
        # ext_bias: tensor of shape (n,), representing external bias;
        #           if disabled, ext_bias is provided as zeros.
        z = X_diff @ self.theta + self.bias + ext_bias
        return torch.sigmoid(z)

class DeltaBTL2:
    def __init__(self, train_path, test_path,
                 emb1_path_train, emb2_path_train,
                 emb1_path_test, emb2_path_test,
                 size=100, lr=1e-3, epochs=10000, lambda_l2=0.0,
                 early_stopping_patience=10, use_external_bias=True,
                 device=None):
        """
        Wrapper for the two-headed BTL (Δ-BTL2) judge.
        
        Data requirements:
         - The CSV files (train and test) must include:
              • "human_score": Binary label (0 or 1), where 1 indicates the human prefers the first response.
              • "llm_score1" and "llm_score2": Original judge's scores for responses 1 and 2.
              • "embedding_index1" and "embedding_index2": Indices for the embeddings of responses 1 and 2.
         - emb*_path_train/test are paths to NumPy files with embeddings (each row is φ(e) ∈ ℝ^d).
         
        Parameters:
          - size: Percentage of training data to use.
          - lr: Learning rate for Adam.
          - epochs: Maximum training epochs.
          - lambda_l2: L2 regularization strength.
          - early_stopping_patience: Number of epochs with no validation improvement before stopping.
          - use_external_bias: If False, disable the external bias (log odds) by replacing it with zeros.
          - device: Torch device.
        """
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
    
    def preprocess(self):
        # Load CSV and embedding arrays.
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        emb_train1 = np.load(self.emb1_path_train)
        emb_train2 = np.load(self.emb2_path_train)
        emb_test1 = np.load(self.emb1_path_test)
        emb_test2 = np.load(self.emb2_path_test)
        
        # Drop rows with missing required columns and cast human_score to int.
        for df in [train_df, test_df]:
            df.dropna(subset=["human_score", "llm_score1", "llm_score2", "embedding_index_critique1", "embedding_index_critique2"], inplace=True)
            df["human_score"] = df["human_score"].astype(int)
        
        # Sample a subset of training data if needed.
        total_rows = len(train_df)
        sample_size = int((self.size/100.0) * total_rows)
        train_df = train_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
        # Compute the difference of embeddings.
        X_train = emb_train1[train_df["embedding_index_critique1"].values] - emb_train2[train_df["embedding_index_critique2"].values]
        y_train = train_df["human_score"].values  # Binary labels: 0 or 1
        
        X_test = emb_test1[test_df["embedding_index_critique1"].values] - emb_test2[test_df["embedding_index_critique2"].values]
        y_test = test_df["human_score"].values
        
        # Compute original judge probability: p = b/(b + b').
        p_train = (train_df["llm_score1"].replace(0, 1e-8).values.astype(np.float32) /
                   (train_df["llm_score1"].replace(0, 1e-8).values.astype(np.float32) +
                    train_df["llm_score2"].replace(0, 1e-8).values.astype(np.float32)))
        p_test = (test_df["llm_score1"].replace(0, 1e-8).values.astype(np.float32) /
                  (test_df["llm_score1"].replace(0, 1e-8).values.astype(np.float32) +
                   test_df["llm_score2"].replace(0, 1e-8).values.astype(np.float32)))
        
        # Compute external bias: log odds.
        log_odds_train = np.log(p_train / np.clip(1-p_train, 1e-8, None))
        log_odds_test = np.log(p_test / np.clip(1-p_test, 1e-8, None))
        
        # Split training data into training and validation.
        X_train, X_val, y_train, y_val, log_odds_train, log_odds_val = train_test_split(
            X_train, y_train, log_odds_train, test_size=0.2, random_state=42)
        
        return X_train, y_train, log_odds_train, X_val, y_val, log_odds_val, X_test, y_test, log_odds_test
    
    def train_model(self, X_train, y_train, log_odds_train, 
                    X_val=None, y_val=None, log_odds_val=None, 
                    verbose=True):
        # Convert inputs to torch tensors.
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
            preds = self.model(X_train_t, log_odds_train_t if self.use_external_bias else torch.zeros(n, device=self.device))
            loss = F.binary_cross_entropy(preds.squeeze(), y_train_t)
            if self.lambda_l2 > 0:
                loss += self.lambda_l2 * torch.sum(self.model.theta ** 2)
            loss.backward()
            optimizer.step()
            
            if verbose and epoch % (self.epochs // 10) == 0:
                print(f"(BTL2) Epoch {epoch}/{self.epochs}, Training Loss: {loss.item():.4f}")
            
            # Early stopping on validation loss.
            if X_val is not None and y_val is not None and log_odds_val is not None:
                self.model.eval()
                with torch.no_grad():
                    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=self.device)
                    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=self.device)
                    n_val = X_val.shape[0]
                    log_odds_val_t = torch.tensor(log_odds_val, dtype=torch.float32, device=self.device) \
                                     if self.use_external_bias else torch.zeros(n_val, device=self.device)
                    val_preds = self.model(X_val_t, log_odds_val_t)
                    val_loss = F.binary_cross_entropy(val_preds.squeeze(), y_val_t)
                    if self.lambda_l2 > 0:
                        val_loss += self.lambda_l2 * torch.sum(self.model.theta ** 2)
                if verbose and epoch % (self.epochs // 10) == 0:
                    print(f"(BTL2) Epoch {epoch}/{self.epochs}, Validation Loss: {val_loss.item():.4f}")
                
                if val_loss.item() < best_val_loss - 1e-4:
                    best_val_loss = val_loss.item()
                    best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                
                if epochs_without_improvement >= self.early_stopping_patience:
                    print(f"(BTL2) Early stopping triggered at epoch {epoch}")
                    if best_state_dict is not None:
                        self.model.load_state_dict(best_state_dict)
                    break
                
    def predict(self, X, log_odds):
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        n = X.shape[0]
        ext_bias = torch.tensor(log_odds, dtype=torch.float32, device=self.device) if self.use_external_bias \
                    else torch.zeros(n, device=self.device)
        with torch.no_grad():
            probs = self.model(X_t, ext_bias).squeeze().cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        return preds, probs

    def eval(self, X_test, y_test, log_odds_test):
        y_pred, proba = self.predict(X_test, log_odds_test)

        # Regression-like metrics on probabilities
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        min_pred = np.min(y_pred)
        max_pred = np.max(y_pred)

        # Classification metrics
        accuracy_val = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        return {
            "MSE": mse,
            "MAE": mae,
            "R2 Score": r2,
            "Min Prediction": min_pred,
            "Max Prediction": max_pred,
            "Accuracy": accuracy_val,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
        }

    def experiment(self):
        X_train, y_train, log_odds_train, X_val, y_val, log_odds_val, X_test, y_test, log_odds_test = self.preprocess()
        self.train_model(X_train, y_train, log_odds_train,
                         X_val, y_val, log_odds_val, verbose=True)
        results = self.eval(X_test, y_test, log_odds_test)
        return results

# Example usage:
# btl2_wrapper = DeltaBTL2JudgeTorchWrapper(
#     train_path='train_btl2.csv', test_path='test_btl2.csv',
#     emb1_path_train='emb1_train.npy', emb2_path_train='emb2_train.npy',
#     emb1_path_test='emb1_test.npy', emb2_path_test='emb2_test.npy',
#     size=100, lr=1e-3, epochs=10000, lambda_l2=1e-4, early_stopping_patience=10,
#     use_external_bias=True  # Set to False to disable the external bias.
# )
# results_btl2 = btl2_wrapper.experiment()
# print("BTL2 Judge
