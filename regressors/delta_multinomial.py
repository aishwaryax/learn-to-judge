import numpy as np
import pandas as pd
import ast
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import random
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

class MNJudge(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MNJudge, self).__init__()
        # Theta has an extra row because the input X is augmented with a bias column.
        self.Theta = nn.Parameter(torch.zeros(input_dim + 1, num_classes))
    
    def forward(self, X, log_p):
        # Logits are computed as a linear transformation of X plus the external bias log_p.
        logits = X @ self.Theta + log_p
        return F.softmax(logits, dim=1)

class DeltaMultinomial:
    def __init__(self, train_path, test_path, train_emb_path, test_emb_path, size=100, 
                 lr=1e-3, epochs=10000, lambda_l2=0.0, lambda_l1=0.0,
                 use_external_bias=True, device=None, seed=42):
        set_seed(seed)
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
        self.min_epochs = 200

    def compute_log_p(self, df):
        log_p_list = []
        label_classes = self.label_encoder.classes_
        indices = self.label_encoder.inverse_transform(np.arange(len(label_classes)))
        for _, row in df.iterrows():
            tp = row["target_probability"]
            p_vec = np.zeros(len(label_classes))
            # Make sure that the probability vector order matches label_encoder.classes_
            for idx, key in enumerate(indices):
                if key in tp:
                    p_vec[idx] = tp[key]
            p_vec = np.clip(p_vec, 1e-8, 1.0 - 1e-8)
            log_p_list.append(np.log(p_vec))
        return np.stack(log_p_list, axis=0)

    def augment_with_bias(self, X):
        n = X.shape[0]
        bias_col = np.ones((n, 1))
        return np.hstack([X, bias_col])

    def preprocess(self):
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

        total_rows = min(len(train_df), train_embeddings.shape[0])
        sample_size = int((self.size / 100.0) * total_rows)
        selected_indices = np.random.choice(train_df.index, size=sample_size, replace=False)
        train_df = train_df.loc[selected_indices].reset_index(drop=True)

        all_labels = np.concatenate([train_df["human_score"].values, test_df["human_score"].values])
        self.label_encoder.fit(all_labels)

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
                    early_stopping_patience=10):
        # Convert training data to tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        if self.use_external_bias:
            log_p_train_t = torch.tensor(log_p_train, dtype=torch.float32, device=self.device)
        else:
            # Use zeros if external bias is disabled.
            num_classes = self.label_encoder.classes_.shape[0]
            log_p_train_t = torch.zeros((X_train.shape[0], num_classes), dtype=torch.float32, device=self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.long, device=self.device)

        n, input_dim_aug = X_train.shape
        num_classes = log_p_train.shape[1]

        # Initialize the model; note that we pass the unaugmented input dimension.
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

            # Add regularization penalties if set
            if self.lambda_l2 > 0:
                l2_reg = torch.sum(self.model.Theta ** 2)
                loss += self.lambda_l2 * l2_reg
            if self.lambda_l1 > 0:
                l1_reg = torch.sum(torch.abs(self.model.Theta))
                loss += self.lambda_l1 * l1_reg

            loss.backward()
            optimizer.step()

            # Validation phase (if provided)
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

                # Skip early stopping check before minimum epochs are completed.
                if epoch < self.min_epochs:
                    continue

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

        return self.model

    def tune_hyperparameters(self, X_train, y_train, log_p_train, X_val, y_val, log_p_val,
                               lambda_l2_values, patience=10):
        best_acc_score = 0
        best_lambda_l2 = self.lambda_l2
        best_state_dict = None

        for l2 in lambda_l2_values:
            self.lambda_l2 = l2

            model = self.train_model(X_train, y_train, log_p_train, 
                                     X_val, y_val, log_p_val, 
                                     early_stopping_patience=patience)
            val_preds, _ = self.predict(X_val, log_p_val)
            average_val = "weighted" if len(np.unique(y_train)) != 2 else "binary"
            val_acc_score = accuracy_score(y_val, val_preds)
            print(f"Validation Accuracy: {val_acc_score:.4f} @ λ₂: {l2}")
            if val_acc_score > best_acc_score:
                best_acc_score = val_acc_score
                best_lambda_l2 = l2
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"Best regularization: λ₂={best_lambda_l2} with validation accuracy {best_acc_score:.4f}")
        self.lambda_l2 = best_lambda_l2
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

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

    def evaluate(self, X_test, y_test, log_p_test):
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

        encoded_classes = np.arange(len(self.label_encoder.classes_))
        
        cm = confusion_matrix(y_test, y_pred, labels=encoded_classes)

        if proba.shape[1] == 2:
            expected_score = proba[:, 1]
        else:
            class_vals = self.label_encoder.inverse_transform(
                np.arange(proba.shape[1])
            ).astype(float)
            expected_score = np.dot(proba, class_vals)
        pearson_r,  _ = pearsonr(y_test, expected_score)
        spearman_rho, _ = spearmanr(y_test, expected_score)
        kendall_tau, _ = kendalltau(y_test, expected_score)
        

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
            "Confusion Matrix": cm,
            "Class Labels": self.label_encoder.classes_.tolist(),
            "Pearson r": pearson_r,
            "Spearman ρ": spearman_rho,
            "Kendall τ": kendall_tau,
        }
        
    def experiment(self):
        X_train, X_val, y_train, y_val, X_test, y_test, log_p_train, log_p_val, log_p_test = self.preprocess()
        X_train_full = np.vstack([X_train, X_val])
        y_train_full = np.hstack([y_train, y_val])
        log_p_train_full = np.vstack([log_p_train, log_p_val])
        
        lambda_l2_values = [10.0 ** x for x in range(-5, 6)]
        
        self.tune_hyperparameters(X_train, y_train, log_p_train, 
                                  X_val, y_val, log_p_val,
                                  lambda_l2_values, patience=10)
        
        results = self.evaluate(X_test, y_test, log_p_test)
        return results

# Example usage:
# model_wrapper = DeltaMultinomial(
#     train_path='train.csv', test_path='test.csv',
#     train_emb_path='train_emb.npy', test_emb_path='test_emb.npy',
#     size=100, lr=1e-3, epochs=10000, lambda_l2=0.0, lambda_l1=0.0,
#     use_external_bias=False  # Set to False to disable the external bias.
# )
# experiment_results = model_wrapper.experiment()
# print("Evaluation Results:", experiment_results)
