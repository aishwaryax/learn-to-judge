import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    accuracy_score, precision_recall_fscore_support
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize
import ast

def softmax(z):
    z = np.clip(z, -500, 500)
    exps = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def log_likelihood(theta_flat, X, y, log_ratio, lambda_l2=0.0, lambda_l1=0.0, epsilon=1e-8):
    n, d = X.shape
    K = log_ratio.shape[1]
    theta = theta_flat.reshape(d, K)
    logits = X.dot(theta) + log_ratio
    probs = softmax(logits)
    
    ll = -np.sum(np.log(probs[np.arange(n), y] + epsilon))
    
    if lambda_l2 > 0:
        ll += lambda_l2 * np.sum(np.square(theta))
    
    if lambda_l1 > 0:
        ll += lambda_l1 * np.sum(np.abs(theta))
    
    return ll

def log_likelihood_grad(theta_flat, X, y, log_ratio, lambda_l2=0.0, lambda_l1=0.0):
    n, d = X.shape
    K = log_ratio.shape[1]
    theta = theta_flat.reshape(d, K)
    logits = X.dot(theta) + log_ratio
    probs = softmax(logits)
    
    y_onehot = np.zeros_like(probs)
    y_onehot[np.arange(n), y] = 1
    
    grad = X.T.dot(probs - y_onehot)    
    if lambda_l2 > 0:
        grad += 2 * lambda_l2 * theta    
    if lambda_l1 > 0:
        grad += lambda_l1 * np.sign(theta)
    
    return grad.flatten()

class DeltaMultinomial:
    def __init__(self, train_path, test_path, train_emb_path, test_emb_path):
        self.best_lambda_l2 = None
        self.best_lambda_l1 = None
        self.theta = None
        self.label_encoder = LabelEncoder()
        self.target_keys = None
        self.train_path = train_path
        self.test_path = test_path
        self.train_emb_path = train_emb_path
        self.test_emb_path = test_emb_path

    def compute_bias(self, df):
        bias_list = []
        for _, row in df.iterrows():
            if self.target_keys is None:
                self.target_keys = sorted(row["target_probability"].keys())
            tp = row["target_probability"]
            pi0 = np.zeros(len(self.label_encoder.classes_))
            for key, value in row["target_probability"].items():
                index = int(self.label_encoder.inverse_transform([int(value)])[0])
                pi0[index] = row["self_consistency_score"] * value
            pi0 = np.clip(pi0, 1e-8, 1.0)
            bias_list.append(np.log(pi0))
        return np.stack(bias_list, axis=0)

    def preprocess(self):
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        train_embeddings = np.load(self.train_emb_path)
        test_embeddings = np.load(self.test_emb_path)

        for df in [train_df, test_df]:
            df.dropna(subset=["human_score"], inplace=True)
            df["human_score"] = df["human_score"].astype(int).astype(str)
            df["target_probability"] = df["target_probability"].apply(ast.literal_eval)

        all_labels = np.concatenate([train_df["human_score"].values, test_df["human_score"].values])
        self.label_encoder.fit(all_labels)

        X = train_embeddings[train_df["embedding_index_critique"].values]
        y_train = self.label_encoder.transform(train_df["human_score"].values)
        y_test = self.label_encoder.transform(test_df["human_score"].values)
        
        X_train, X_val, y_train, y_val = train_test_split(X, y_train, test_size=0.2, random_state=42)
        X_test = test_embeddings[test_df["embedding_index_critique"].values]

        bias_all_train = self.compute_bias(train_df)
        bias_train, bias_val = train_test_split(bias_all_train, test_size=0.2, random_state=42)
        bias_test = self.compute_bias(test_df)

        return X_train, X_val, y_train, y_val, X_test, y_test, bias_train, bias_val, bias_test

    def tune_hyperparameters(self, X_train, X_val, y_train, y_val, bias_train, bias_val, C_values):
        best_accuracy = 0
        best_l2 = None
        best_l1 = None
        for l2 in C_values:
            for l1 in C_values:
                theta_candidate = self._train_internal(X_train, y_train, bias_train, lambda_l2=l2, lambda_l1=l1)
                preds = self._predict_internal(X_val, bias_val, theta_candidate)
                y_pred = np.argmax(preds, axis=1)
                accuracy = accuracy_score(y_val, y_pred)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_l2 = l2
                    best_l1 = l1
                    self.theta = theta_candidate
        
        print(f"Best L2 lambda: {best_l2}, Best L1 lambda: {best_l1}")
        self.best_lambda_l2 = best_l2
        self.best_lambda_l1 = best_l1

    def _train_internal(self, X_train, y_train, bias_train, lambda_l2=0.0, lambda_l1=0.0):
        n, d = X_train.shape
        K = bias_train.shape[1]
        initial_theta = np.zeros(d * K)
        result = minimize(
            log_likelihood,
            initial_theta,
            args=(X_train, y_train, bias_train, lambda_l2, lambda_l1),
            jac=log_likelihood_grad,
            method='L-BFGS-B',
            options={'disp': True, 'maxiter': 2000}
        )
        return result.x.reshape(d, K)

    def train(self, X_train, y_train, bias_train):
        self.theta = self._train_internal(X_train, y_train, bias_train, self.best_lambda_l2, self.best_lambda_l1)

    def _predict_internal(self, X, bias, theta):
        logits = X.dot(theta) + bias
        return softmax(logits)

    def predict(self, X_test, bias_test):
        probs = self._predict_internal(X_test, bias_test, self.theta)
        return np.argmax(probs, axis=1)

    def eval(self, X_test, y_test, bias_test):
        y_pred = self.predict(X_test, bias_test)
        
        y_test_numeric = self.label_encoder.inverse_transform(y_test)
        y_pred_numeric = self.label_encoder.inverse_transform(y_pred)

        mse = mean_squared_error(y_test_numeric, y_pred_numeric)
        mae = mean_absolute_error(y_test_numeric, y_pred_numeric)
        r2 = r2_score(y_test_numeric, y_pred_numeric)
        min_pred, max_pred = np.min(y_pred_numeric), np.max(y_pred_numeric)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=1
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
            "F1 Score": f1
        }
        
    def experiment(self):
        X_train, X_val, y_train, y_val, X_test, y_test, bias_train, bias_val, bias_test = self.preprocess()
        X_train_full = np.vstack([X_train, X_val])
        y_train_full = np.hstack([y_train, y_val])
        bias_train_full = np.vstack([bias_train, bias_val])
        C_values = np.logspace(-6, 6, num=10)
        self.tune_hyperparameters(X_train_full, X_val, y_train_full, y_val, bias_train_full, bias_val, C_values)
        self.train(X_train_full, y_train_full, bias_train_full)
        return self.eval(X_test, y_test, bias_test)