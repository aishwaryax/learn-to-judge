import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))
    
def predict(X, theta, log_ratio):
    logits = X.dot(theta) + log_ratio
    preds = sigmoid(logits)
    return preds
    
def log_likelihood(theta, X, y, log_ratio, lambda_l2=0.0, lambda_l1=0.0, epsilon=1e-8):
    predictions = predict(X, theta, log_ratio)
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    ll = -np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    if lambda_l2 > 0:
        ll += lambda_l2 * np.sum(np.square(theta))
    if lambda_l1 > 0:
        ll += lambda_l1 * np.sum(np.abs(theta))
    
    return ll

def log_likelihood_grad(theta, X, y, log_ratio, lambda_l2=0.0, lambda_l1=0.0):
    logits = X.dot(theta) + log_ratio
    predictions = sigmoid(logits)
    gradient = X.T.dot(predictions - y)
    if lambda_l2 > 0:
        gradient += 2 * lambda_l2 * theta
    if lambda_l1 > 0:
        gradient += lambda_l1 * np.sign(theta)
    
    return gradient

class DeltaBTL2:
    def __init__(self, train_path, test_path, train_emb1_path, train_emb2_path, test_emb1_path, test_emb2_path):
        self.train_path = train_path
        self.test_path = test_path
        self.train_emb1_path = train_emb1_path
        self.train_emb2_path = train_emb2_path
        self.test_emb1_path = test_emb1_path
        self.test_emb2_path = test_emb2_path
        self.theta = None
        self.learning_curve = None
        self.best_lambda_l2 = None
        self.best_lambda_l1 = None

    def preprocess(self):
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        
        train_embeddings1 = np.load(self.train_emb1_path)
        train_embeddings2 = np.load(self.train_emb2_path)
        test_embeddings1 = np.load(self.test_emb1_path)
        test_embeddings2 = np.load(self.test_emb2_path)
        
        for df in [train_df, test_df]:
            df.dropna(subset=["human_score"], inplace=True)
            df["human_score"] = df["human_score"].astype(int)
        
        X_train = train_embeddings1[train_df["embedding_index_critique1"].values] - train_embeddings2[train_df["embedding_index_critique2"].values]
        y_train = train_df["human_score"].values
        X_test = test_embeddings1[test_df["embedding_index_critique1"].values] - test_embeddings2[test_df["embedding_index_critique2"].values]
        y_test = test_df["human_score"].values
        
        s_train1 = train_df['llm_score1'].replace(0, 1e-8).values
        s_train2 = train_df['llm_score2'].replace(0, 1e-8).values
        X_bias_train = np.log(s_train1 / s_train2 + 1e-8)
        
        s_test1 = test_df['llm_score1'].replace(0, 1e-8).values
        s_test2 = test_df['llm_score2'].replace(0, 1e-8).values
        X_bias_test = np.log(s_test1 / s_test2 + 1e-8)
        
        X_train, X_val, y_train, y_val, X_bias_train, X_bias_val = train_test_split(
            X_train, y_train, X_bias_train, test_size=0.2, random_state=42
        )
        
        return X_train, y_train, X_val, y_val, X_bias_train, X_bias_val, X_test, y_test, X_bias_test

    def minimize_log_likelihood(self, X, y, log_ratio, lambda_l2=0.0, lambda_l1=0.0):
        initial_theta = np.zeros(X.shape[1])
        
        result = minimize(
            log_likelihood,
            initial_theta,
            args=(X, y, log_ratio, lambda_l2, lambda_l1),
            jac=log_likelihood_grad,
            method='L-BFGS-B',
            options={'disp': True, 'maxiter': 2000},
            bounds=[(None, None)] * len(initial_theta)
        )        
        return result.x

    def train(self, X_train, y_train, log_ratio_train, lambda_l2=0.0, lambda_l1=0.0):
        self.theta = self.minimize_log_likelihood(X_train, y_train, log_ratio_train, lambda_l2, lambda_l1)
    
    def predict(self, X, log_ratio):
        preds = predict(X, self.theta, log_ratio)
        return (preds >= 0.5).astype(int), preds
    
    def eval(self, X_test, y_test, log_ratio_test):
        y_pred, y_pred_probs = self.predict(X_test, log_ratio_test)
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        min_pred, max_pred = np.min(y_pred), np.max(y_pred)
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

    def tune_hyperparameters(self, X_train, y_train, X_val, y_val, log_ratio_train, log_ratio_val, lambda_values):
        best_accuracy = 0
        best_lambda_l2 = None
        best_lambda_l1 = None
        
        for lambda_l2 in lambda_values:
            for lambda_l1 in lambda_values:
                self.train(X_train, y_train, log_ratio_train, lambda_l2, lambda_l1)
                predictions = self.predict(X_val, log_ratio_val)[0]
                accuracy = accuracy_score(y_val, predictions)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_lambda_l2 = lambda_l2
                    best_lambda_l1 = lambda_l1
        
        print(f"Best Lambda L2: {best_lambda_l2}, Best Lambda L1: {best_lambda_l1}")
        self.best_lambda_l2 = best_lambda_l2
        self.best_lambda_l1 = best_lambda_l1

    def experiment(self):
        X_train, y_train, X_val, y_val, X_bias_train, X_bias_val, X_test, y_test, X_bias_test = self.preprocess()        
        lambda_values = np.logspace(-6, 6, num=10)
        self.tune_hyperparameters(X_train, y_train, X_val, y_val, X_bias_train, X_bias_val, lambda_values)        
        X_train_full = np.vstack([X_train, X_val])
        y_train_full = np.hstack([y_train, y_val])
        bias_train_full = np.hstack([X_bias_train, X_bias_val]).flatten()
        X_bias_test = X_bias_test.flatten()
        print(bias_train_full.shape)
        self.train(X_train_full, y_train_full, bias_train_full, self.best_lambda_l2, self.best_lambda_l1)        
        return self.eval(X_test, y_test, X_bias_test)
