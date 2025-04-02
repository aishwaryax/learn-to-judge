import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    accuracy_score, precision_recall_fscore_support,
    roc_curve, auc
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sys

class BTL2:
    def __init__(self, train_path, test_path, train_emb1_path, train_emb2_path, test_emb1_path, test_emb2_path):
        self.best_C = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.train_path = train_path
        self.test_path = test_path
        self.train_emb1_path = train_emb1_path
        self.train_emb2_path = train_emb2_path
        self.test_emb1_path = test_emb1_path
        self.test_emb2_path = test_emb2_path
        
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

        X = train_embeddings1[train_df["embedding_index_critique1"].values] - train_embeddings2[train_df["embedding_index_critique2"].values]
        y_train = train_df["human_score"].values
        X_train, X_val, y_train, y_val = train_test_split(X, y_train, test_size=0.2, random_state=42)

        X_test = test_embeddings1[test_df["embedding_index_critique1"].values] - test_embeddings2[test_df["embedding_index_critique2"].values]
        y_test = test_df["human_score"].values

        return X_train, X_val, y_train, y_val, X_test, y_test

    def tune_hyperparameters(self, X_train, X_val, y_train, y_val, C_values):
        best_accuracy = 0

        for C in C_values:
            model = LogisticRegression(C=C, solver="lbfgs", max_iter=2000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_C = C
                self.model = model

        print(f"Best Regularization (C): {self.best_C}")

    def train(self, X_train, y_train):
        self.model = LogisticRegression(C=self.best_C, penalty="l2", solver="liblinear", max_iter=3000, warm_start=True)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def eval(self, X_test, y_test):
        y_pred = self.predict(X_test)   

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
        
    def experiment(self):
        X_train, X_val, y_train, y_val, X_test, y_test = self.preprocess()
        C_values = [10**i for i in np.arange(-6, 6, 10, dtype=float)]
        self.tune_hyperparameters(X_train, X_val, y_train, y_val, C_values)
        X_full_train = np.vstack((X_train, X_val))
        y_full_train = np.hstack((y_train, y_val))
        self.train(X_full_train, y_full_train)
        return self.eval(X_test, y_test)
