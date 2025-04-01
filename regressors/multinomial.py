import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    accuracy_score, precision_recall_fscore_support
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sys

class Multinomial:
    def __init__(self, train_path, test_path, train_emb_path, test_emb_path):
        self.best_C = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.train_path = train_path
        self.test_path = test_path
        self.train_emb_path = train_emb_path
        self.test_emb_path = test_emb_path
        
    def preprocess(self):
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        train_embeddings = np.load(self.train_emb_path)
        test_embeddings = np.load(self.test_emb_path)

        for df in [train_df, test_df]:
            df.dropna(subset=["human_score"], inplace=True)
            df["human_score"] = df["human_score"].astype(int).astype(str)

        all_labels = np.concatenate([train_df["human_score"].values, test_df["human_score"].values])
        self.label_encoder.fit(all_labels)

        y_train = self.label_encoder.transform(train_df["human_score"].values)
        y_test = self.label_encoder.transform(test_df["human_score"].values)

        X = train_embeddings[train_df["embedding_index_critique"].values]
        X_train, X_val, y_train, y_val = train_test_split(X, y_train, test_size=0.2, random_state=42)

        X_test = test_embeddings[test_df["embedding_index_critique"].values]

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
        self.model = LogisticRegression(C=self.best_C, multi_class="multinomial", solver="lbfgs", max_iter=1000)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def eval(self, X_test, y_test):
        y_pred = self.predict(X_test)

        y_test_numeric = self.label_encoder.inverse_transform(y_test)
        y_pred_numeric = self.label_encoder.inverse_transform(y_pred)

        mse = mean_squared_error(y_test_numeric, y_pred_numeric)
        mae = mean_absolute_error(y_test_numeric, y_pred_numeric)
        r2 = r2_score(y_test_numeric, y_pred_numeric)

        min_pred, max_pred = np.min(y_pred_numeric), np.max(y_pred_numeric)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=1)

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
            "Support": support
        }
        
    def experiment(self):
        X_train, X_val, y_train, y_val, X_test, y_test = self.preprocess()
        C_values = [10**i for i in np.arange(-6, 6, 1, dtype=float)]
        self.tune_hyperparameters(X_train, X_val, y_train, y_val, C_values)
        X_full_train = np.vstack((X_train, X_val))
        y_full_train = np.hstack((y_train, y_val))
        self.train(X_full_train, y_full_train)
        return self.eval(X_test, y_test)
