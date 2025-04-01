import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_recall_fscore_support, r2_score
from sklearn.model_selection import train_test_split
import ast

class DeltaLS:
    def __init__(self, train_path, test_path, train_emb_path, test_emb_path):
        self.train_path = train_path
        self.test_path = test_path
        self.train_emb_path = train_emb_path
        self.test_emb_path = test_emb_path
        self.best_alpha = None
        self.model = None

    def preprocess(self):
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        
        train_embeddings = np.load(self.train_emb_path)
        test_embeddings = np.load(self.test_emb_path)

        for df in [train_df, test_df]:
            df["human_score"] = pd.to_numeric(df["human_score"], errors="coerce")
            df["llm_score"] = pd.to_numeric(df["llm_score"], errors="coerce")
            df.dropna(subset=["human_score", "llm_score"], inplace=True)

        X = train_embeddings[train_df["embedding_index_critique"].values]
        X_bias = train_df[['llm_score']].values.squeeze(-1)
        y = train_df["human_score"].values - X_bias
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_test = test_embeddings[test_df["embedding_index_critique"].values]
        X_test_bias = test_df[['llm_score']].values.squeeze(-1)
        y_test = test_df["human_score"].values

        return X_train, X_val, y_train, y_val, X_test, X_test_bias, y_test

    def tune_hyperparameters(self, X_train, X_val, y_train, y_val, alphas):
        best_mse = float('inf')

        for alpha in alphas:
            model = sm.OLS(y_train, X_train).fit_regularized(alpha=alpha, L1_wt=0)
            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)

            if mse < best_mse:
                best_mse = mse
                self.best_alpha = alpha
                self.model = model

        print(f"Best Alpha (Regularization): {self.best_alpha}")

    def train(self, X_train, y_train):
        self.model = sm.OLS(y_train, X_train).fit_regularized(alpha=self.best_alpha, L1_wt=0)

    def predict(self, X_test, X_test_bias=None):  # submit bias during inference
        if X_test_bias is None:
            return self.model.predict(X_test)
        return self.model.predict(X_test) + X_test_bias

    def eval(self, X_test, X_test_bias, y_test):
        y_pred = self.predict(X_test, X_test_bias)
        min_pred, max_pred = np.min(y_pred), np.max(y_pred)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        y_test_rounded = np.round(y_test)
        y_pred_rounded = np.round(np.clip(y_pred, min_pred, max_pred))

        accuracy = accuracy_score(y_test_rounded, y_pred_rounded)
        precision, recall, f1, support = precision_recall_fscore_support(y_test_rounded, y_pred_rounded, average='weighted', zero_division=1)

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
        X_train, X_val, y_train, y_val, X_test, X_test_bias, y_test = self.preprocess()
        alphas = np.logspace(-6, 6, 10)
        self.tune_hyperparameters(X_train, X_val, y_train, y_val, alphas)
        X_full_train = np.vstack((X_train, X_val))
        y_full_train = np.hstack((y_train, y_val))
        self.train(X_full_train, y_full_train)
        return self.eval(X_test, X_test_bias, y_test)
