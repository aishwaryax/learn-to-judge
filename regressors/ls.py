import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_recall_fscore_support, r2_score
from sklearn.model_selection import train_test_split

class LS:
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
        y = train_df["human_score"].values
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_test = test_embeddings[test_df["embedding_index_critique"].values]
        y_test = test_df["human_score"].values

        return X_train, X_val, y_train, y_val, X_test, y_test

    def tune_hyperparameters(self, X_train, X_val, y_train, y_val, alphas):
        best_mse = float('inf')

        for alpha in alphas:
            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)

            if mse < best_mse:
                best_mse = mse
                self.best_alpha = alpha
                self.model = model

        print(f"Best Alpha (Regularization): {self.best_alpha}")

    def train(self, X_train, y_train):
        self.model = Ridge(alpha=self.best_alpha)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def eval(self, X_test, y_test):
        y_pred = self.predict(X_test)
        min_pred, max_pred = np.min(y_pred), np.max(y_pred)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        y_test_rounded = np.round(y_test)
        y_pred_rounded = np.round(np.clip(y_pred, min_pred, max_pred))

        accuracy = accuracy_score(y_test_rounded, y_pred_rounded)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test_rounded, y_pred_rounded, average='weighted', zero_division=1)

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
        alphas = np.logspace(-6, 6, 10)
        self.tune_hyperparameters(X_train, X_val, y_train, y_val, alphas)
        X_combined = np.concatenate([X_train, X_val])
        y_combined = np.concatenate([y_train, y_val])
        self.train(X_combined, y_combined)
        return self.eval(X_test, y_test)