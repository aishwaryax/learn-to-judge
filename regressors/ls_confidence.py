import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_recall_fscore_support, r2_score
from sklearn.model_selection import train_test_split

class LS_conf:
    def __init__(self, train_path, test_path, train_emb_path, test_emb_path, confidence_threshold=0.5):
        self.train_path = train_path
        self.test_path = test_path
        self.train_emb_path = train_emb_path
        self.test_emb_path = test_emb_path
        self.best_alpha = None
        self.model = None
        self.confidence_threshold = confidence_threshold
        self.Sigma = None  # Covariance matrix

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
        # Compute covariance matrix
        I_d = np.eye(X_train.shape[1])
        self.Sigma = np.linalg.inv(X_train.T @ X_train + self.best_alpha * I_d)

    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def compute_confidence_width(self, X_test):
        """Compute confidence width for each test example."""
        confidence_widths = np.sqrt(np.einsum('ij,ji->i', X_test @ self.Sigma, X_test.T))
        return confidence_widths
    
    def eval(self, X_test, y_test):
        y_pred = self.predict(X_test)
        confidence_widths = self.compute_confidence_width(X_test)
        #print max and min confidence widths
        print(f"Max confidence width: {confidence_widths.max()}")
        print(f"Min confidence width: {confidence_widths.min()}")
        confident_indices = confidence_widths < self.confidence_threshold
        num_total_samples = len(y_test)
        num_excluded = num_total_samples - confident_indices.sum()
        percent_excluded = (num_excluded / num_total_samples) * 100

        mse_all = mean_squared_error(y_test, y_pred)
        mse_confident = mean_squared_error(y_test[confident_indices], y_pred[confident_indices]) if confident_indices.sum() > 0 else 0
        confidence_error_correlation = np.corrcoef(np.abs(y_test - y_pred), confidence_widths)[0, 1]

        return {
            "Total Samples": num_total_samples,
            "Excluded Samples (Human Review)": num_excluded,
            "Percentage Excluded": percent_excluded,
            "MSE (All Predictions)": mse_all,
            "MSE (Confident Predictions)": mse_confident,
            "Confidence Threshold": self.confidence_threshold,
            "Confidence Error Correlation": confidence_error_correlation
        }
    
    def experiment(self):
        X_train, X_val, y_train, y_val, X_test, y_test = self.preprocess()
        alphas = np.logspace(-6, 6, 10)
        self.tune_hyperparameters(X_train, X_val, y_train, y_val, alphas)
        X_combined = np.concatenate([X_train, X_val])
        y_combined = np.concatenate([y_train, y_val])
        self.train(X_combined, y_combined)
        return self.eval(X_test, y_test)
