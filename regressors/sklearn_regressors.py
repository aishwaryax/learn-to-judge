import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    precision_recall_fscore_support, confusion_matrix, make_scorer,
    accuracy_score, precision_recall_fscore_support
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import pearsonr, spearmanr, kendalltau
import ast

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class DeltaLS:
    def __init__(self, train_path: str, test_path: str,
                 train_emb_path: str, test_emb_path: str,
                 size: int = -1, use_external_bias: bool = True,
                 seed: int = 42, standardize: bool = True):
        self.seed = seed
        set_seed(seed)
        self.train_path = train_path
        self.test_path = test_path
        self.train_emb_path = train_emb_path
        self.test_emb_path = test_emb_path
        self.size = size
        self.use_external_bias = use_external_bias
        self.standardize = standardize
        self.model = None
        self.best_lambda_theta = None
        self.reg_data = {}

    def _clean_data(self, df):
        df.dropna(subset=["human_score", "llm_score"], inplace=True)
        df["human_score"] = df["human_score"].astype(int)
        df["llm_score"] = df["llm_score"].astype(int)
        df["target_probability"] = df["target_probability"].apply(ast.literal_eval)

    def preprocess(self):
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        self._clean_data(train_df)
        self._clean_data(test_df)

        train_embeddings = np.load(self.train_emb_path)
        test_embeddings = np.load(self.test_emb_path)

        if self.size != -1:
            selected_indices = np.random.choice(train_df.index, size=min(self.size, len(train_df)), replace=False)
            train_df = train_df.loc[selected_indices].reset_index(drop=True)

        def prepare_features(df, embeddings):
            X = embeddings[df["embedding_index_critique"].values]
            if self.use_external_bias:
                bias = df[["llm_score"]].values
                X = np.hstack([X, bias])
            y = df["human_score"].values
            return X, y

        X_train, y_train = prepare_features(train_df, train_embeddings)
        X_test, y_test = prepare_features(test_df, test_embeddings)

        if self.standardize:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        return X_train, y_train, X_test, y_test

    def tune_hyperparameters(self, X_train, y_train, k_folds=5, lambda_values=None):
        ridge = Ridge(random_state=self.seed)
        param_grid = {"alpha": lambda_values}

        grid_search = GridSearchCV(
            ridge,
            param_grid,
            scoring="neg_mean_squared_error",
            cv=KFold(n_splits=k_folds, shuffle=True, random_state=self.seed),
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        self.best_lambda_theta = grid_search.best_params_["alpha"]
        best_score = -grid_search.best_score_
        for mean_score, params in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']):
            self.reg_data[params['alpha']] = -mean_score
        print(f"Best regularization: λ_theta={self.best_lambda_theta} with Avg MSE {best_score:.4f}")

    def train(self, X_train, y_train, lambda_theta=None):
        model = Ridge(alpha=lambda_theta, random_state=self.seed)
        model.fit(X_train, y_train)
        return model

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        if hasattr(y, "cpu"):
            y = y.cpu().numpy()

        if np.issubdtype(y.dtype, np.floating):
            y_rounded = np.round(y).astype(int)
        else:
            y_rounded = y.astype(int)

        y_pred_rounded = np.round(y_pred).astype(int)

        return {
            "MSE": mean_squared_error(y, y_pred),
            "MAE": mean_absolute_error(y, y_pred),
            "R2 Score": r2_score(y, y_pred),
            "Min Prediction": np.min(y_pred),
            "Max Prediction": np.max(y_pred),
            "Accuracy": accuracy_score(y_rounded, y_pred_rounded),
            "Precision": precision_recall_fscore_support(y_rounded, y_pred_rounded, average="weighted", zero_division=1)[0],
            "Recall": precision_recall_fscore_support(y_rounded, y_pred_rounded, average="weighted", zero_division=1)[1],
            "F1 Score": precision_recall_fscore_support(y_rounded, y_pred_rounded, average="weighted", zero_division=1)[2],
            "Confusion Matrix": confusion_matrix(y_rounded, y_pred_rounded, labels=np.arange(y_rounded.max() + 1)),
            "Class Labels": list(range(y_rounded.max() + 1)),
            "Pearson r": pearsonr(y, y_pred)[0],
            "Spearman ρ": spearmanr(y, y_pred)[0],
            "Kendall τ": kendalltau(y, y_pred)[0],
        }


    def experiment(self) -> dict:
        X_train, y_train, X_test, y_test = self.preprocess()

        lambda_values = [10.0 ** x for x in range(-5, 6)]
        self.tune_hyperparameters(X_train, y_train, k_folds=5, lambda_values=lambda_values)
        self.model = self.train(X_train, y_train, lambda_theta=self.best_lambda_theta)

        return self.evaluate(X_test, y_test)


class DeltaMultinomial:
    def __init__(self, train_path, test_path, train_emb_path, test_emb_path,
                 size=-1, epochs=10000, lambda_theta=0.0,
                 seed=42, standardize=True, use_external_bias=True):
        set_seed(seed)
        self.train_path = train_path
        self.test_path = test_path
        self.train_emb_path = train_emb_path
        self.test_emb_path = test_emb_path
        self.size = size
        self.epochs = epochs
        self.lambda_theta = lambda_theta
        self.seed = seed
        self.standardize = standardize
        self.use_external_bias = use_external_bias
        self.model = None
        self.scaler = None
        self.label_encoder = LabelEncoder()
        self.reg_data = {}

    def _load_data(self):
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        train_embeddings = np.load(self.train_emb_path)
        test_embeddings = np.load(self.test_emb_path)
        return train_df, test_df, train_embeddings, test_embeddings

    def _clean_data(self, df):
        df.dropna(subset=["human_score", "llm_score"], inplace=True)
        df["human_score"] = df["human_score"].astype(int)
        df["llm_score"] = df["llm_score"].astype(int)
        df["target_probability"] = df["target_probability"].apply(ast.literal_eval)

    def _encode_labels(self, train_df, test_df):
        all_labels = np.concatenate([train_df["human_score"], test_df["human_score"]])
        self.label_encoder.fit(all_labels)

    def compute_log_p(self, df):
        label_classes = self.label_encoder.classes_
        reverse_label_dict = {label: idx for idx, label in enumerate(label_classes)}
        log_p_list = []
        for _, row in df.iterrows():
            tp = row["target_probability"]
            p_vec = np.zeros(len(label_classes))
            for key in tp:
                if int(key) in reverse_label_dict:
                    p_vec[reverse_label_dict[int(key)]] = tp[key]
            log_p_list.append(np.log(np.clip(p_vec, 1e-8, 1.0 - 1e-8)))
        return np.stack(log_p_list)

    def preprocess(self):
        train_df, test_df, train_embeddings, test_embeddings = self._load_data()
        self._clean_data(train_df)
        self._clean_data(test_df)
        self._encode_labels(train_df, test_df)

        if self.size != -1:
            selected_indices = np.random.choice(train_df.index, size=min(self.size, len(train_df)), replace=False)
            train_df = train_df.loc[selected_indices].reset_index(drop=True)

        X = train_embeddings[train_df["embedding_index_critique"].values]
        y = self.label_encoder.transform(train_df["human_score"])
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        X_test = test_embeddings[test_df["embedding_index_critique"].values]
        y_test = self.label_encoder.transform(test_df["human_score"])

        log_p_all_train = self.compute_log_p(train_df)
        log_p_train, log_p_val = train_test_split(log_p_all_train, test_size=0.2, random_state=42)
        log_p_test = self.compute_log_p(test_df)

        if self.use_external_bias:
            X_train = np.hstack([X_train, log_p_train])
            X_val = np.hstack([X_val, log_p_val])
            X_test = np.hstack([X_test, log_p_test])

        if self.standardize:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
            X_test = self.scaler.transform(X_test)

        return X_train, X_val, y_train, y_val, X_test, y_test

    def train_model(self, X_train, y_train, lambda_theta=None):
        model = LogisticRegression(
            penalty='l2',
            C=1.0 / (lambda_theta if lambda_theta is not None else 1e-8),
            solver='lbfgs',
            multi_class='multinomial',
            max_iter=self.epochs,
            random_state=self.seed
        )
        model.fit(X_train, y_train)
        return model

    def tune_hyperparameters(self, X_train, y_train, k_folds=5, lambda_values=None):
        param_grid = {'C': [1.0 / l if l != 0 else 1e8 for l in lambda_values]}
        base_model = LogisticRegression(
            penalty='l2',
            solver='lbfgs',
            multi_class='multinomial',
            max_iter=self.epochs,
            random_state=self.seed
        )
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            scoring='accuracy',
            cv=KFold(n_splits=k_folds, shuffle=True, random_state=self.seed),
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_C = grid_search.best_params_["C"]
        self.lambda_theta = 1.0 / best_C
        for mean_score, params in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']):
            lambda_val = 1.0 / params['C'] if params['C'] != 0 else 0.0
            self.reg_data[lambda_val] = mean_score
        print(f"Best λ_theta={self.lambda_theta:.6f} with CV Accuracy={grid_search.best_score_:.4f}")

    def predict(self, X):
        probs = self.model.predict_proba(X)
        preds = np.argmax(probs, axis=1)
        return preds, probs

    def evaluate(self, X, y):
        y_pred, probs = self.predict(X)
        if probs.shape[1] == 2:
            expected_score = probs[:, 1]
        else:
            expected_score = np.dot(probs, self.label_encoder.inverse_transform(np.arange(probs.shape[1])).astype(float))

        metrics = {
            "MSE": mean_squared_error(y, expected_score),
            "MAE": mean_absolute_error(y, expected_score),
            "R2 Score": r2_score(y, expected_score),
            "Min Prediction": np.min(y_pred),
            "Max Prediction": np.max(y_pred),
            "Accuracy": accuracy_score(y, y_pred),
            "Precision": precision_recall_fscore_support(y, y_pred, average="weighted", zero_division=1)[0],
            "Recall": precision_recall_fscore_support(y, y_pred, average="weighted", zero_division=1)[1],
            "F1 Score": precision_recall_fscore_support(y, y_pred, average="weighted", zero_division=1)[2],
            "Confusion Matrix": confusion_matrix(y, y_pred, labels=np.arange(len(self.label_encoder.classes_))),
            "Class Labels": self.label_encoder.classes_.tolist(),
            "Pearson r": pearsonr(y, expected_score)[0],
            "Spearman ρ": spearmanr(y, expected_score)[0],
            "Kendall τ": kendalltau(y, expected_score)[0],
        }
        return metrics

    def experiment(self):
        X_train, X_val, y_train, y_val, X_test, y_test = self.preprocess()
        lambda_values = [10.0 ** x for x in range(-5, 6)]
        self.tune_hyperparameters(X_train, y_train, k_folds=5, lambda_values=lambda_values)
        X_full = np.vstack([X_train, X_val])
        y_full = np.hstack([y_train, y_val])
        self.model = self.train_model(X_full, y_full, lambda_theta=self.lambda_theta)
        return self.evaluate(X_test, y_test)

class DeltaBTL2:
    def __init__(self, train_path: str, test_path: str,
                 emb1_path_train: str, emb2_path_train: str,
                 emb1_path_test: str, emb2_path_test: str,
                 size: int = -1, epochs: int = 10000,
                 lambda_theta: float = 0.0,
                 use_external_bias: bool = True, 
                 seed: int = 42, standardize: bool = True):
        set_seed(seed)
        self.seed = seed
        self.train_path = train_path
        self.test_path = test_path
        self.emb1_path_train = emb1_path_train
        self.emb2_path_train = emb2_path_train
        self.emb1_path_test = emb1_path_test
        self.emb2_path_test = emb2_path_test
        self.size = size
        self.epochs = epochs
        self.lambda_theta = lambda_theta
        self.use_external_bias = use_external_bias
        self.standardize = standardize
        self.model = None
        self.reg_data = {}

    def _load_data(self):
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        emb_train1 = np.load(self.emb1_path_train)
        emb_train2 = np.load(self.emb2_path_train)
        emb_test1 = np.load(self.emb1_path_test)
        emb_test2 = np.load(self.emb2_path_test)
        return train_df, test_df, emb_train1, emb_train2, emb_test1, emb_test2

    def _filter_and_sample(self, df):
        df.dropna(subset=["human_score", "llm_score1", "llm_score2",
                           "embedding_index_critique1", "embedding_index_critique2"], inplace=True)
        df["human_score"] = df["human_score"].astype(int)
        if self.size != -1:
            df = df.sample(n=min(self.size, len(df)), random_state=42).reset_index(drop=True)
        return df

    def _compute_log_odds(self, df):
        p = df["llm_score1"].replace(0, 1e-8).astype(np.float32) / (
            df["llm_score1"].replace(0, 1e-8).astype(np.float32) +
            df["llm_score2"].replace(0, 1e-8).astype(np.float32))
        return np.log(p / np.clip(1 - p, 1e-8, None))

    def preprocess(self):
        train_df, test_df, emb_train1, emb_train2, emb_test1, emb_test2 = self._load_data()
        train_df = self._filter_and_sample(train_df)
        test_df = self._filter_and_sample(test_df)

        X_train = emb_train1[train_df["embedding_index_critique1"]] - emb_train2[train_df["embedding_index_critique2"]]
        y_train = train_df["human_score"].values
        log_odds_train = self._compute_log_odds(train_df).values

        X_test = emb_test1[test_df["embedding_index_critique1"]] - emb_test2[test_df["embedding_index_critique2"]]
        y_test = test_df["human_score"].values
        log_odds_test = self._compute_log_odds(test_df).values

        X_train, X_val, y_train, y_val, log_odds_train, log_odds_val = train_test_split(
            X_train, y_train, log_odds_train, test_size=0.2, random_state=42)

        if self.use_external_bias:
            X_train = np.hstack([X_train, log_odds_train[:, None]])
            X_val = np.hstack([X_val, log_odds_val[:, None]])
            X_test = np.hstack([X_test, log_odds_test[:, None]])

        if self.standardize:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
            X_test = self.scaler.transform(X_test)

        return X_train, y_train, X_val, y_val, X_test, y_test

    def train(self, X_train, y_train, lambda_theta=None):
        model = LogisticRegression(
            penalty='l2',
            C=1.0 / (lambda_theta if lambda_theta is not None else 1e-8),
            solver='lbfgs',
            max_iter=self.epochs,
            random_state=self.seed
        )
        model.fit(X_train, y_train)
        return model

    def tune_hyperparameters(self, X_train, y_train, X_val=None, y_val=None, lambda_values=None):
        param_grid = {'C': [1.0 / l if l != 0 else 1e8 for l in lambda_values]}
        base_model = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=self.epochs, random_state=self.seed)

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            scoring="accuracy",
            cv=KFold(n_splits=5, shuffle=True, random_state=self.seed),
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        best_C = grid_search.best_params_["C"]
        self.lambda_theta = 1.0 / best_C
        for mean_score, params in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']):
            lambda_val = 1.0 / params['C'] if params['C'] != 0 else 0.0
            self.reg_data[lambda_val] = mean_score
        print(f"Best λ_theta={self.lambda_theta} found with avg CV accuracy={grid_search.best_score_:.4f}")

    def predict(self, X):
        probs = self.model.predict_proba(X)[:, 1]
        preds = (probs > 0.5).astype(int)
        return preds, probs

    def evaluate(self, X_test, y_test):
        y_pred, probs = self.predict(X_test)

        metrics = {
            "MSE": mean_squared_error(y_test, probs),
            "MAE": mean_absolute_error(y_test, probs),
            "R2 Score": r2_score(y_test, probs),
            "Min Prediction": np.min(y_pred),
            "Max Prediction": np.max(y_pred),
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=1)[0],
            "Recall": precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=1)[1],
            "F1 Score": precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=1)[2],
            "Confusion Matrix": confusion_matrix(y_test, y_pred),
            "Class Labels": np.unique(y_test).astype(int).tolist(),
            "Pearson r": pearsonr(y_test, probs)[0],
            "Spearman ρ": spearmanr(y_test, probs)[0],
            "Kendall τ": kendalltau(y_test, probs)[0]
        }
        return metrics

    def experiment(self):
        X_train, y_train, X_val, y_val, X_test, y_test = self.preprocess()
        lambda_values = [10.0 ** x for x in range(-5, 6)]
        self.tune_hyperparameters(X_train, y_train, X_val, y_val, lambda_values)
        self.model = self.train(X_train, y_train, lambda_theta=self.lambda_theta)
        return self.evaluate(X_test, y_test)

