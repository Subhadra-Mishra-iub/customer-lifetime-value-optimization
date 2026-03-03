"""
Modeling module for churn probability prediction.

Trains Logistic Regression and Random Forest classifiers, evaluates performance,
and extracts feature importance for business interpretation.
"""

import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split features and target into train and test sets.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    test_size : float
        Fraction of data for test set.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Standardize features for Logistic Regression.

    Random Forest is scale-invariant, but scaling helps LR convergence.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Train and test feature matrices.

    Returns
    -------
    X_train_scaled, X_test_scaled : np.ndarray
    scaler : StandardScaler
        Fitted scaler for potential reuse.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: pd.Series,
    random_state: int = 42,
) -> LogisticRegression:
    """Train Logistic Regression with class weight balancing for imbalanced churn."""
    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        class_weight="balanced",
        solver="lbfgs",
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    n_estimators: int = 100,
) -> RandomForestClassifier:
    """Train Random Forest for churn prediction with built-in feature importance."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
        max_depth=10,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: pd.Series,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Compute ROC-AUC, confusion matrix, precision, and recall.

    Parameters
    ----------
    model : fitted classifier
    X_test : np.ndarray or pd.DataFrame
    y_test : pd.Series
    model_name : str
        Label for the results dict.

    Returns
    -------
    dict with keys: roc_auc, confusion_matrix, precision, recall, y_pred, y_proba, fpr, tpr
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    return {
        "model_name": model_name,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "fpr": fpr,
        "tpr": tpr,
    }


def get_feature_importance(
    model: Any,
    feature_names: list,
    model_type: str = "rf",
) -> pd.DataFrame:
    """
    Extract feature importance from the model.

    For Random Forest, uses impurity-based importance.
    For Logistic Regression, uses absolute coefficient magnitudes.

    Parameters
    ----------
    model : fitted classifier
    feature_names : list
        Ordered list of feature names matching model input.
    model_type : str
        'rf' for Random Forest, 'lr' for Logistic Regression.

    Returns
    -------
    pd.DataFrame with columns: feature, importance, sorted by importance desc.
    """
    if model_type == "rf":
        imp = model.feature_importances_
    else:
        imp = np.abs(model.coef_.ravel())

    df = pd.DataFrame({"feature": feature_names, "importance": imp})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Full modeling pipeline: split, scale, train both models, evaluate.

    Returns
    -------
    results : dict
        Keys: 'lr' and 'rf', each with evaluation metrics and models.
    X_test, y_test : for downstream simulation (need predictions on test set).
    """
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size, random_state=random_state)
    feature_names = list(X.columns)

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    lr_model = train_logistic_regression(X_train_scaled, y_train, random_state=random_state)
    rf_model = train_random_forest(X_train, y_train, random_state=random_state)

    lr_results = evaluate_model(lr_model, X_test_scaled, y_test, "Logistic Regression")
    rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest")

    lr_results["model"] = lr_model
    lr_results["scaler"] = scaler
    lr_results["feature_importance"] = get_feature_importance(lr_model, feature_names, "lr")

    rf_results["model"] = rf_model
    rf_results["feature_importance"] = get_feature_importance(rf_model, feature_names, "rf")

    results = {"lr": lr_results, "rf": rf_results}
    test_data = {"X_test": X_test, "y_test": y_test, "X_test_scaled": X_test_scaled}

    return results, test_data


def save_model(model: Any, path: Path) -> None:
    """Save a fitted model to disk for reproducibility."""
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: Path) -> Any:
    """Load a saved model from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)
