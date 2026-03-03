"""
Data preprocessing for CLV Prediction and Retention Budget Optimization.

Handles loading, cleaning, and feature engineering for the Telco churn dataset.
Designed for reproducibility with explicit handling of edge cases.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

# Column name mappings for dataset variations (IBM Telco can have different exports)
CHURN_COL_ALIASES = ["Churn", "Churn Label", "churn"]
TOTAL_CHARGES_ALIASES = ["TotalCharges", "Total Charges", "total_charges"]
CLTV_ALIASES = ["CLTV", "CLV", "Customer Lifetime Value", "clv"]

# Columns to exclude from modeling (identifiers, leakage, or post-churn info)
EXCLUDE_COLS = [
    "CustomerID", "customerID", "customer_id",
    "Count", "count",
    "Country", "State", "City", "Zip Code", "Zip code",
    "Lat Long", "Latitude", "Longitude",
    "Churn Score", "Churn Value", "ChurnReason", "Churn Reason", "churn_reason",
]


def _resolve_column(df: pd.DataFrame, aliases: list) -> Optional[str]:
    """Return the first matching column name from aliases, or None."""
    for alias in aliases:
        if alias in df.columns:
            return alias
    return None


def load_data(data_path: Path) -> pd.DataFrame:
    """
    Load the Telco churn dataset from Excel or CSV.

    Supports both .xlsx and .csv formats for flexibility across data sources.

    Parameters
    ----------
    data_path : Path
        Full path to the data file.

    Returns
    -------
    pd.DataFrame
        Raw dataset.

    Raises
    ------
    FileNotFoundError
        If the data file does not exist.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    suffix = data_path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(data_path)
    elif suffix == ".csv":
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    return df


def convert_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert TotalCharges to numeric, handling common data issues.

    IBM Telco data often has whitespace or empty strings for new customers
    (zero tenure). These are coerced to NaN and then filled with 0.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a TotalCharges-like column.

    Returns
    -------
    pd.DataFrame
        DataFrame with TotalCharges as float64.
    """
    col = _resolve_column(df, TOTAL_CHARGES_ALIASES)
    if col is None:
        return df

    # Replace whitespace-only strings with NaN before conversion
    series = df[col].replace(r"^\s*$", np.nan, regex=True)
    df = df.copy()
    df[col] = pd.to_numeric(series, errors="coerce").fillna(0).astype(np.float64)
    return df


def map_churn_to_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map churn indicator to binary (Yes/1 = churned, No/0 = retained).

    Handles both string (Yes/No) and numeric (0/1) encodings.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a Churn-like column.

    Returns
    -------
    pd.DataFrame
        DataFrame with binary churn column.
    """
    col = _resolve_column(df, CHURN_COL_ALIASES)
    if col is None:
        raise ValueError("No churn column found. Expected one of: " + ", ".join(CHURN_COL_ALIASES))

    series = df[col]
    if series.dtype in (np.int64, np.float64):
        df = df.copy()
        df["churn"] = (series > 0).astype(int)
    else:
        mapping = {"Yes": 1, "yes": 1, "No": 0, "no": 0}
        df = df.copy()
        df["churn"] = series.map(mapping)
        if df["churn"].isna().any():
            raise ValueError(f"Unexpected churn values: {series.unique().tolist()}")

    return df


def get_categorical_columns(df: pd.DataFrame, exclude: Optional[list] = None) -> list:
    """Return column names that are object/string type, excluding specified columns."""
    exclude = exclude or []
    return [
        c for c in df.select_dtypes(include=["object", "string"]).columns
        if c not in exclude
    ]


def prepare_features(
    df: pd.DataFrame,
    target_col: str = "churn",
    exclude_cols: Optional[list] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix X and target y for modeling.

    Drops identifiers and leakage columns, one-hot encodes categoricals,
    and separates target. CLTV is excluded from features (used for simulation only).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with churn and CLTV.
    target_col : str
        Name of the target column.
    exclude_cols : list, optional
        Additional columns to drop.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    """
    exclude_cols = exclude_cols or []
    drop_cols = [c for c in EXCLUDE_COLS + exclude_cols if c in df.columns]
    drop_cols.append(target_col)
    # Exclude CLTV from features to avoid leakage; it's used for simulation
    for a in CLTV_ALIASES:
        if a in df.columns:
            drop_cols.append(a)

    # Also drop Churn Label/Churn if we created 'churn'
    for a in CHURN_COL_ALIASES:
        if a in df.columns and a != target_col:
            drop_cols.append(a)

    work = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Identify categorical vs numeric
    cat_cols = get_categorical_columns(work, exclude=[target_col])
    num_cols = [c for c in work.columns if c not in cat_cols and c != target_col]

    # One-hot encode categoricals
    if cat_cols:
        dummies = pd.get_dummies(work[cat_cols], drop_first=True, dtype=int)
        X = pd.concat([work[num_cols].reset_index(drop=True), dummies], axis=1)
    else:
        X = work[num_cols].copy()

    y = df[target_col].astype(int)
    return X, y


def preprocess_pipeline(
    data_path: Path,
    handle_missing: str = "drop",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Full preprocessing pipeline: load, clean, and prepare data.

    Parameters
    ----------
    data_path : Path
        Path to the raw data file.
    handle_missing : str
        'drop' to remove rows with missing values, 'fill' to use median/mode.

    Returns
    -------
    df_clean : pd.DataFrame
        Cleaned DataFrame with churn, CLTV, and all features (for simulation).
    X : pd.DataFrame
        Feature matrix for modeling.
    y : pd.Series
        Target vector.
    """
    df = load_data(data_path)
    df = convert_total_charges(df)
    df = map_churn_to_binary(df)

    # Resolve CLTV column for later use
    cltv_col = _resolve_column(df, CLTV_ALIASES)
    if cltv_col and cltv_col != "CLTV":
        df["CLTV"] = df[cltv_col]

    # Standardize TotalCharges column name for downstream use
    tc_col = _resolve_column(df, TOTAL_CHARGES_ALIASES)
    if tc_col and tc_col != "TotalCharges":
        df["TotalCharges"] = df[tc_col]

    # Drop Churn Reason before handling missing—it's only present for churned customers,
    # so keeping it would force dropna to remove all retained customers
    for col in ["Churn Reason", "ChurnReason", "churn_reason"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    if handle_missing == "drop":
        df = df.dropna()
    else:
        # Fill numeric with median, categorical with mode
        for col in df.columns:
            if df[col].dtype in (np.int64, np.float64):
                df[col] = df[col].fillna(df[col].median())
            elif df[col].dtype == object:
                df[col] = df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "")

    X, y = prepare_features(df, target_col="churn")
    return df, X, y
