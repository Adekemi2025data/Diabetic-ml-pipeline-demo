import pandas as pd
import numpy as np

def validate_dataframe(df, required_columns, target_column):
    """Validate that the diabetes dataset contains required columns."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    if df.empty:
        raise ValueError("Dataframe is empty")

    return True


def clean_data(df, numeric_columns):
    """
    Clean diabetes dataset:
    - Replace biologically impossible zeros with NaN
    - Fill missing numeric values with median
    """
    df = df.copy()

    # Columns where zero means "missing"
    zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    for col in zero_as_missing:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    # Fill numeric missing values with median
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df


def encode_categoricals(df, columns):
    """
    Diabetes dataset has no categorical columns by default,
    but function is kept for compatibility.
    """
    df = df.copy()
    if columns:
        df = pd.get_dummies(df, columns=columns, drop_first=True, dtype=int)
    return df


def check_data_quality(df, numeric_columns):
    """Return quality metrics for the diabetes dataset."""
    report = {
        "total_rows": len(df),
        "total_nulls": int(df.isnull().sum().sum()),
        "null_percentage": round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    for col in numeric_columns:
        if col in df.columns:
            report[f"{col}_min"] = float(df[col].min())
            report[f"{col}_max"] = float(df[col].max())

    return report
