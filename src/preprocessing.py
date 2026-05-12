import pandas as pd
import numpy as np

class DiabetesPreprocessor:
    """
    A full preprocessing pipeline for the Pima Indians Diabetes dataset.
    Handles validation, cleaning, zero-replacement, and optional encoding.
    """

    def __init__(self, target_column="Outcome"):
        self.target_column = target_column

        # Columns where zero means "missing"
        self.zero_as_missing = [
            "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
        ]

        # All numeric columns except target
        self.numeric_columns = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]

    # ---------------------------------------------------------
    # 1. VALIDATION
    # ---------------------------------------------------------
    def validate(self, df):
        """Ensure required columns and target exist."""
        missing = [col for col in self.numeric_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found")

        if df.empty:
            raise ValueError("Dataframe is empty")

        return True

    # ---------------------------------------------------------
    # 2. CLEANING
    # ---------------------------------------------------------
    def replace_zeros(self, df):
        """Replace biologically impossible zeros with NaN."""
        df = df.copy()
        for col in self.zero_as_missing:
            if col in df.columns:
                df[col] = df[col].replace(0, np.nan)
        return df

    def fill_missing(self, df):
        """Fill missing numeric values with median."""
        df = df.copy()
        for col in self.numeric_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        return df

    # ---------------------------------------------------------
    # 3. OPTIONAL ENCODING (not needed for diabetes dataset)
    # ---------------------------------------------------------
    def encode(self, df, categorical_columns=None):
        """One-hot encode categorical columns (placeholder)."""
        df = df.copy()
        if categorical_columns:
            df = pd.get_dummies(df, columns=categorical_columns, drop_first=True, dtype=int)
        return df

    # ---------------------------------------------------------
    # 4. QUALITY REPORT
    # ---------------------------------------------------------
    def quality_report(self, df):
        """Generate a data quality report."""
        report = {
            "total_rows": len(df),
            "total_nulls": int(df.isnull().sum().sum()),
            "null_percentage": round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
            "duplicate_rows": int(df.duplicated().sum()),
        }

        for col in self.numeric_columns:
            if col in df.columns:
                report[f"{col}_min"] = float(df[col].min())
                report[f"{col}_max"] = float(df[col].max())

        return report

    # ---------------------------------------------------------
    # 5. FULL PIPELINE
    # ---------------------------------------------------------
    def run(self, df):
        """Run the full preprocessing pipeline."""
        self.validate(df)
        df = self.replace_zeros(df)
        df = self.fill_missing(df)
        return df
