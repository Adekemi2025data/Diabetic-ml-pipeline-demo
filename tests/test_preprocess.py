import pandas as pd
import numpy as np
import pytest
import sys

# Add src to path
sys.path.insert(0, "src")

from preprocessing import DiabetesPreprocessor


# ---------------------------------------------------------
# SAMPLE DIABETES DATA
# ---------------------------------------------------------
@pytest.fixture
def sample_data():
    """Small dataset that mimics the diabetes dataset structure."""
    return pd.DataFrame({
        "Pregnancies": [2, 4, 1, 0, 3, 5],
        "Glucose": [120, 0, 150, 100, 0, 130],  # zeros should become NaN
        "BloodPressure": [70, 80, 0, 75, 65, 0],  # zeros should become NaN
        "SkinThickness": [35, 0, 29, 0, 32, 28],
        "Insulin": [0, 130, 0, 100, 85, 0],
        "BMI": [33.6, 0, 28.1, 30.5, 0, 27.8],
        "DiabetesPedigreeFunction": [0.627, 0.351, 0.672, 0.245, 0.134, 0.543],
        "Age": [50, 31, 29, 45, 22, 41],
        "Outcome": [1, 0, 1, 0, 1, 0]
    })


# ---------------------------------------------------------
# TEST VALIDATION
# ---------------------------------------------------------
class TestValidateDataframe:

    def test_valid_dataframe_passes(self, sample_data):
        prep = DiabetesPreprocessor()
        assert prep.validate(sample_data) is True

    def test_missing_column_raises(self, sample_data):
        prep = DiabetesPreprocessor()
        bad_df = sample_data.drop(columns=["Glucose"])
        with pytest.raises(ValueError, match="Missing required columns"):
            prep.validate(bad_df)

    def test_missing_target_raises(self, sample_data):
        prep = DiabetesPreprocessor()
        bad_df = sample_data.drop(columns=["Outcome"])
        with pytest.raises(ValueError, match="Target column"):
            prep.validate(bad_df)

    def test_empty_dataframe_raises(self):
        prep = DiabetesPreprocessor()
        empty_df = pd.DataFrame({"Pregnancies": [], "Outcome": []})
        with pytest.raises(ValueError, match="Missing required columns"):
            prep.validate(empty_df)


# ---------------------------------------------------------
# TEST ZERO REPLACEMENT + MISSING FILLING
# ---------------------------------------------------------
class TestCleaning:

    def test_zero_replaced_with_nan(self, sample_data):
        prep = DiabetesPreprocessor()
        cleaned = prep.replace_zeros(sample_data)
        assert cleaned["Glucose"].isna().sum() == 2
        assert cleaned["BloodPressure"].isna().sum() == 2

    def test_fill_missing_replaces_nan(self, sample_data):
        prep = DiabetesPreprocessor()
        df = prep.replace_zeros(sample_data)
        cleaned = prep.fill_missing(df)
        assert cleaned.isna().sum().sum() == 0

    def test_original_not_modified(self, sample_data):
        prep = DiabetesPreprocessor()
        df_copy = sample_data.copy()
        prep.replace_zeros(sample_data)
        assert sample_data.equals(df_copy)

    def test_fill_uses_median(self, sample_data):
        prep = DiabetesPreprocessor()
        df = prep.replace_zeros(sample_data)
        cleaned = prep.fill_missing(df)
        median_glucose = df["Glucose"].median()
        assert cleaned["Glucose"].iloc[1] == median_glucose


# ---------------------------------------------------------
# TEST QUALITY REPORT
# ---------------------------------------------------------
class TestQualityReport:

    def test_counts_rows(self, sample_data):
        prep = DiabetesPreprocessor()
        report = prep.quality_report(sample_data)
        assert report["total_rows"] == 6

    def test_counts_nulls(self, sample_data):
        prep = DiabetesPreprocessor()
        report = prep.quality_report(sample_data)
        # zeros are not replaced here, so no NaN yet
        assert report["total_nulls"] == 0

    def test_reports_numeric_ranges(self, sample_data):
        prep = DiabetesPreprocessor()
        report = prep.quality_report(sample_data)
        assert report["Age_min"] == 22
        assert report["Age_max"] == 50
