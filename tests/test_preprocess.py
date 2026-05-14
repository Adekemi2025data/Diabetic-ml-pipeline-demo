import pandas as pd
import numpy as np
import pytest
import sys
sys.path.insert(0, "src")

from preprocessing import validate_dataframe, clean_data, encode_categoricals, check_data_quality


@pytest.fixture
def sample_data():
    """Small dataset that mimics the diabetes dataset structure."""
    return pd.DataFrame({
        "Pregnancies": [2, 4, 1, 0, 3, 5],
        "Glucose": [120, 0, 150, 100, 0, 130],  # zeros = missing
        "BloodPressure": [70, 80, 0, 75, 65, 0],  # zeros = missing
        "SkinThickness": [20, 0, 35, 0, 15, 25],  # zeros = missing
        "Insulin": [80, 0, 130, 0, 100, 0],  # zeros = missing
        "BMI": [32.5, 0, 28.1, 30.0, 0, 27.5],  # zeros = missing
        "DiabetesPedigreeFunction": [0.5, 0.8, 0.3, 0.6, 0.9, 0.4],
        "Age": [25, 32, 45, 29, 41, 50],
        "Outcome": [0, 1, 0, 1, 0, 1]
    })


# -----------------------------
# validate_dataframe tests
# -----------------------------
class TestValidateDataframe:

    def test_valid_dataframe_passes(self, sample_data):
        result = validate_dataframe(
            sample_data,
            required_columns=[
                "Pregnancies", "Glucose", "BloodPressure", "Outcome"
            ],
            target_column="Outcome"
        )
        assert result is True

    def test_missing_column_raises(self, sample_data):
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_dataframe(
                sample_data,
                required_columns=["Glucose", "Nonexistent"],
                target_column="Outcome"
            )

    def test_missing_target_raises(self, sample_data):
        with pytest.raises(ValueError, match="Target column"):
            validate_dataframe(
                sample_data,
                required_columns=["Glucose"],
                target_column="NotAColumn"
            )

    def test_empty_dataframe_raises(self):
        empty_df = pd.DataFrame({"Glucose": [], "Outcome": []})
        with pytest.raises(ValueError, match="empty"):
            validate_dataframe(empty_df, ["Glucose"], "Outcome")


# -----------------------------
# clean_data tests
# -----------------------------
class TestCleanData:

    def test_replaces_zeros_and_fills_numeric(self, sample_data):
        numeric_cols = [
            "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI"
        ]
        result = clean_data(sample_data, numeric_cols)

        # All zeros should be replaced with median
        assert (result[numeric_cols] == 0).sum().sum() == 0
        assert result[numeric_cols].isna().sum().sum() == 0

    def test_does_not_modify_original(self, sample_data):
        original = sample_data.copy()
        clean_data(sample_data, ["Glucose"])
        assert sample_data.equals(original)

    def test_fills_with_median(self, sample_data):
        result = clean_data(sample_data, ["Glucose"])
        median_glucose = np.median([120, 150, 100, 130])  # zeros excluded
        assert result["Glucose"].iloc[1] == median_glucose
        assert result["Glucose"].iloc[4] == median_glucose


# -----------------------------
# encode_categoricals tests
# -----------------------------
class TestEncodeCategoricals:

    def test_no_categoricals_returns_same_df(self, sample_data):
        result = encode_categoricals(sample_data, [])
        assert result.equals(sample_data)

    def test_preserves_row_count(self, sample_data):
        result = encode_categoricals(sample_data, [])
        assert len(result) == len(sample_data)


# -----------------------------
# check_data_quality tests
# -----------------------------
class TestDataQuality:

    def test_counts_nulls(self, sample_data):
        # Before cleaning, zeros are not nulls
        report = check_data_quality(sample_data, ["Glucose", "BloodPressure"])
        assert report["total_nulls"] == 0

    def test_counts_rows(self, sample_data):
        report = check_data_quality(sample_data, ["Glucose"])
        assert report["total_rows"] == 6

    def test_reports_numeric_ranges(self, sample_data):
        report = check_data_quality(sample_data, ["Age"])
        assert report["Age_min"] == 25
        assert report["Age_max"] == 50
