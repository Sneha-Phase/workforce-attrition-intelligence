"""
tests/test_loader.py
=====================
Unit and integration tests for src/data/loader.py.

Tests are self-contained: they use the real source CSV for integration
tests and synthesise minimal DataFrames for unit tests of validation logic.
"""

from __future__ import annotations

import copy
import io

import pandas as pd
import pytest

from src.config import (
    RAW_CSV,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    TARGET_POSITIVE,
    TARGET_NEGATIVE,
    EXPECTED_SHAPE,
    ID_COLUMN,
)
from src.data.loader import (
    load_data,
    validate_schema,
    get_class_distribution,
    get_attrition_rate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_df(nrows: int = 10) -> pd.DataFrame:
    """
    Return a minimal valid DataFrame that satisfies the schema contract.
    Uses real column names and minimal plausible values.
    """
    assert nrows >= 2, "_make_valid_df requires at least 2 rows"
    # Split rows so both target classes are always represented.
    n_yes = max(1, nrows // 5)   # ~20% Yes mirrors real distribution
    n_no  = nrows - n_yes
    data = {
        "Employee_ID": range(1, nrows + 1),
        "Age": [35] * nrows,
        "Gender": ["Female"] * nrows,
        "Marital_Status": ["Single"] * nrows,
        "Department": ["IT"] * nrows,
        "Job_Role": ["Analyst"] * nrows,
        "Job_Level": [2] * nrows,
        "Monthly_Income": [8000] * nrows,
        "Hourly_Rate": [40] * nrows,
        "Years_at_Company": [5] * nrows,
        "Years_in_Current_Role": [3] * nrows,
        "Years_Since_Last_Promotion": [2] * nrows,
        "Work_Life_Balance": [3] * nrows,
        "Job_Satisfaction": [3] * nrows,
        "Performance_Rating": [3] * nrows,
        "Training_Hours_Last_Year": [20] * nrows,
        "Overtime": ["No"] * nrows,
        "Project_Count": [4] * nrows,
        "Average_Hours_Worked_Per_Week": [40] * nrows,
        "Absenteeism": [5] * nrows,
        "Work_Environment_Satisfaction": [3] * nrows,
        "Relationship_with_Manager": [3] * nrows,
        "Job_Involvement": [3] * nrows,
        "Distance_From_Home": [10] * nrows,
        "Number_of_Companies_Worked": [2] * nrows,
        "Attrition": (["Yes"] * n_yes) + (["No"] * n_no),
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Integration tests — require the actual source CSV
# ---------------------------------------------------------------------------

class TestLoadDataIntegration:

    def test_csv_exists(self):
        """Source CSV must be present at the configured path."""
        assert RAW_CSV.exists(), (
            f"Source CSV missing: {RAW_CSV}. "
            "Place the file and re-run."
        )

    def test_load_returns_dataframe(self):
        df = load_data()
        assert isinstance(df, pd.DataFrame)

    def test_load_correct_shape(self):
        df = load_data()
        assert df.shape == EXPECTED_SHAPE, (
            f"Expected shape {EXPECTED_SHAPE}, got {df.shape}"
        )

    def test_load_all_required_columns_present(self):
        df = load_data()
        for col in REQUIRED_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_load_no_missing_values(self):
        df = load_data()
        total_nulls = df.isnull().sum().sum()
        assert total_nulls == 0, f"Unexpected nulls: {total_nulls}"

    def test_load_target_values_only_yes_no(self):
        df = load_data()
        unique_vals = set(df[TARGET_COLUMN].unique())
        assert unique_vals == {TARGET_POSITIVE, TARGET_NEGATIVE}, (
            f"Unexpected target values: {unique_vals}"
        )

    def test_load_employee_id_all_unique(self):
        df = load_data()
        assert df[ID_COLUMN].nunique() == len(df), (
            "Employee_ID values are not all unique."
        )

    def test_load_does_not_modify_source(self):
        """
        Load the data twice and confirm both copies are identical.
        (Guards against any accidental in-place mutation inside load_data.)
        """
        df1 = load_data()
        df2 = load_data()
        pd.testing.assert_frame_equal(df1, df2)

    def test_attrition_rate_approx_20_pct(self):
        df = load_data()
        rate = get_attrition_rate(df)
        # Profiling showed 19.97%; allow ±2% tolerance
        assert 0.17 <= rate <= 0.23, (
            f"Attrition rate {rate:.4f} is outside expected 17–23% band."
        )

    def test_class_distribution_both_classes_present(self):
        df = load_data()
        dist = get_class_distribution(df)
        assert TARGET_POSITIVE in dist.index
        assert TARGET_NEGATIVE in dist.index

    def test_class_distribution_majority_is_no(self):
        df = load_data()
        dist = get_class_distribution(df)
        assert dist[TARGET_NEGATIVE] > dist[TARGET_POSITIVE], (
            "Expected more 'No' than 'Yes' attrition rows."
        )


# ---------------------------------------------------------------------------
# Unit tests — validate_schema with synthesised DataFrames
# ---------------------------------------------------------------------------

class TestValidateSchema:

    def test_valid_df_passes(self):
        """A properly shaped valid DataFrame should not raise."""
        df = _make_valid_df(EXPECTED_SHAPE[0])
        # Should not raise — no assertion needed beyond no exception
        validate_schema(df)

    def test_missing_column_raises(self):
        df = _make_valid_df(EXPECTED_SHAPE[0])
        df = df.drop(columns=["Age"])
        with pytest.raises(ValueError, match="missing columns"):
            validate_schema(df)

    def test_wrong_shape_raises(self):
        df = _make_valid_df(50)   # too few rows
        with pytest.raises(ValueError, match="Shape mismatch"):
            validate_schema(df)

    def test_missing_values_raises(self):
        df = _make_valid_df(EXPECTED_SHAPE[0])
        df.loc[0, "Age"] = None
        with pytest.raises(ValueError, match="Missing values"):
            validate_schema(df)

    def test_unexpected_target_value_raises(self):
        df = _make_valid_df(EXPECTED_SHAPE[0])
        df.loc[0, "Attrition"] = "Maybe"
        with pytest.raises(ValueError, match="Unexpected target values"):
            validate_schema(df)


# ---------------------------------------------------------------------------
# Unit tests — load_data path override
# ---------------------------------------------------------------------------

class TestLoadDataPath:

    def test_nonexistent_path_raises_file_not_found(self, tmp_path):
        fake_path = tmp_path / "nonexistent.csv"
        with pytest.raises(FileNotFoundError):
            load_data(path=fake_path)

    def test_load_from_custom_path(self, tmp_path):
        """Verify that the path override works for test isolation."""
        df_out = _make_valid_df(EXPECTED_SHAPE[0])
        csv_path = tmp_path / "test.csv"
        df_out.to_csv(csv_path, index=False)
        df_in = load_data(path=csv_path)
        assert df_in.shape == EXPECTED_SHAPE
        assert list(df_in.columns) == list(df_out.columns)


# ---------------------------------------------------------------------------
# Unit tests — utility functions
# ---------------------------------------------------------------------------

class TestUtilityFunctions:

    def test_get_attrition_rate_all_yes(self):
        df = _make_valid_df(10)
        df["Attrition"] = "Yes"
        assert get_attrition_rate(df) == 1.0

    def test_get_attrition_rate_all_no(self):
        df = _make_valid_df(10)
        df["Attrition"] = "No"
        assert get_attrition_rate(df) == 0.0

    def test_get_attrition_rate_half(self):
        df = _make_valid_df(10)
        df.loc[:4, "Attrition"] = "Yes"
        df.loc[5:, "Attrition"] = "No"
        rate = get_attrition_rate(df)
        assert abs(rate - 0.5) < 1e-9

    def test_get_class_distribution_returns_series(self):
        df = _make_valid_df(10)
        dist = get_class_distribution(df)
        assert isinstance(dist, pd.Series)

    def test_get_class_distribution_sums_to_nrows(self):
        df = _make_valid_df(10)
        dist = get_class_distribution(df)
        assert dist.sum() == len(df)
