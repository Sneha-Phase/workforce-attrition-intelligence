"""
tests/test_features.py
=======================
Unit and integration tests for src/features/build_features.py.

Covers:
- Each individual feature constructor
- The master build_features() function
- Edge cases (boundary values, anomaly rows, all-min / all-max inputs)
- Immutability guarantee (input DataFrame is not mutated)
- describe_engineered_features() error handling
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    add_role_tenure_anomaly,
    add_satisfaction_composite,
    add_tenure_promotion_ratio,
    add_hours_overload,
    add_income_to_role_ratio,
    build_features,
    describe_engineered_features,
    ENGINEERED_COLUMNS,
    COL_ROLE_TENURE_ANOMALY,
    COL_SATISFACTION_COMPOSITE,
    COL_TENURE_PROMOTION_RATIO,
    COL_HOURS_OVERLOAD,
    COL_INCOME_TO_ROLE_RATIO,
)
from src.config import EXPECTED_ANOMALY_COUNT


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _base_row(**overrides) -> dict:
    """Return a dict representing one 'typical' employee row."""
    row = {
        "Employee_ID": 1,
        "Age": 35,
        "Gender": "Female",
        "Marital_Status": "Single",
        "Department": "IT",
        "Job_Role": "Analyst",
        "Job_Level": 2,
        "Monthly_Income": 8000,
        "Hourly_Rate": 40,
        "Years_at_Company": 5,
        "Years_in_Current_Role": 3,
        "Years_Since_Last_Promotion": 2,
        "Work_Life_Balance": 3,
        "Job_Satisfaction": 3,
        "Performance_Rating": 3,
        "Training_Hours_Last_Year": 20,
        "Overtime": "No",
        "Project_Count": 4,
        "Average_Hours_Worked_Per_Week": 40,
        "Absenteeism": 5,
        "Work_Environment_Satisfaction": 3,
        "Relationship_with_Manager": 3,
        "Job_Involvement": 3,
        "Distance_From_Home": 10,
        "Number_of_Companies_Worked": 2,
        "Attrition": "No",
    }
    row.update(overrides)
    return row


def _df(*rows: dict) -> pd.DataFrame:
    """Build a DataFrame from one or more row dicts."""
    return pd.DataFrame(list(rows))


@pytest.fixture
def typical_row() -> pd.DataFrame:
    return _df(_base_row())


@pytest.fixture
def anomaly_row() -> pd.DataFrame:
    """A row where Years_in_Current_Role > Years_at_Company."""
    return _df(_base_row(Years_at_Company=3, Years_in_Current_Role=9))


@pytest.fixture
def consistent_row() -> pd.DataFrame:
    """A row where Years_in_Current_Role <= Years_at_Company."""
    return _df(_base_row(Years_at_Company=10, Years_in_Current_Role=3))


# ---------------------------------------------------------------------------
# Immutability: input DataFrame must never be mutated
# ---------------------------------------------------------------------------

class TestImmutability:

    def test_build_features_does_not_mutate_input(self):
        df = _df(_base_row())
        original_cols = list(df.columns)
        original_shape = df.shape
        _ = build_features(df)
        assert list(df.columns) == original_cols
        assert df.shape == original_shape

    def test_add_role_tenure_anomaly_does_not_mutate(self):
        df = _df(_base_row())
        original_shape = df.shape
        _ = add_role_tenure_anomaly(df)
        assert df.shape == original_shape

    def test_add_satisfaction_composite_does_not_mutate(self):
        df = _df(_base_row())
        original_shape = df.shape
        _ = add_satisfaction_composite(df)
        assert df.shape == original_shape


# ---------------------------------------------------------------------------
# Role_Tenure_Anomaly
# ---------------------------------------------------------------------------

class TestRoleTenureAnomaly:

    def test_anomaly_flag_is_1_when_role_exceeds_company(self, anomaly_row):
        result = add_role_tenure_anomaly(anomaly_row)
        assert result[COL_ROLE_TENURE_ANOMALY].iloc[0] == 1

    def test_anomaly_flag_is_0_when_consistent(self, consistent_row):
        result = add_role_tenure_anomaly(consistent_row)
        assert result[COL_ROLE_TENURE_ANOMALY].iloc[0] == 0

    def test_anomaly_flag_is_0_when_equal(self):
        df = _df(_base_row(Years_at_Company=5, Years_in_Current_Role=5))
        result = add_role_tenure_anomaly(df)
        assert result[COL_ROLE_TENURE_ANOMALY].iloc[0] == 0

    def test_anomaly_flag_dtype_is_int(self, typical_row):
        result = add_role_tenure_anomaly(typical_row)
        assert result[COL_ROLE_TENURE_ANOMALY].dtype in (int, "int64", "int32")

    def test_anomaly_flag_only_binary_values(self):
        rows = [
            _base_row(Years_at_Company=5, Years_in_Current_Role=3),
            _base_row(Years_at_Company=2, Years_in_Current_Role=9),
            _base_row(Years_at_Company=5, Years_in_Current_Role=5),
        ]
        df = _df(*rows)
        result = add_role_tenure_anomaly(df)
        unique_vals = set(result[COL_ROLE_TENURE_ANOMALY].unique())
        assert unique_vals.issubset({0, 1})

    def test_original_columns_unchanged_after_flag(self, anomaly_row):
        result = add_role_tenure_anomaly(anomaly_row)
        # Original values must NOT be modified
        assert result["Years_in_Current_Role"].iloc[0] == 9
        assert result["Years_at_Company"].iloc[0] == 3


# ---------------------------------------------------------------------------
# Satisfaction_Composite
# ---------------------------------------------------------------------------

class TestSatisfactionComposite:

    def test_composite_is_mean_of_five_columns(self, typical_row):
        result = add_satisfaction_composite(typical_row)
        expected = (3 + 3 + 3 + 3 + 3) / 5  # all inputs are 3
        assert abs(result[COL_SATISFACTION_COMPOSITE].iloc[0] - expected) < 1e-9

    def test_composite_all_min_values(self):
        df = _df(_base_row(
            Job_Satisfaction=1,
            Work_Life_Balance=1,
            Work_Environment_Satisfaction=1,
            Relationship_with_Manager=1,
            Job_Involvement=1,
        ))
        result = add_satisfaction_composite(df)
        assert result[COL_SATISFACTION_COMPOSITE].iloc[0] == 1.0

    def test_composite_all_max_values(self):
        df = _df(_base_row(
            Job_Satisfaction=5,
            Work_Life_Balance=4,
            Work_Environment_Satisfaction=4,
            Relationship_with_Manager=4,
            Job_Involvement=4,
        ))
        result = add_satisfaction_composite(df)
        expected = (5 + 4 + 4 + 4 + 4) / 5
        assert abs(result[COL_SATISFACTION_COMPOSITE].iloc[0] - expected) < 1e-9

    def test_composite_no_nans(self, typical_row):
        result = add_satisfaction_composite(typical_row)
        assert not result[COL_SATISFACTION_COMPOSITE].isnull().any()

    def test_composite_within_expected_range_on_full_dataset(self):
        """Integration: composite must be in [1.0, 5.0] on real data."""
        try:
            from src.data.loader import load_data
            df = load_data()
            result = add_satisfaction_composite(df)
            col = result[COL_SATISFACTION_COMPOSITE]
            assert col.min() >= 1.0, f"Composite min {col.min()} < 1.0"
            assert col.max() <= 5.0, f"Composite max {col.max()} > 5.0"
        except FileNotFoundError:
            pytest.skip("Source CSV not available — skipping integration test")


# ---------------------------------------------------------------------------
# Tenure_Promotion_Ratio
# ---------------------------------------------------------------------------

class TestTenurePromotionRatio:

    def test_ratio_formula_correct(self, typical_row):
        # typical_row: Years_at_Company=5, Years_Since_Last_Promotion=2
        result = add_tenure_promotion_ratio(typical_row)
        expected = 5 / (2 + 1)
        assert abs(result[COL_TENURE_PROMOTION_RATIO].iloc[0] - expected) < 1e-9

    def test_no_division_by_zero_when_promotion_is_zero(self):
        """
        Years_Since_Last_Promotion = 0 should NOT produce inf or NaN.
        The +1 offset must be applied.
        """
        df = _df(_base_row(Years_at_Company=5, Years_Since_Last_Promotion=0))
        result = add_tenure_promotion_ratio(df)
        val = result[COL_TENURE_PROMOTION_RATIO].iloc[0]
        assert np.isfinite(val), f"Expected finite value, got {val}"
        assert val == 5.0  # 5 / (0+1)

    def test_ratio_no_inf_or_nan_on_full_dataset(self):
        """Integration: no infinite or NaN values on real data."""
        try:
            from src.data.loader import load_data
            df = load_data()
            result = add_tenure_promotion_ratio(df)
            col = result[COL_TENURE_PROMOTION_RATIO]
            assert not col.isnull().any(), "NaN found in Tenure_Promotion_Ratio"
            assert not np.isinf(col).any(), "Inf found in Tenure_Promotion_Ratio"
        except FileNotFoundError:
            pytest.skip("Source CSV not available — skipping integration test")

    def test_ratio_non_negative(self):
        df = _df(_base_row(Years_at_Company=1, Years_Since_Last_Promotion=9))
        result = add_tenure_promotion_ratio(df)
        assert result[COL_TENURE_PROMOTION_RATIO].iloc[0] >= 0


# ---------------------------------------------------------------------------
# Hours_Overload
# ---------------------------------------------------------------------------

class TestHoursOverload:

    def test_overload_is_zero_at_40_hours(self, typical_row):
        result = add_hours_overload(typical_row)
        assert result[COL_HOURS_OVERLOAD].iloc[0] == 0

    def test_overload_positive_above_40(self):
        df = _df(_base_row(Average_Hours_Worked_Per_Week=50))
        result = add_hours_overload(df)
        assert result[COL_HOURS_OVERLOAD].iloc[0] == 10

    def test_overload_negative_below_40(self):
        df = _df(_base_row(Average_Hours_Worked_Per_Week=30))
        result = add_hours_overload(df)
        assert result[COL_HOURS_OVERLOAD].iloc[0] == -10

    def test_overload_range_on_full_dataset(self):
        """Integration: range matches profile (30–59 input → −10 to +19)."""
        try:
            from src.data.loader import load_data
            df = load_data()
            result = add_hours_overload(df)
            col = result[COL_HOURS_OVERLOAD]
            assert col.min() >= -10, f"Min {col.min()} below expected -10"
            assert col.max() <= 19,  f"Max {col.max()} above expected +19"
        except FileNotFoundError:
            pytest.skip("Source CSV not available — skipping integration test")


# ---------------------------------------------------------------------------
# Income_to_Role_Ratio
# ---------------------------------------------------------------------------

class TestIncomeToRoleRatio:

    def test_ratio_formula_correct(self, typical_row):
        # typical_row: Monthly_Income=8000, Job_Level=2
        result = add_income_to_role_ratio(typical_row)
        expected = 8000 / 2
        assert abs(result[COL_INCOME_TO_ROLE_RATIO].iloc[0] - expected) < 1e-9

    def test_no_division_by_zero_for_job_level_1(self):
        df = _df(_base_row(Job_Level=1, Monthly_Income=5000))
        result = add_income_to_role_ratio(df)
        val = result[COL_INCOME_TO_ROLE_RATIO].iloc[0]
        assert np.isfinite(val) and val == 5000.0

    def test_ratio_positive(self, typical_row):
        result = add_income_to_role_ratio(typical_row)
        assert result[COL_INCOME_TO_ROLE_RATIO].iloc[0] > 0

    def test_ratio_no_inf_or_nan_on_full_dataset(self):
        """Integration: no infinite or NaN on real data."""
        try:
            from src.data.loader import load_data
            df = load_data()
            result = add_income_to_role_ratio(df)
            col = result[COL_INCOME_TO_ROLE_RATIO]
            assert not col.isnull().any()
            assert not np.isinf(col).any()
        except FileNotFoundError:
            pytest.skip("Source CSV not available — skipping integration test")


# ---------------------------------------------------------------------------
# build_features (master builder)
# ---------------------------------------------------------------------------

class TestBuildFeatures:

    def test_all_engineered_columns_added(self, typical_row):
        result = build_features(typical_row)
        for col in ENGINEERED_COLUMNS:
            assert col in result.columns, f"Missing engineered column: {col}"

    def test_output_has_correct_extra_columns(self, typical_row):
        n_original = len(typical_row.columns)
        result = build_features(typical_row)
        assert len(result.columns) == n_original + len(ENGINEERED_COLUMNS)

    def test_idempotent(self, typical_row):
        """Calling build_features twice should produce the same result."""
        once = build_features(typical_row)
        twice = build_features(typical_row)
        pd.testing.assert_frame_equal(once, twice)

    def test_anomaly_count_matches_profiling(self):
        """
        Integration: the number of Role_Tenure_Anomaly==1 rows must match
        the count confirmed during dataset profiling (2,284).
        """
        try:
            from src.data.loader import load_data
            df = load_data()
            result = build_features(df)
            anomaly_count = result[COL_ROLE_TENURE_ANOMALY].sum()
            assert anomaly_count == EXPECTED_ANOMALY_COUNT, (
                f"Expected {EXPECTED_ANOMALY_COUNT} anomalous rows, "
                f"got {anomaly_count}."
            )
        except FileNotFoundError:
            pytest.skip("Source CSV not available — skipping integration test")

    def test_no_nans_in_engineered_columns(self):
        """Integration: no NaN in any engineered column on real data."""
        try:
            from src.data.loader import load_data
            df = load_data()
            result = build_features(df)
            null_counts = result[ENGINEERED_COLUMNS].isnull().sum()
            assert null_counts.sum() == 0, (
                f"NaN values found:\n{null_counts[null_counts > 0]}"
            )
        except FileNotFoundError:
            pytest.skip("Source CSV not available — skipping integration test")

    def test_build_features_all_min_values(self):
        """Edge case: all inputs at their minimum values."""
        df = _df(_base_row(
            Years_at_Company=1,
            Years_in_Current_Role=1,
            Years_Since_Last_Promotion=0,
            Job_Satisfaction=1,
            Work_Life_Balance=1,
            Work_Environment_Satisfaction=1,
            Relationship_with_Manager=1,
            Job_Involvement=1,
            Average_Hours_Worked_Per_Week=30,
            Monthly_Income=3000,
            Job_Level=1,
        ))
        result = build_features(df)
        # None of the engineered columns should be NaN or Inf
        for col in ENGINEERED_COLUMNS:
            val = result[col].iloc[0]
            assert np.isfinite(val), f"{col} = {val} (not finite) at min inputs"

    def test_build_features_all_max_values(self):
        """Edge case: all inputs at their maximum values."""
        df = _df(_base_row(
            Years_at_Company=29,
            Years_in_Current_Role=14,
            Years_Since_Last_Promotion=9,
            Job_Satisfaction=5,
            Work_Life_Balance=4,
            Work_Environment_Satisfaction=4,
            Relationship_with_Manager=4,
            Job_Involvement=4,
            Average_Hours_Worked_Per_Week=59,
            Monthly_Income=19999,
            Job_Level=5,
        ))
        result = build_features(df)
        for col in ENGINEERED_COLUMNS:
            val = result[col].iloc[0]
            assert np.isfinite(val), f"{col} = {val} (not finite) at max inputs"


# ---------------------------------------------------------------------------
# describe_engineered_features
# ---------------------------------------------------------------------------

class TestDescribeEngineeredFeatures:

    def test_returns_dataframe(self, typical_row):
        df_eng = build_features(typical_row)
        summary = describe_engineered_features(df_eng)
        assert isinstance(summary, pd.DataFrame)

    def test_correct_columns_in_summary(self, typical_row):
        df_eng = build_features(typical_row)
        summary = describe_engineered_features(df_eng)
        for col in ENGINEERED_COLUMNS:
            assert col in summary.columns

    def test_raises_if_engineered_columns_missing(self, typical_row):
        """Should raise ValueError if build_features has not been called."""
        with pytest.raises(ValueError, match="engineered columns are missing"):
            describe_engineered_features(typical_row)
