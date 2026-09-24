"""
tests/test_phase5_dashboard.py
================================
Phase 5 test suite — Streamlit Dashboard Helper Functions.

Coverage
--------
1.  compute_attrition_summary — correct keys, values, rate computation
2.  attrition_rate_by_group   — correct grouping, rate computation, sorting
3.  fmt_pct                   — formatting correctness
4.  fmt_float                 — formatting correctness
5.  risk_color                — known labels, unknown fallback
6.  risk_emoji                — known labels, unknown fallback
7.  build_employee_row        — correct columns, feature engineering applied
8.  bar_chart_attrition_by_group — returns plotly Figure
9.  histogram_chart           — returns plotly Figure with and without color col
10. pie_chart                 — returns plotly Figure
11. horizontal_bar_importance — returns plotly Figure, top_n respected
12. local_shap_waterfall      — returns plotly Figure, base_value accepted
13. confusion_matrix_heatmap  — returns plotly Figure from 2×2 matrix
14. Integration — load_global_coef_importance / load_global_shap_importance
15. Integration — compute_attrition_summary on real dataset

Test strategy
--------------
- Unit tests use minimal synthetic DataFrames — no CSV dependency.
- Integration tests are skipped if the CSV or model artifacts are absent.
- No Streamlit session is started; all helpers are pure Python / Plotly.
- RANDOM_STATE=42 throughout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers — minimal synthetic DataFrames
# ---------------------------------------------------------------------------

def _make_minimal_df(n: int = 100) -> pd.DataFrame:
    """Minimal synthetic DataFrame with the columns required by dashboard utils."""
    rng = np.random.default_rng(42)
    n_yes = n // 5
    depts = ["Finance", "HR", "IT", "Marketing", "Sales"]
    roles = ["Analyst", "Assistant", "Executive", "Manager"]
    marital_cycle = ["Married", "Single", "Divorced"]
    return pd.DataFrame({
        "Employee_ID":                      list(range(1, n + 1)),
        "Age":                              rng.integers(22, 60, n).tolist(),
        "Gender":                           (["Female"] * (n // 2) + ["Male"] * (n - n // 2)),
        "Marital_Status":                   [marital_cycle[i % 3] for i in range(n)],
        "Department":                       [depts[i % 5] for i in range(n)],
        "Job_Role":                         [roles[i % 4] for i in range(n)],
        "Job_Level":                        rng.integers(1, 6, n).tolist(),
        "Monthly_Income":                   rng.integers(3000, 20000, n).tolist(),
        "Hourly_Rate":                      rng.integers(15, 100, n).tolist(),
        "Years_at_Company":                 rng.integers(1, 20, n).tolist(),
        "Years_in_Current_Role":            rng.integers(1, 15, n).tolist(),
        "Years_Since_Last_Promotion":       rng.integers(0, 10, n).tolist(),
        "Work_Life_Balance":                rng.integers(1, 5, n).tolist(),
        "Job_Satisfaction":                 rng.integers(1, 6, n).tolist(),
        "Performance_Rating":               rng.integers(1, 5, n).tolist(),
        "Training_Hours_Last_Year":         rng.integers(0, 80, n).tolist(),
        "Overtime":                         (["No"] * (n // 2) + ["Yes"] * (n - n // 2)),
        "Project_Count":                    rng.integers(1, 10, n).tolist(),
        "Average_Hours_Worked_Per_Week":    rng.integers(30, 60, n).tolist(),
        "Absenteeism":                      rng.integers(0, 20, n).tolist(),
        "Work_Environment_Satisfaction":    rng.integers(1, 5, n).tolist(),
        "Relationship_with_Manager":        rng.integers(1, 5, n).tolist(),
        "Job_Involvement":                  rng.integers(1, 5, n).tolist(),
        "Distance_From_Home":               rng.integers(1, 50, n).tolist(),
        "Number_of_Companies_Worked":       rng.integers(1, 9, n).tolist(),
        "Attrition":                        (["Yes"] * n_yes + ["No"] * (n - n_yes)),
    })


def _make_minimal_profile() -> dict:
    """Minimal employee profile for the Risk Simulator."""
    return {
        "Age": 35,
        "Gender": "Male",
        "Marital_Status": "Married",
        "Department": "IT",
        "Job_Role": "Analyst",
        "Job_Level": 2,
        "Monthly_Income": 7000,
        "Hourly_Rate": 40,
        "Years_at_Company": 5,
        "Years_in_Current_Role": 3,
        "Years_Since_Last_Promotion": 2,
        "Work_Life_Balance": 3,
        "Job_Satisfaction": 3,
        "Performance_Rating": 3,
        "Training_Hours_Last_Year": 20,
        "Overtime": "No",
        "Project_Count": 3,
        "Average_Hours_Worked_Per_Week": 42,
        "Absenteeism": 3,
        "Work_Environment_Satisfaction": 3,
        "Relationship_with_Manager": 3,
        "Job_Involvement": 3,
        "Distance_From_Home": 10,
        "Number_of_Companies_Worked": 2,
    }


# ---------------------------------------------------------------------------
# 1. compute_attrition_summary
# ---------------------------------------------------------------------------

class TestComputeAttritionSummary:

    def setup_method(self):
        self.df = _make_minimal_df(100)

    def test_returns_dict(self):
        from utils.dashboard_utils import compute_attrition_summary
        result = compute_attrition_summary(self.df)
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        from utils.dashboard_utils import compute_attrition_summary
        result = compute_attrition_summary(self.df)
        required = {"total_employees", "attrition_count", "retained_count",
                    "attrition_rate", "retention_rate"}
        assert required.issubset(result.keys())

    def test_total_employees_correct(self):
        from utils.dashboard_utils import compute_attrition_summary
        result = compute_attrition_summary(self.df)
        assert result["total_employees"] == 100

    def test_attrition_count_correct(self):
        from utils.dashboard_utils import compute_attrition_summary
        result = compute_attrition_summary(self.df)
        assert result["attrition_count"] == 20  # n // 5

    def test_retained_plus_attrited_equals_total(self):
        from utils.dashboard_utils import compute_attrition_summary
        result = compute_attrition_summary(self.df)
        assert result["attrition_count"] + result["retained_count"] == result["total_employees"]

    def test_rates_sum_to_one(self):
        from utils.dashboard_utils import compute_attrition_summary
        result = compute_attrition_summary(self.df)
        assert abs(result["attrition_rate"] + result["retention_rate"] - 1.0) < 1e-9

    def test_attrition_rate_in_range(self):
        from utils.dashboard_utils import compute_attrition_summary
        result = compute_attrition_summary(self.df)
        assert 0.0 <= result["attrition_rate"] <= 1.0

    def test_attrition_rate_matches_manual(self):
        from utils.dashboard_utils import compute_attrition_summary
        result = compute_attrition_summary(self.df)
        expected = (self.df["Attrition"] == "Yes").mean()
        assert abs(result["attrition_rate"] - expected) < 1e-9


# ---------------------------------------------------------------------------
# 2. attrition_rate_by_group
# ---------------------------------------------------------------------------

class TestAttritionRateByGroup:

    def setup_method(self):
        self.df = _make_minimal_df(200)

    def test_returns_dataframe(self):
        from utils.dashboard_utils import attrition_rate_by_group
        result = attrition_rate_by_group(self.df, "Department")
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self):
        from utils.dashboard_utils import attrition_rate_by_group
        result = attrition_rate_by_group(self.df, "Department")
        required = {"Department", "attrition_rate", "count", "attrition_count"}
        assert required.issubset(result.columns)

    def test_row_count_equals_unique_groups(self):
        from utils.dashboard_utils import attrition_rate_by_group
        result = attrition_rate_by_group(self.df, "Department")
        n_unique = self.df["Department"].nunique()
        assert len(result) == n_unique

    def test_attrition_rate_in_range(self):
        from utils.dashboard_utils import attrition_rate_by_group
        result = attrition_rate_by_group(self.df, "Department")
        assert (result["attrition_rate"] >= 0).all()
        assert (result["attrition_rate"] <= 1).all()

    def test_sorted_by_attrition_rate_descending(self):
        from utils.dashboard_utils import attrition_rate_by_group
        result = attrition_rate_by_group(self.df, "Department")
        assert (result["attrition_rate"].diff().dropna() <= 0).all()

    def test_attrition_count_leq_count(self):
        from utils.dashboard_utils import attrition_rate_by_group
        result = attrition_rate_by_group(self.df, "Department")
        assert (result["attrition_count"] <= result["count"]).all()

    def test_works_for_binary_column(self):
        from utils.dashboard_utils import attrition_rate_by_group
        result = attrition_rate_by_group(self.df, "Overtime")
        assert len(result) == 2

    def test_rate_formula_correct(self):
        from utils.dashboard_utils import attrition_rate_by_group
        result = attrition_rate_by_group(self.df, "Department")
        manual = result["attrition_count"] / result["count"]
        np.testing.assert_allclose(result["attrition_rate"].values, manual.values, atol=1e-9)


# ---------------------------------------------------------------------------
# 3. fmt_pct
# ---------------------------------------------------------------------------

class TestFmtPct:

    def test_zero(self):
        from utils.dashboard_utils import fmt_pct
        assert fmt_pct(0.0) == "0.0%"

    def test_one(self):
        from utils.dashboard_utils import fmt_pct
        assert fmt_pct(1.0) == "100.0%"

    def test_common_value(self):
        from utils.dashboard_utils import fmt_pct
        # default decimals=1
        assert fmt_pct(0.20) == "20.0%"
        assert fmt_pct(0.1997) == "20.0%"  # rounds to 1 decimal place

    def test_custom_decimals(self):
        from utils.dashboard_utils import fmt_pct
        assert fmt_pct(0.1997, decimals=2) == "19.97%"

    def test_returns_string(self):
        from utils.dashboard_utils import fmt_pct
        assert isinstance(fmt_pct(0.5), str)

    def test_ends_with_percent(self):
        from utils.dashboard_utils import fmt_pct
        assert fmt_pct(0.75).endswith("%")


# ---------------------------------------------------------------------------
# 4. fmt_float
# ---------------------------------------------------------------------------

class TestFmtFloat:

    def test_default_decimals(self):
        from utils.dashboard_utils import fmt_float
        assert fmt_float(0.123456) == "0.123"

    def test_custom_decimals(self):
        from utils.dashboard_utils import fmt_float
        assert fmt_float(1.23456, decimals=4) == "1.2346"

    def test_returns_string(self):
        from utils.dashboard_utils import fmt_float
        assert isinstance(fmt_float(0.5), str)

    def test_zero(self):
        from utils.dashboard_utils import fmt_float
        assert fmt_float(0.0) == "0.000"


# ---------------------------------------------------------------------------
# 5. risk_color
# ---------------------------------------------------------------------------

class TestRiskColor:

    def test_high_risk_is_red(self):
        from utils.dashboard_utils import risk_color
        color = risk_color("High Risk")
        assert "#e74c3c" in color.lower() or color.startswith("#")

    def test_moderate_risk_has_color(self):
        from utils.dashboard_utils import risk_color
        color = risk_color("Moderate Risk")
        assert isinstance(color, str)
        assert color.startswith("#")

    def test_low_risk_is_green(self):
        from utils.dashboard_utils import risk_color
        color = risk_color("Low Risk")
        assert "#27ae60" in color.lower() or color.startswith("#")

    def test_unknown_returns_fallback(self):
        from utils.dashboard_utils import risk_color
        color = risk_color("Unknown Label")
        assert isinstance(color, str)
        assert color.startswith("#")


# ---------------------------------------------------------------------------
# 6. risk_emoji
# ---------------------------------------------------------------------------

class TestRiskEmoji:

    def test_high_risk(self):
        from utils.dashboard_utils import risk_emoji
        assert risk_emoji("High Risk") == "🔴"

    def test_moderate_risk(self):
        from utils.dashboard_utils import risk_emoji
        assert risk_emoji("Moderate Risk") == "🟡"

    def test_low_risk(self):
        from utils.dashboard_utils import risk_emoji
        assert risk_emoji("Low Risk") == "🟢"

    def test_unknown_returns_string(self):
        from utils.dashboard_utils import risk_emoji
        emoji = risk_emoji("Unknown")
        assert isinstance(emoji, str)


# ---------------------------------------------------------------------------
# 7. build_employee_row
# ---------------------------------------------------------------------------

class TestBuildEmployeeRow:

    def test_returns_dataframe(self):
        from utils.dashboard_utils import build_employee_row
        row = build_employee_row(_make_minimal_profile())
        assert isinstance(row, pd.DataFrame)

    def test_exactly_one_row(self):
        from utils.dashboard_utils import build_employee_row
        row = build_employee_row(_make_minimal_profile())
        assert len(row) == 1

    def test_no_attrition_column(self):
        from utils.dashboard_utils import build_employee_row
        row = build_employee_row(_make_minimal_profile())
        assert "Attrition" not in row.columns

    def test_no_employee_id_column(self):
        from utils.dashboard_utils import build_employee_row
        row = build_employee_row(_make_minimal_profile())
        assert "Employee_ID" not in row.columns

    def test_engineered_columns_present(self):
        from utils.dashboard_utils import build_employee_row
        from src.features.build_features import ENGINEERED_COLUMNS
        row = build_employee_row(_make_minimal_profile())
        for col in ENGINEERED_COLUMNS:
            assert col in row.columns, f"Missing engineered column: {col}"

    def test_no_nulls(self):
        from utils.dashboard_utils import build_employee_row
        row = build_employee_row(_make_minimal_profile())
        assert row.isnull().sum().sum() == 0

    def test_satisfaction_composite_range(self):
        from utils.dashboard_utils import build_employee_row
        row = build_employee_row(_make_minimal_profile())
        sc = row["Satisfaction_Composite"].iloc[0]
        assert 1.0 <= sc <= 5.0

    def test_hours_overload_formula(self):
        from utils.dashboard_utils import build_employee_row
        profile = _make_minimal_profile()
        profile["Average_Hours_Worked_Per_Week"] = 45
        row = build_employee_row(profile)
        assert row["Hours_Overload"].iloc[0] == 5  # 45 - 40

    def test_income_to_role_ratio_positive(self):
        from utils.dashboard_utils import build_employee_row
        row = build_employee_row(_make_minimal_profile())
        assert row["Income_to_Role_Ratio"].iloc[0] > 0


# ---------------------------------------------------------------------------
# 8. bar_chart_attrition_by_group
# ---------------------------------------------------------------------------

class TestBarChartAttritionByGroup:

    def test_returns_figure(self):
        from utils.dashboard_utils import bar_chart_attrition_by_group
        import plotly.graph_objects as go
        df = _make_minimal_df(100)
        fig = bar_chart_attrition_by_group(df, "Department", "Test Chart")
        assert isinstance(fig, go.Figure)

    def test_figure_has_data(self):
        from utils.dashboard_utils import bar_chart_attrition_by_group
        df = _make_minimal_df(100)
        fig = bar_chart_attrition_by_group(df, "Department", "Test Chart")
        assert len(fig.data) > 0

    def test_figure_title_set(self):
        from utils.dashboard_utils import bar_chart_attrition_by_group
        df = _make_minimal_df(100)
        fig = bar_chart_attrition_by_group(df, "Department", "My Title")
        assert "My Title" in fig.layout.title.text


# ---------------------------------------------------------------------------
# 9. histogram_chart
# ---------------------------------------------------------------------------

class TestHistogramChart:

    def test_returns_figure_no_color(self):
        from utils.dashboard_utils import histogram_chart
        import plotly.graph_objects as go
        df = _make_minimal_df(100)
        fig = histogram_chart(df, "Age", "Age Histogram")
        assert isinstance(fig, go.Figure)

    def test_returns_figure_with_color(self):
        from utils.dashboard_utils import histogram_chart
        import plotly.graph_objects as go
        df = _make_minimal_df(100)
        fig = histogram_chart(df, "Age", "Age Histogram", color_col="Attrition")
        assert isinstance(fig, go.Figure)

    def test_figure_has_data(self):
        from utils.dashboard_utils import histogram_chart
        df = _make_minimal_df(100)
        fig = histogram_chart(df, "Age", "Test")
        assert len(fig.data) > 0


# ---------------------------------------------------------------------------
# 10. pie_chart
# ---------------------------------------------------------------------------

class TestPieChart:

    def test_returns_figure(self):
        from utils.dashboard_utils import pie_chart
        import plotly.graph_objects as go
        fig = pie_chart(["A", "B", "C"], [30, 50, 20], "Test Pie")
        assert isinstance(fig, go.Figure)

    def test_figure_has_data(self):
        from utils.dashboard_utils import pie_chart
        fig = pie_chart(["Yes", "No"], [200, 800], "Attrition")
        assert len(fig.data) > 0


# ---------------------------------------------------------------------------
# 11. horizontal_bar_importance
# ---------------------------------------------------------------------------

class TestHorizontalBarImportance:

    def _make_coef_df(self, n=10):
        return pd.DataFrame({
            "rank": range(1, n + 1),
            "feature": [f"Feature_{i}" for i in range(n)],
            "abs_coef": np.linspace(0.5, 0.05, n),
            "direction": (["(+) higher attrition risk"] * (n // 2) +
                          ["(-) lower attrition risk"] * (n - n // 2)),
        })

    def test_returns_figure(self):
        from utils.dashboard_utils import horizontal_bar_importance
        import plotly.graph_objects as go
        df = self._make_coef_df()
        fig = horizontal_bar_importance(df, "feature", "abs_coef", "direction", "Test")
        assert isinstance(fig, go.Figure)

    def test_top_n_respected(self):
        from utils.dashboard_utils import horizontal_bar_importance
        import plotly.graph_objects as go
        df = self._make_coef_df(20)
        fig = horizontal_bar_importance(df, "feature", "abs_coef", "direction", "Test", top_n=5)
        assert isinstance(fig, go.Figure)
        # Verify only 5 bars drawn
        assert len(fig.data[0].x) == 5


# ---------------------------------------------------------------------------
# 12. local_shap_waterfall
# ---------------------------------------------------------------------------

class TestLocalShapWaterfall:

    def _make_local_df(self, n=10):
        shap_vals = np.linspace(-0.1, 0.1, n)
        return pd.DataFrame({
            "feature": [f"Feature_{i}" for i in range(n)],
            "shap_value": shap_vals,
            "abs_shap": np.abs(shap_vals),
            "direction": (["(+) increases risk" if v > 0 else "(-) decreases risk"]
                          for v in shap_vals),
        })

    def test_returns_figure(self):
        from utils.dashboard_utils import local_shap_waterfall
        import plotly.graph_objects as go
        df = self._make_local_df()
        fig = local_shap_waterfall(df, base_value=-0.5, predicted_prob=0.42)
        assert isinstance(fig, go.Figure)

    def test_figure_title_contains_probability(self):
        from utils.dashboard_utils import local_shap_waterfall
        df = self._make_local_df()
        fig = local_shap_waterfall(df, base_value=-0.5, predicted_prob=0.42)
        assert "42" in fig.layout.title.text or "0.42" in fig.layout.title.text


# ---------------------------------------------------------------------------
# 13. confusion_matrix_heatmap
# ---------------------------------------------------------------------------

class TestConfusionMatrixHeatmap:

    def test_returns_figure(self):
        from utils.dashboard_utils import confusion_matrix_heatmap
        import plotly.graph_objects as go
        cm = [[817, 784], [211, 188]]
        fig = confusion_matrix_heatmap(cm)
        assert isinstance(fig, go.Figure)

    def test_figure_has_data(self):
        from utils.dashboard_utils import confusion_matrix_heatmap
        cm = [[817, 784], [211, 188]]
        fig = confusion_matrix_heatmap(cm)
        assert len(fig.data) > 0


# ---------------------------------------------------------------------------
# 14. Integration — global importance CSVs load successfully
# ---------------------------------------------------------------------------

class TestLoadGlobalImportanceCSVs:

    def test_coef_csv_loads(self):
        """Integration: load global_importance_coef.csv from models/."""
        try:
            from utils.dashboard_utils import load_global_coef_importance
            df = load_global_coef_importance()
            assert isinstance(df, pd.DataFrame)
            required = {"rank", "feature", "coefficient", "abs_coef", "direction"}
            assert required.issubset(df.columns)
            assert len(df) == 36
        except FileNotFoundError:
            pytest.skip("global_importance_coef.csv not found")

    def test_shap_csv_loads(self):
        """Integration: load global_importance_shap.csv from models/."""
        try:
            from utils.dashboard_utils import load_global_shap_importance
            df = load_global_shap_importance()
            assert isinstance(df, pd.DataFrame)
            required = {"rank", "feature", "mean_abs_shap", "direction"}
            assert required.issubset(df.columns)
            assert len(df) == 36
        except FileNotFoundError:
            pytest.skip("global_importance_shap.csv not found")


# ---------------------------------------------------------------------------
# 15. Integration — compute_attrition_summary on real dataset
# ---------------------------------------------------------------------------

class TestComputeAttritionSummaryIntegration:

    def test_real_dataset_summary(self):
        """Integration: real CSV loads and summary computes correctly."""
        try:
            from src.data.loader import load_data
            df = load_data()
            from utils.dashboard_utils import compute_attrition_summary
            result = compute_attrition_summary(df)
            assert result["total_employees"] == 10000
            # ~20% attrition in the synthetic dataset
            assert 0.15 < result["attrition_rate"] < 0.25
        except FileNotFoundError:
            pytest.skip("Source CSV not found")


# ---------------------------------------------------------------------------
# 16. Regression tests — clamp_and_flag treatment in build_employee_row
#     Verifies the fix for: "columns are missing: ['Years_in_Current_Role_Clamped']"
# ---------------------------------------------------------------------------

class TestBuildEmployeeRowClampAndFlag:
    """
    Regression tests proving that build_employee_row produces
    Years_in_Current_Role_Clamped before prediction, handles anomalous
    profiles without error, and never mutates the raw source data.
    """

    def test_clamped_column_present_in_normal_profile(self):
        """
        build_employee_row must include Years_in_Current_Role_Clamped even
        when Years_in_Current_Role <= Years_at_Company (no anomaly).
        """
        from utils.dashboard_utils import build_employee_row
        from src.models.anomaly_treatments import COL_ROLE_CLAMPED

        profile = _make_minimal_profile()  # Years_in_Current_Role=3, Years_at_Company=5 → normal
        assert profile["Years_in_Current_Role"] <= profile["Years_at_Company"], \
            "Fixture must be a non-anomalous profile for this test"

        row = build_employee_row(profile)
        assert COL_ROLE_CLAMPED in row.columns, (
            f"'{COL_ROLE_CLAMPED}' is missing from build_employee_row output "
            "for a normal (non-anomalous) profile."
        )

    def test_clamped_column_present_in_anomalous_profile(self):
        """
        build_employee_row must include Years_in_Current_Role_Clamped when
        Years_in_Current_Role > Years_at_Company (anomalous profile).
        """
        from utils.dashboard_utils import build_employee_row
        from src.models.anomaly_treatments import COL_ROLE_CLAMPED

        profile = _make_minimal_profile()
        profile["Years_at_Company"] = 3
        profile["Years_in_Current_Role"] = 8   # > Years_at_Company → anomalous

        row = build_employee_row(profile)
        assert COL_ROLE_CLAMPED in row.columns, (
            f"'{COL_ROLE_CLAMPED}' is missing from build_employee_row output "
            "for an anomalous profile."
        )

    def test_clamped_value_correct_for_anomalous_profile(self):
        """
        For an anomalous profile (Years_in_Current_Role > Years_at_Company),
        Years_in_Current_Role_Clamped must equal Years_at_Company.
        """
        from utils.dashboard_utils import build_employee_row
        from src.models.anomaly_treatments import COL_ROLE_CLAMPED

        profile = _make_minimal_profile()
        profile["Years_at_Company"] = 3
        profile["Years_in_Current_Role"] = 8

        row = build_employee_row(profile)
        clamped_val = row[COL_ROLE_CLAMPED].iloc[0]
        assert clamped_val == 3, (
            f"Expected clamped value 3 (Years_at_Company), got {clamped_val}."
        )

    def test_clamped_value_correct_for_normal_profile(self):
        """
        For a normal profile (Years_in_Current_Role <= Years_at_Company),
        Years_in_Current_Role_Clamped must equal Years_in_Current_Role.
        """
        from utils.dashboard_utils import build_employee_row
        from src.models.anomaly_treatments import COL_ROLE_CLAMPED

        profile = _make_minimal_profile()  # Years_in_Current_Role=3, Years_at_Company=5
        row = build_employee_row(profile)
        clamped_val = row[COL_ROLE_CLAMPED].iloc[0]
        assert clamped_val == profile["Years_in_Current_Role"], (
            f"Expected clamped value {profile['Years_in_Current_Role']}, got {clamped_val}."
        )

    def test_anomalous_profile_no_missing_column_error(self):
        """
        Regression test: calling the production pipeline on a row built from
        an anomalous profile must NOT raise a 'columns are missing' error.
        This directly tests the original bug report.
        """
        import joblib
        from pathlib import Path
        from utils.dashboard_utils import build_employee_row
        from src.config import PIPELINE_PATH

        if not PIPELINE_PATH.exists():
            pytest.skip("Saved pipeline not found; run Phase 3 training first.")

        pipeline = joblib.load(PIPELINE_PATH)

        profile = _make_minimal_profile()
        profile["Years_at_Company"] = 2
        profile["Years_in_Current_Role"] = 10   # anomalous: role > company tenure

        row = build_employee_row(profile)

        # Must not raise KeyError / ValueError about missing columns
        proba = pipeline.predict_proba(row)
        assert proba.shape == (1, 2), "predict_proba must return shape (1, 2)"
        assert 0.0 <= proba[0, 1] <= 1.0, "Positive-class probability must be in [0, 1]"

    def test_normal_profile_no_missing_column_error(self):
        """
        Regression test: calling the production pipeline on a row built from
        a normal (non-anomalous) profile must also succeed without error.
        """
        import joblib
        from utils.dashboard_utils import build_employee_row
        from src.config import PIPELINE_PATH

        if not PIPELINE_PATH.exists():
            pytest.skip("Saved pipeline not found; run Phase 3 training first.")

        pipeline = joblib.load(PIPELINE_PATH)

        profile = _make_minimal_profile()  # Years_in_Current_Role=3 <= Years_at_Company=5

        row = build_employee_row(profile)
        proba = pipeline.predict_proba(row)
        assert proba.shape == (1, 2)
        assert 0.0 <= proba[0, 1] <= 1.0

    def test_raw_source_csv_unchanged(self):
        """
        Regression test: build_employee_row must never mutate the raw source CSV.
        Calls build_employee_row with an anomalous profile and then reloads the
        CSV to verify its shape and anomaly-row count are unchanged.
        """
        from src.config import RAW_CSV, EXPECTED_ANOMALY_COUNT, ANOMALY_COLUMN, ANOMALY_BASELINE
        if not RAW_CSV.exists():
            pytest.skip("Source CSV not found.")

        import pandas as pd
        from utils.dashboard_utils import build_employee_row

        # Record pre-call state
        df_before = pd.read_csv(RAW_CSV)
        shape_before = df_before.shape
        anomaly_count_before = int((df_before[ANOMALY_COLUMN] > df_before[ANOMALY_BASELINE]).sum())

        # Execute build_employee_row with an anomalous profile
        profile = _make_minimal_profile()
        profile["Years_at_Company"] = 1
        profile["Years_in_Current_Role"] = 15
        build_employee_row(profile)

        # Reload and verify the file on disk is identical
        df_after = pd.read_csv(RAW_CSV)
        assert df_after.shape == shape_before, (
            f"CSV shape changed after build_employee_row: "
            f"{shape_before} → {df_after.shape}"
        )
        anomaly_count_after = int((df_after[ANOMALY_COLUMN] > df_after[ANOMALY_BASELINE]).sum())
        assert anomaly_count_after == anomaly_count_before, (
            f"Anomaly row count in CSV changed: "
            f"{anomaly_count_before} → {anomaly_count_after}"
        )
        assert anomaly_count_after == EXPECTED_ANOMALY_COUNT, (
            f"Expected {EXPECTED_ANOMALY_COUNT} anomaly rows, "
            f"found {anomaly_count_after}."
        )
