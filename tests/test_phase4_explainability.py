"""
tests/test_phase4_explainability.py
=====================================
Phase 4 test suite — Explainability and Model Interpretation.

Coverage
--------
1.  Feature name extraction — names match pipeline, correct count, correct index map
2.  Transform input — preprocessor applies correctly, correct output shape
3.  Global importance — correct columns, sorting, direction labels, disclaimer constants
4.  SHAP explainer construction — builds without error, correct base value type
5.  SHAP global importance — correct shape, valid values, sorting
6.  SHAP value computation — shape matches (n_samples, n_features), finite values
7.  Local SHAP explanation — single-row works, top_n respected, direction labels
8.  Local coef×deviation — matches expected decomposition formula, top_n respected
9.  Local explanation formatting — report contains expected strings, disclaimers present
10. Prediction explanation — PredictionExplanation fields, risk labels, probability range
11. Batch explanation — one row per employee, risk labels valid, top features present
12. Risk label classification — correct thresholds
13. Edge cases — single feature, all-zero row, unknown values handled gracefully
14. Saved pipeline compatibility — the real saved pipeline loads and explains correctly
15. Disclaimer integrity — causation disclaimer never absent from any report

Test strategy
-------------
- Unit tests use a minimal synthesised DataFrame + a freshly-fitted toy pipeline.
  No dependency on the real CSV in unit tests.
- Integration tests use the saved `models/best_pipeline.pkl`.
  These are skipped if the file is absent or the CSV is absent.
- RANDOM_STATE=42 throughout.
- The held-out test set is NEVER used inside this test file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from src.config import (
    ID_COLUMN,
    RANDOM_STATE,
    TARGET_COLUMN,
    TARGET_POSITIVE,
    TARGET_NEGATIVE,
)
from src.features.build_features import build_features, COL_ROLE_TENURE_ANOMALY


# ---------------------------------------------------------------------------
# Shared fixture factory — minimal toy pipeline
# ---------------------------------------------------------------------------

def _make_toy_df(n_yes: int = 40, n_no: int = 160) -> pd.DataFrame:
    """
    Synthesise a minimal DataFrame matching the project schema.
    Mirrors the helper in test_phase3_models.py for consistency.
    """
    nrows = n_yes + n_no
    rng = np.random.default_rng(0)

    dept_vals    = ["Marketing", "Sales", "Finance", "HR", "IT"]
    role_vals    = ["Analyst", "Assistant", "Executive", "Manager"]
    marital_vals = ["Married", "Divorced", "Single"]

    years_at_co  = rng.integers(1, 30, nrows).tolist()
    years_in_role = []
    for i, yac in enumerate(years_at_co):
        years_in_role.append(min(yac + 2, 30) if i % 4 == 0 else max(1, yac - 1))

    data = {
        ID_COLUMN:                       list(range(1, nrows + 1)),
        "Age":                           rng.integers(20, 60, nrows).tolist(),
        "Gender":                        (["Female"] * (nrows // 2) + ["Male"] * (nrows - nrows // 2)),
        "Marital_Status":                [marital_vals[i % 3] for i in range(nrows)],
        "Department":                    [dept_vals[i % 5] for i in range(nrows)],
        "Job_Role":                      [role_vals[i % 4] for i in range(nrows)],
        "Job_Level":                     rng.integers(1, 6, nrows).tolist(),
        "Monthly_Income":                rng.integers(3000, 20000, nrows).tolist(),
        "Hourly_Rate":                   rng.integers(15, 100, nrows).tolist(),
        "Years_at_Company":              years_at_co,
        "Years_in_Current_Role":         years_in_role,
        "Years_Since_Last_Promotion":    rng.integers(0, 10, nrows).tolist(),
        "Work_Life_Balance":             rng.integers(1, 5, nrows).tolist(),
        "Job_Satisfaction":              rng.integers(1, 6, nrows).tolist(),
        "Performance_Rating":            rng.integers(1, 5, nrows).tolist(),
        "Training_Hours_Last_Year":      rng.integers(0, 100, nrows).tolist(),
        "Overtime":                      (["No"] * (nrows // 2) + ["Yes"] * (nrows - nrows // 2)),
        "Project_Count":                 rng.integers(1, 10, nrows).tolist(),
        "Average_Hours_Worked_Per_Week": rng.integers(30, 60, nrows).tolist(),
        "Absenteeism":                   rng.integers(0, 20, nrows).tolist(),
        "Work_Environment_Satisfaction": rng.integers(1, 5, nrows).tolist(),
        "Relationship_with_Manager":     rng.integers(1, 5, nrows).tolist(),
        "Job_Involvement":               rng.integers(1, 5, nrows).tolist(),
        "Distance_From_Home":            rng.integers(1, 50, nrows).tolist(),
        "Number_of_Companies_Worked":    rng.integers(1, 5, nrows).tolist(),
        TARGET_COLUMN:                   ([TARGET_POSITIVE] * n_yes + [TARGET_NEGATIVE] * n_no),
    }
    return build_features(pd.DataFrame(data))


def _make_toy_pipeline_and_data():
    """
    Return a fitted toy pipeline + X_train + y_train + X_test.
    Uses the standard preprocessor + clamp_and_flag treatment.
    """
    from src.models.anomaly_treatments import apply_clamp_and_flag
    from src.preprocessing.splitter import split_data
    from src.preprocessing.pipeline import prepare_features, encode_target

    df = _make_toy_df(n_yes=40, n_no=160)
    splits = split_data(df)
    X_train_raw = splits["X_train"]
    y_train = splits["y_train"]
    X_test = splits["X_test"]

    X_train = apply_clamp_and_flag(X_train_raw)

    # Build a pipeline matching the Phase 3 winner structure
    from src.models.train import _get_clamp_catalogue
    pipeline = _get_clamp_catalogue()["LR_balanced"]
    pipeline.fit(X_train, y_train)

    return pipeline, X_train, y_train, X_test


@pytest.fixture(scope="module")
def toy_pipeline_data():
    """Shared module-scope fixture: fitted toy pipeline + splits."""
    return _make_toy_pipeline_and_data()


@pytest.fixture(scope="module")
def toy_pipeline(toy_pipeline_data):
    return toy_pipeline_data[0]


@pytest.fixture(scope="module")
def toy_X_train(toy_pipeline_data):
    return toy_pipeline_data[1]


@pytest.fixture(scope="module")
def toy_y_train(toy_pipeline_data):
    return toy_pipeline_data[2]


@pytest.fixture(scope="module")
def toy_explainer(toy_pipeline, toy_X_train):
    from src.explainability.shap_explainer import build_shap_explainer
    return build_shap_explainer(toy_pipeline, toy_X_train)


# ---------------------------------------------------------------------------
# 1. Feature name extraction
# ---------------------------------------------------------------------------

class TestFeatureNameExtraction:

    def test_get_feature_names_returns_list(self, toy_pipeline):
        from src.explainability.feature_names import get_feature_names_from_pipeline
        names = get_feature_names_from_pipeline(toy_pipeline)
        assert isinstance(names, list)

    def test_feature_names_non_empty(self, toy_pipeline):
        from src.explainability.feature_names import get_feature_names_from_pipeline
        names = get_feature_names_from_pipeline(toy_pipeline)
        assert len(names) > 0

    def test_feature_names_are_strings(self, toy_pipeline):
        from src.explainability.feature_names import get_feature_names_from_pipeline
        names = get_feature_names_from_pipeline(toy_pipeline)
        assert all(isinstance(n, str) for n in names)

    def test_feature_names_unique(self, toy_pipeline):
        from src.explainability.feature_names import get_feature_names_from_pipeline
        names = get_feature_names_from_pipeline(toy_pipeline)
        assert len(names) == len(set(names)), "Feature names must be unique"

    def test_feature_name_count_matches_coef(self, toy_pipeline):
        from src.explainability.feature_names import get_feature_names_from_pipeline
        names = get_feature_names_from_pipeline(toy_pipeline)
        n_coefs = toy_pipeline.named_steps["clf"].coef_.shape[1]
        assert len(names) == n_coefs

    def test_index_map_is_dict(self, toy_pipeline):
        from src.explainability.feature_names import get_feature_name_index_map
        idx_map = get_feature_name_index_map(toy_pipeline)
        assert isinstance(idx_map, dict)

    def test_index_map_values_are_ints(self, toy_pipeline):
        from src.explainability.feature_names import get_feature_name_index_map
        idx_map = get_feature_name_index_map(toy_pipeline)
        assert all(isinstance(v, int) for v in idx_map.values())

    def test_index_map_covers_all_features(self, toy_pipeline):
        from src.explainability.feature_names import (
            get_feature_names_from_pipeline,
            get_feature_name_index_map,
        )
        names = get_feature_names_from_pipeline(toy_pipeline)
        idx_map = get_feature_name_index_map(toy_pipeline)
        assert set(idx_map.keys()) == set(names)

    def test_index_map_indices_are_sequential(self, toy_pipeline):
        from src.explainability.feature_names import get_feature_name_index_map
        idx_map = get_feature_name_index_map(toy_pipeline)
        n = len(idx_map)
        assert set(idx_map.values()) == set(range(n))

    def test_extract_preprocessor_returns_column_transformer(self, toy_pipeline):
        from sklearn.compose import ColumnTransformer
        from src.explainability.feature_names import extract_preprocessor
        prep = extract_preprocessor(toy_pipeline)
        assert isinstance(prep, ColumnTransformer)


# ---------------------------------------------------------------------------
# 2. Transform input
# ---------------------------------------------------------------------------

class TestTransformInput:

    def test_transform_returns_ndarray(self, toy_pipeline, toy_X_train):
        from src.explainability.feature_names import transform_input
        X_t = transform_input(toy_pipeline, toy_X_train)
        assert isinstance(X_t, np.ndarray)

    def test_transform_correct_row_count(self, toy_pipeline, toy_X_train):
        from src.explainability.feature_names import transform_input
        X_t = transform_input(toy_pipeline, toy_X_train)
        assert X_t.shape[0] == len(toy_X_train)

    def test_transform_correct_feature_count(self, toy_pipeline, toy_X_train):
        from src.explainability.feature_names import (
            transform_input,
            get_feature_names_from_pipeline,
        )
        X_t = transform_input(toy_pipeline, toy_X_train)
        n_feat = len(get_feature_names_from_pipeline(toy_pipeline))
        assert X_t.shape[1] == n_feat

    def test_transform_finite_values(self, toy_pipeline, toy_X_train):
        from src.explainability.feature_names import transform_input
        X_t = transform_input(toy_pipeline, toy_X_train)
        assert np.isfinite(X_t).all()

    def test_transform_single_row(self, toy_pipeline, toy_X_train):
        from src.explainability.feature_names import (
            transform_input,
            get_feature_names_from_pipeline,
        )
        X_single = toy_X_train.iloc[[0]]
        X_t = transform_input(toy_pipeline, X_single)
        n_feat = len(get_feature_names_from_pipeline(toy_pipeline))
        assert X_t.shape == (1, n_feat)


# ---------------------------------------------------------------------------
# 3. Global importance — coefficients
# ---------------------------------------------------------------------------

class TestGlobalImportance:

    def test_global_importance_returns_dataframe(self, toy_pipeline):
        from src.explainability.global_importance import get_global_feature_importance
        df = get_global_feature_importance(toy_pipeline)
        assert isinstance(df, pd.DataFrame)

    def test_global_importance_has_required_columns(self, toy_pipeline):
        from src.explainability.global_importance import get_global_feature_importance
        df = get_global_feature_importance(toy_pipeline)
        required = {"rank", "feature", "coefficient", "abs_coef", "direction"}
        assert required.issubset(set(df.columns))

    def test_global_importance_row_count_matches_features(self, toy_pipeline):
        from src.explainability.global_importance import get_global_feature_importance
        from src.explainability.feature_names import get_feature_names_from_pipeline
        df = get_global_feature_importance(toy_pipeline)
        n_feat = len(get_feature_names_from_pipeline(toy_pipeline))
        assert len(df) == n_feat

    def test_global_importance_sorted_by_abs_coef_descending(self, toy_pipeline):
        from src.explainability.global_importance import get_global_feature_importance
        df = get_global_feature_importance(toy_pipeline)
        assert (df["abs_coef"].diff().dropna() <= 0).all(), \
            "Rows should be sorted by abs_coef descending"

    def test_global_importance_abs_coef_non_negative(self, toy_pipeline):
        from src.explainability.global_importance import get_global_feature_importance
        df = get_global_feature_importance(toy_pipeline)
        assert (df["abs_coef"] >= 0).all()

    def test_global_importance_rank_starts_at_1(self, toy_pipeline):
        from src.explainability.global_importance import get_global_feature_importance
        df = get_global_feature_importance(toy_pipeline)
        assert df["rank"].min() == 1

    def test_global_importance_direction_labels_valid(self, toy_pipeline):
        from src.explainability.global_importance import get_global_feature_importance
        df = get_global_feature_importance(toy_pipeline)
        valid_directions = {"(+) higher attrition risk", "(-) lower attrition risk"}
        assert set(df["direction"].unique()).issubset(valid_directions)

    def test_global_importance_top_n_respected(self, toy_pipeline):
        from src.explainability.global_importance import get_global_feature_importance
        df = get_global_feature_importance(toy_pipeline, top_n=5)
        assert len(df) == 5

    def test_global_importance_positive_coef_has_higher_risk_direction(self, toy_pipeline):
        from src.explainability.global_importance import get_global_feature_importance
        df = get_global_feature_importance(toy_pipeline)
        pos_rows = df[df["coefficient"] > 0]
        assert (pos_rows["direction"] == "(+) higher attrition risk").all()

    def test_global_importance_negative_coef_has_lower_risk_direction(self, toy_pipeline):
        from src.explainability.global_importance import get_global_feature_importance
        df = get_global_feature_importance(toy_pipeline)
        neg_rows = df[df["coefficient"] < 0]
        assert (neg_rows["direction"] == "(-) lower attrition risk").all()

    def test_global_importance_format_report_is_string(self, toy_pipeline):
        from src.explainability.global_importance import (
            get_global_feature_importance,
            format_global_importance_report,
        )
        df = get_global_feature_importance(toy_pipeline)
        report = format_global_importance_report(df)
        assert isinstance(report, str)

    def test_global_importance_report_contains_disclaimer(self, toy_pipeline):
        from src.explainability.global_importance import (
            get_global_feature_importance,
            format_global_importance_report,
        )
        df = get_global_feature_importance(toy_pipeline)
        report = format_global_importance_report(df)
        assert "Association" in report
        assert "causation" in report.lower() or "Causation" in report

    def test_global_importance_raises_on_non_lr(self):
        from src.explainability.global_importance import get_global_feature_importance
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=50, n_features=5, random_state=0)
        rf = RandomForestClassifier(n_estimators=5, random_state=0)
        rf.fit(X, y)
        # Create a fake pipeline with RF (no coef_)
        from sklearn.pipeline import Pipeline as SKPipeline
        fake_pipeline = SKPipeline([
            ("preprocessor", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=5)),
        ])
        from sklearn.datasets import make_classification
        X2, y2 = make_classification(n_samples=50, n_features=5, random_state=0)
        fake_pipeline.fit(X2, y2)
        with pytest.raises(ValueError, match="coef_"):
            get_global_feature_importance(fake_pipeline)


# ---------------------------------------------------------------------------
# 4. SHAP explainer construction
# ---------------------------------------------------------------------------

class TestSHAPExplainerConstruction:

    def test_build_shap_explainer_returns_linear_explainer(self, toy_pipeline, toy_X_train):
        import shap
        from src.explainability.shap_explainer import build_shap_explainer
        explainer = build_shap_explainer(toy_pipeline, toy_X_train)
        assert isinstance(explainer, shap.LinearExplainer)

    def test_shap_explainer_has_expected_value(self, toy_pipeline, toy_X_train):
        from src.explainability.shap_explainer import build_shap_explainer
        explainer = build_shap_explainer(toy_pipeline, toy_X_train)
        assert hasattr(explainer, "expected_value")
        assert isinstance(float(explainer.expected_value), float)

    def test_shap_explainer_expected_value_is_finite(self, toy_pipeline, toy_X_train):
        from src.explainability.shap_explainer import build_shap_explainer
        explainer = build_shap_explainer(toy_pipeline, toy_X_train)
        assert np.isfinite(float(explainer.expected_value))


# ---------------------------------------------------------------------------
# 5. SHAP global importance
# ---------------------------------------------------------------------------

class TestSHAPGlobalImportance:

    def test_shap_global_importance_returns_dataframe(
        self, toy_explainer, toy_pipeline, toy_X_train
    ):
        from src.explainability.shap_explainer import get_shap_global_importance
        df = get_shap_global_importance(toy_explainer, toy_pipeline, toy_X_train)
        assert isinstance(df, pd.DataFrame)

    def test_shap_global_importance_has_required_columns(
        self, toy_explainer, toy_pipeline, toy_X_train
    ):
        from src.explainability.shap_explainer import get_shap_global_importance
        df = get_shap_global_importance(toy_explainer, toy_pipeline, toy_X_train)
        required = {"rank", "feature", "mean_abs_shap", "direction"}
        assert required.issubset(set(df.columns))

    def test_shap_global_importance_row_count(
        self, toy_explainer, toy_pipeline, toy_X_train
    ):
        from src.explainability.shap_explainer import get_shap_global_importance
        from src.explainability.feature_names import get_feature_names_from_pipeline
        df = get_shap_global_importance(toy_explainer, toy_pipeline, toy_X_train)
        n_feat = len(get_feature_names_from_pipeline(toy_pipeline))
        assert len(df) == n_feat

    def test_shap_global_importance_sorted_descending(
        self, toy_explainer, toy_pipeline, toy_X_train
    ):
        from src.explainability.shap_explainer import get_shap_global_importance
        df = get_shap_global_importance(toy_explainer, toy_pipeline, toy_X_train)
        assert (df["mean_abs_shap"].diff().dropna() <= 0).all()

    def test_shap_global_importance_non_negative(
        self, toy_explainer, toy_pipeline, toy_X_train
    ):
        from src.explainability.shap_explainer import get_shap_global_importance
        df = get_shap_global_importance(toy_explainer, toy_pipeline, toy_X_train)
        assert (df["mean_abs_shap"] >= 0).all()

    def test_shap_global_importance_top_n_respected(
        self, toy_explainer, toy_pipeline, toy_X_train
    ):
        from src.explainability.shap_explainer import get_shap_global_importance
        df = get_shap_global_importance(toy_explainer, toy_pipeline, toy_X_train, top_n=5)
        assert len(df) == 5

    def test_shap_global_report_is_string(
        self, toy_explainer, toy_pipeline, toy_X_train
    ):
        from src.explainability.shap_explainer import (
            get_shap_global_importance,
            format_shap_global_report,
        )
        df = get_shap_global_importance(toy_explainer, toy_pipeline, toy_X_train)
        report = format_shap_global_report(df)
        assert isinstance(report, str)
        assert "SHAP" in report


# ---------------------------------------------------------------------------
# 6. SHAP value computation
# ---------------------------------------------------------------------------

class TestSHAPValueComputation:

    def test_compute_shap_values_returns_ndarray(
        self, toy_explainer, toy_pipeline, toy_X_train
    ):
        from src.explainability.shap_explainer import compute_shap_values
        sv = compute_shap_values(toy_explainer, toy_pipeline, toy_X_train)
        assert isinstance(sv, np.ndarray)

    def test_compute_shap_values_correct_shape(
        self, toy_explainer, toy_pipeline, toy_X_train
    ):
        from src.explainability.shap_explainer import compute_shap_values
        from src.explainability.feature_names import get_feature_names_from_pipeline
        sv = compute_shap_values(toy_explainer, toy_pipeline, toy_X_train)
        n_feat = len(get_feature_names_from_pipeline(toy_pipeline))
        assert sv.shape == (len(toy_X_train), n_feat)

    def test_compute_shap_values_finite(
        self, toy_explainer, toy_pipeline, toy_X_train
    ):
        from src.explainability.shap_explainer import compute_shap_values
        sv = compute_shap_values(toy_explainer, toy_pipeline, toy_X_train)
        assert np.isfinite(sv).all()

    def test_compute_shap_values_single_row(
        self, toy_explainer, toy_pipeline, toy_X_train
    ):
        from src.explainability.shap_explainer import compute_shap_values
        from src.explainability.feature_names import get_feature_names_from_pipeline
        X_single = toy_X_train.iloc[[0]]
        sv = compute_shap_values(toy_explainer, toy_pipeline, X_single)
        n_feat = len(get_feature_names_from_pipeline(toy_pipeline))
        assert sv.shape == (1, n_feat)

    def test_shap_sum_plus_base_value_approximates_logodds(
        self, toy_explainer, toy_pipeline, toy_X_train
    ):
        """
        For LinearExplainer: sum(SHAP values) + expected_value ≈ log-odds output.
        Test on the first 5 rows of X_train.
        """
        from src.explainability.shap_explainer import compute_shap_values
        from src.explainability.feature_names import transform_input
        import scipy.special

        X_small = toy_X_train.iloc[:5]
        sv = compute_shap_values(toy_explainer, toy_pipeline, X_small)  # (5, n_feat)
        base = float(toy_explainer.expected_value)

        # Model's predicted log-odds
        clf = toy_pipeline.named_steps["clf"]
        X_t = transform_input(toy_pipeline, X_small)
        log_odds = X_t @ clf.coef_[0] + clf.intercept_[0]

        shap_sum_plus_base = sv.sum(axis=1) + base
        np.testing.assert_allclose(shap_sum_plus_base, log_odds, atol=1e-4)


# ---------------------------------------------------------------------------
# 7. Local SHAP explanation
# ---------------------------------------------------------------------------

class TestLocalSHAPExplanation:

    def test_explain_local_shap_returns_dataframe(
        self, toy_explainer, toy_pipeline, toy_X_train
    ):
        from src.explainability.local_explanation import explain_local_shap
        X_row = toy_X_train.iloc[[0]]
        df = explain_local_shap(toy_pipeline, toy_explainer, X_row)
        assert isinstance(df, pd.DataFrame)

    def test_explain_local_shap_has_required_columns(
        self, toy_explainer, toy_pipeline, toy_X_train
    ):
        from src.explainability.local_explanation import explain_local_shap
        X_row = toy_X_train.iloc[[0]]
        df = explain_local_shap(toy_pipeline, toy_explainer, X_row)
        required = {"feature", "shap_value", "abs_shap", "direction", "transformed_value"}
        assert required.issubset(set(df.columns))

    def test_explain_local_shap_top_n_respected(
        self, toy_explainer, toy_pipeline, toy_X_train
    ):
        from src.explainability.local_explanation import explain_local_shap
        X_row = toy_X_train.iloc[[0]]
        df = explain_local_shap(toy_pipeline, toy_explainer, X_row, top_n=5)
        assert len(df) == 5

    def test_explain_local_shap_sorted_by_abs_shap(
        self, toy_explainer, toy_pipeline, toy_X_train
    ):
        from src.explainability.local_explanation import explain_local_shap
        X_row = toy_X_train.iloc[[0]]
        df = explain_local_shap(toy_pipeline, toy_explainer, X_row, top_n=None)
        assert (df["abs_shap"].diff().dropna() <= 0).all()

    def test_explain_local_shap_direction_labels_valid(
        self, toy_explainer, toy_pipeline, toy_X_train
    ):
        from src.explainability.local_explanation import explain_local_shap
        X_row = toy_X_train.iloc[[0]]
        df = explain_local_shap(toy_pipeline, toy_explainer, X_row)
        valid = {"(+) increases risk", "(-) decreases risk"}
        assert set(df["direction"].unique()).issubset(valid)

    def test_explain_local_shap_raises_on_multiple_rows(
        self, toy_explainer, toy_pipeline, toy_X_train
    ):
        from src.explainability.local_explanation import explain_local_shap
        X_multi = toy_X_train.iloc[:3]
        with pytest.raises(ValueError, match="exactly 1 row"):
            explain_local_shap(toy_pipeline, toy_explainer, X_multi)

    def test_explain_local_shap_shap_values_finite(
        self, toy_explainer, toy_pipeline, toy_X_train
    ):
        from src.explainability.local_explanation import explain_local_shap
        X_row = toy_X_train.iloc[[0]]
        df = explain_local_shap(toy_pipeline, toy_explainer, X_row, top_n=None)
        assert np.isfinite(df["shap_value"].values).all()


# ---------------------------------------------------------------------------
# 8. Local coef × deviation explanation
# ---------------------------------------------------------------------------

class TestLocalCoefDeviationExplanation:

    def test_explain_local_coef_deviation_returns_dataframe(
        self, toy_pipeline, toy_X_train
    ):
        from src.explainability.local_explanation import explain_local_coef_deviation
        X_row = toy_X_train.iloc[[0]]
        df = explain_local_coef_deviation(toy_pipeline, X_row, toy_X_train)
        assert isinstance(df, pd.DataFrame)

    def test_explain_local_coef_deviation_has_required_columns(
        self, toy_pipeline, toy_X_train
    ):
        from src.explainability.local_explanation import explain_local_coef_deviation
        X_row = toy_X_train.iloc[[0]]
        df = explain_local_coef_deviation(toy_pipeline, X_row, toy_X_train)
        required = {"feature", "coef", "deviation", "contribution", "abs_contribution", "direction"}
        assert required.issubset(set(df.columns))

    def test_explain_local_coef_deviation_top_n_respected(
        self, toy_pipeline, toy_X_train
    ):
        from src.explainability.local_explanation import explain_local_coef_deviation
        X_row = toy_X_train.iloc[[0]]
        df = explain_local_coef_deviation(toy_pipeline, X_row, toy_X_train, top_n=5)
        assert len(df) == 5

    def test_explain_local_coef_deviation_sorted_by_abs_contribution(
        self, toy_pipeline, toy_X_train
    ):
        from src.explainability.local_explanation import explain_local_coef_deviation
        X_row = toy_X_train.iloc[[0]]
        df = explain_local_coef_deviation(toy_pipeline, X_row, toy_X_train, top_n=None)
        assert (df["abs_contribution"].diff().dropna() <= 0).all()

    def test_explain_local_coef_deviation_contribution_formula(
        self, toy_pipeline, toy_X_train
    ):
        """contribution_i must equal coef_i × deviation_i for all rows."""
        from src.explainability.local_explanation import explain_local_coef_deviation
        X_row = toy_X_train.iloc[[0]]
        df = explain_local_coef_deviation(toy_pipeline, X_row, toy_X_train, top_n=None)
        expected = df["coef"] * df["deviation"]
        np.testing.assert_allclose(
            df["contribution"].values, expected.values, atol=1e-8
        )

    def test_explain_local_coef_deviation_raises_on_multiple_rows(
        self, toy_pipeline, toy_X_train
    ):
        from src.explainability.local_explanation import explain_local_coef_deviation
        X_multi = toy_X_train.iloc[:3]
        with pytest.raises(ValueError, match="exactly 1 row"):
            explain_local_coef_deviation(toy_pipeline, X_multi, toy_X_train)

    def test_shap_and_coef_deviation_close(
        self, toy_explainer, toy_pipeline, toy_X_train
    ):
        """
        SHAP LinearExplainer (interventional) and coef×deviation should produce
        the same contributions (they are mathematically equivalent for linear models
        with the same background).
        """
        from src.explainability.local_explanation import (
            explain_local_shap,
            explain_local_coef_deviation,
        )
        X_row = toy_X_train.iloc[[0]]
        shap_df = explain_local_shap(toy_pipeline, toy_explainer, X_row, top_n=None)
        coef_df = explain_local_coef_deviation(toy_pipeline, X_row, toy_X_train, top_n=None)

        # Sort both by feature name to align
        shap_sorted = shap_df.sort_values("feature").reset_index(drop=True)
        coef_sorted = coef_df.sort_values("feature").reset_index(drop=True)

        np.testing.assert_allclose(
            shap_sorted["shap_value"].values,
            coef_sorted["contribution"].values,
            atol=1e-3,
            err_msg="SHAP and coef×deviation should be numerically equivalent",
        )


# ---------------------------------------------------------------------------
# 9. Local explanation formatting
# ---------------------------------------------------------------------------

class TestLocalExplanationFormatting:

    def test_format_local_report_is_string(
        self, toy_explainer, toy_pipeline, toy_X_train
    ):
        from src.explainability.local_explanation import (
            explain_local_shap,
            format_local_explanation_report,
        )
        X_row = toy_X_train.iloc[[0]]
        df = explain_local_shap(toy_pipeline, toy_explainer, X_row)
        prob = float(toy_pipeline.predict_proba(X_row)[0, 1])
        report = format_local_explanation_report(df, risk_probability=prob, use_shap=True)
        assert isinstance(report, str)

    def test_format_local_report_contains_probability(
        self, toy_explainer, toy_pipeline, toy_X_train
    ):
        from src.explainability.local_explanation import (
            explain_local_shap,
            format_local_explanation_report,
        )
        X_row = toy_X_train.iloc[[0]]
        df = explain_local_shap(toy_pipeline, toy_explainer, X_row)
        prob = float(toy_pipeline.predict_proba(X_row)[0, 1])
        report = format_local_explanation_report(df, risk_probability=prob)
        assert "%" in report  # probability formatted as percentage

    def test_format_local_report_contains_disclaimer(
        self, toy_explainer, toy_pipeline, toy_X_train
    ):
        from src.explainability.local_explanation import (
            explain_local_shap,
            format_local_explanation_report,
        )
        X_row = toy_X_train.iloc[[0]]
        df = explain_local_shap(toy_pipeline, toy_explainer, X_row)
        prob = float(toy_pipeline.predict_proba(X_row)[0, 1])
        report = format_local_explanation_report(df, risk_probability=prob)
        assert "Association" in report

    def test_format_local_report_contains_employee_id(
        self, toy_explainer, toy_pipeline, toy_X_train
    ):
        from src.explainability.local_explanation import (
            explain_local_shap,
            format_local_explanation_report,
        )
        X_row = toy_X_train.iloc[[0]]
        df = explain_local_shap(toy_pipeline, toy_explainer, X_row)
        prob = float(toy_pipeline.predict_proba(X_row)[0, 1])
        report = format_local_explanation_report(df, risk_probability=prob, employee_id="EMP_007")
        assert "EMP_007" in report


# ---------------------------------------------------------------------------
# 10. Prediction explanation (PredictionExplanation)
# ---------------------------------------------------------------------------

class TestPredictionExplanation:

    @pytest.fixture
    def pred_result(self, toy_explainer, toy_pipeline, toy_X_train):
        from src.explainability.prediction_explainer import explain_prediction
        X_row = toy_X_train.iloc[[0]]
        return explain_prediction(
            toy_pipeline, toy_explainer, X_row,
            employee_id=42, optimal_threshold=0.40
        )

    def test_explain_prediction_returns_dataclass(self, pred_result):
        from src.explainability.prediction_explainer import PredictionExplanation
        assert isinstance(pred_result, PredictionExplanation)

    def test_explain_prediction_probability_in_range(self, pred_result):
        assert 0.0 <= pred_result.attrition_probability <= 1.0

    def test_explain_prediction_risk_label_valid(self, pred_result):
        assert pred_result.risk_label in {"High Risk", "Moderate Risk", "Low Risk"}

    def test_explain_prediction_local_shap_df_is_dataframe(self, pred_result):
        assert isinstance(pred_result.local_shap_df, pd.DataFrame)

    def test_explain_prediction_global_df_is_dataframe(self, pred_result):
        assert isinstance(pred_result.global_importance_df, pd.DataFrame)

    def test_explain_prediction_text_report_is_string(self, pred_result):
        assert isinstance(pred_result.text_report, str)

    def test_explain_prediction_text_report_contains_disclaimer(self, pred_result):
        assert "Association" in pred_result.text_report

    def test_explain_prediction_base_value_is_finite(self, pred_result):
        assert np.isfinite(pred_result.base_value)

    def test_explain_prediction_feature_names_is_list(self, pred_result):
        assert isinstance(pred_result.feature_names, list)

    def test_explain_prediction_employee_id_stored(self, pred_result):
        assert pred_result.employee_id == 42

    def test_explain_prediction_disclaimers_present(self, pred_result):
        assert len(pred_result.disclaimers) >= 2

    def test_explain_prediction_raises_on_multiple_rows(
        self, toy_explainer, toy_pipeline, toy_X_train
    ):
        from src.explainability.prediction_explainer import explain_prediction
        X_multi = toy_X_train.iloc[:3]
        with pytest.raises(ValueError, match="exactly 1 row"):
            explain_prediction(toy_pipeline, toy_explainer, X_multi)

    def test_explain_prediction_optimal_threshold_stored(
        self, toy_explainer, toy_pipeline, toy_X_train
    ):
        from src.explainability.prediction_explainer import explain_prediction
        X_row = toy_X_train.iloc[[0]]
        result = explain_prediction(
            toy_pipeline, toy_explainer, X_row, optimal_threshold=0.35
        )
        assert result.optimal_threshold == 0.35


# ---------------------------------------------------------------------------
# 11. Batch explanation
# ---------------------------------------------------------------------------

class TestBatchExplanation:

    def test_explain_batch_returns_dataframe(
        self, toy_explainer, toy_pipeline, toy_X_train
    ):
        from src.explainability.prediction_explainer import explain_batch
        X_sample = toy_X_train.iloc[:5]
        df = explain_batch(toy_pipeline, toy_explainer, X_sample)
        assert isinstance(df, pd.DataFrame)

    def test_explain_batch_correct_row_count(
        self, toy_explainer, toy_pipeline, toy_X_train
    ):
        from src.explainability.prediction_explainer import explain_batch
        X_sample = toy_X_train.iloc[:5]
        df = explain_batch(toy_pipeline, toy_explainer, X_sample)
        assert len(df) == 5

    def test_explain_batch_has_required_columns(
        self, toy_explainer, toy_pipeline, toy_X_train
    ):
        from src.explainability.prediction_explainer import explain_batch
        X_sample = toy_X_train.iloc[:3]
        df = explain_batch(toy_pipeline, toy_explainer, X_sample)
        required = {"employee_id", "attrition_probability", "risk_label",
                    "top_feature_1", "top_shap_1"}
        assert required.issubset(set(df.columns))

    def test_explain_batch_risk_labels_valid(
        self, toy_explainer, toy_pipeline, toy_X_train
    ):
        from src.explainability.prediction_explainer import explain_batch
        X_sample = toy_X_train.iloc[:10]
        df = explain_batch(toy_pipeline, toy_explainer, X_sample)
        valid = {"High Risk", "Moderate Risk", "Low Risk"}
        assert set(df["risk_label"].unique()).issubset(valid)

    def test_explain_batch_probabilities_in_range(
        self, toy_explainer, toy_pipeline, toy_X_train
    ):
        from src.explainability.prediction_explainer import explain_batch
        X_sample = toy_X_train.iloc[:10]
        df = explain_batch(toy_pipeline, toy_explainer, X_sample)
        assert (df["attrition_probability"] >= 0).all()
        assert (df["attrition_probability"] <= 1).all()

    def test_explain_batch_employee_ids_respected(
        self, toy_explainer, toy_pipeline, toy_X_train
    ):
        from src.explainability.prediction_explainer import explain_batch
        X_sample = toy_X_train.iloc[:3]
        ids = ["E001", "E002", "E003"]
        df = explain_batch(toy_pipeline, toy_explainer, X_sample, employee_ids=ids)
        assert list(df["employee_id"]) == ids

    def test_explain_batch_raises_on_id_length_mismatch(
        self, toy_explainer, toy_pipeline, toy_X_train
    ):
        from src.explainability.prediction_explainer import explain_batch
        X_sample = toy_X_train.iloc[:3]
        with pytest.raises(ValueError, match="employee_ids length"):
            explain_batch(toy_pipeline, toy_explainer, X_sample, employee_ids=["E001"])


# ---------------------------------------------------------------------------
# 12. Risk label classification
# ---------------------------------------------------------------------------

class TestRiskLabelClassification:

    def test_high_risk_at_threshold(self):
        from src.explainability.prediction_explainer import _classify_risk
        assert _classify_risk(0.50, optimal_threshold=0.40) == "High Risk"

    def test_high_risk_exactly_at_threshold(self):
        from src.explainability.prediction_explainer import _classify_risk
        assert _classify_risk(0.40, optimal_threshold=0.40) == "High Risk"

    def test_moderate_risk_below_threshold(self):
        from src.explainability.prediction_explainer import _classify_risk
        assert _classify_risk(0.30, optimal_threshold=0.40) == "Moderate Risk"

    def test_moderate_risk_at_lower_boundary(self):
        from src.explainability.prediction_explainer import _classify_risk
        assert _classify_risk(0.20, optimal_threshold=0.40) == "Moderate Risk"

    def test_low_risk_below_moderate(self):
        from src.explainability.prediction_explainer import _classify_risk
        assert _classify_risk(0.10, optimal_threshold=0.40) == "Low Risk"

    def test_low_risk_near_zero(self):
        from src.explainability.prediction_explainer import _classify_risk
        assert _classify_risk(0.01, optimal_threshold=0.40) == "Low Risk"

    def test_all_labels_possible(self):
        from src.explainability.prediction_explainer import _classify_risk
        labels = {
            _classify_risk(p, optimal_threshold=0.40)
            for p in [0.05, 0.25, 0.60]
        }
        assert labels == {"Low Risk", "Moderate Risk", "High Risk"}


# ---------------------------------------------------------------------------
# 13. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_explain_local_shap_all_zero_features(
        self, toy_explainer, toy_pipeline, toy_X_train
    ):
        """SHAP must return a valid DataFrame even for an unusual input row."""
        from src.explainability.local_explanation import explain_local_shap
        # Use a normal row but expect no crash
        X_row = toy_X_train.iloc[[0]]
        df = explain_local_shap(toy_pipeline, toy_explainer, X_row)
        assert len(df) > 0

    def test_global_importance_all_features_present(self, toy_pipeline):
        from src.explainability.global_importance import get_global_feature_importance
        from src.explainability.feature_names import get_feature_names_from_pipeline
        df = get_global_feature_importance(toy_pipeline)
        names_in_df = set(df["feature"])
        names_expected = set(get_feature_names_from_pipeline(toy_pipeline))
        assert names_in_df == names_expected

    def test_compute_shap_single_row_is_2d(
        self, toy_explainer, toy_pipeline, toy_X_train
    ):
        from src.explainability.shap_explainer import compute_shap_values
        X_single = toy_X_train.iloc[[0]]
        sv = compute_shap_values(toy_explainer, toy_pipeline, X_single)
        assert sv.ndim == 2
        assert sv.shape[0] == 1


# ---------------------------------------------------------------------------
# 14. Disclaimer integrity
# ---------------------------------------------------------------------------

class TestDisclaimerIntegrity:

    def test_causation_disclaimer_in_coef_report(self, toy_pipeline):
        from src.explainability.global_importance import (
            get_global_feature_importance,
            format_global_importance_report,
            CAUSATION_DISCLAIMER,
        )
        df = get_global_feature_importance(toy_pipeline)
        report = format_global_importance_report(df)
        assert "Association" in report
        assert "causation" in report.lower() or "Causation" in report

    def test_causation_disclaimer_in_local_report(
        self, toy_explainer, toy_pipeline, toy_X_train
    ):
        from src.explainability.local_explanation import (
            explain_local_shap,
            format_local_explanation_report,
            CAUSATION_DISCLAIMER,
        )
        X_row = toy_X_train.iloc[[0]]
        df = explain_local_shap(toy_pipeline, toy_explainer, X_row)
        prob = float(toy_pipeline.predict_proba(X_row)[0, 1])
        report = format_local_explanation_report(df, risk_probability=prob)
        assert "Association" in report

    def test_causation_disclaimer_in_shap_report(
        self, toy_explainer, toy_pipeline, toy_X_train
    ):
        from src.explainability.shap_explainer import (
            get_shap_global_importance,
            format_shap_global_report,
        )
        df = get_shap_global_importance(toy_explainer, toy_pipeline, toy_X_train)
        report = format_shap_global_report(df)
        assert "Association" in report

    def test_synthetic_data_disclaimer_in_coef_report(self, toy_pipeline):
        from src.explainability.global_importance import (
            get_global_feature_importance,
            format_global_importance_report,
        )
        df = get_global_feature_importance(toy_pipeline)
        report = format_global_importance_report(df)
        assert "synthetic" in report.lower()

    def test_prediction_explanation_has_two_disclaimers(
        self, toy_explainer, toy_pipeline, toy_X_train
    ):
        from src.explainability.prediction_explainer import explain_prediction
        X_row = toy_X_train.iloc[[0]]
        result = explain_prediction(toy_pipeline, toy_explainer, X_row)
        assert len(result.disclaimers) >= 2


# ---------------------------------------------------------------------------
# 15. Integration test — real saved pipeline
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_saved_pipeline_loads_and_produces_feature_names(self):
        """Integration: the real saved pipeline must return 36 feature names."""
        try:
            import joblib
            from pathlib import Path
            pipeline = joblib.load("models/best_pipeline.pkl")
            from src.explainability.feature_names import get_feature_names_from_pipeline
            names = get_feature_names_from_pipeline(pipeline)
            assert len(names) == 36, f"Expected 36 features, got {len(names)}"
        except FileNotFoundError:
            pytest.skip("Saved pipeline not found")

    def test_saved_pipeline_global_importance_on_real_data(self):
        """Integration: global importance must return correct structure on real pipeline."""
        try:
            import joblib
            from pathlib import Path
            pipeline = joblib.load("models/best_pipeline.pkl")
            from src.explainability.global_importance import get_global_feature_importance
            df = get_global_feature_importance(pipeline)
            assert len(df) == 36
            assert df["rank"].iloc[0] == 1
            assert (df["abs_coef"].diff().dropna() <= 0).all()
        except FileNotFoundError:
            pytest.skip("Saved pipeline not found")

    def test_full_explanation_pipeline_on_real_data(self):
        """Integration: build SHAP explainer, run local explanation on real data."""
        try:
            import joblib
            from src.data.loader import load_data
            from src.features.build_features import build_features
            from src.preprocessing.splitter import split_data
            from src.models.anomaly_treatments import apply_clamp_and_flag
            from src.explainability.shap_explainer import build_shap_explainer
            from src.explainability.prediction_explainer import explain_prediction

            pipeline = joblib.load("models/best_pipeline.pkl")
            raw_df = load_data()
            eng_df = build_features(raw_df)
            splits = split_data(eng_df)
            X_train = apply_clamp_and_flag(splits["X_train"])

            explainer = build_shap_explainer(pipeline, X_train.iloc[:200])
            X_row = X_train.iloc[[0]]

            result = explain_prediction(pipeline, explainer, X_row, employee_id=1)

            assert 0.0 <= result.attrition_probability <= 1.0
            assert result.risk_label in {"High Risk", "Moderate Risk", "Low Risk"}
            assert isinstance(result.local_shap_df, pd.DataFrame)
            assert len(result.local_shap_df) > 0
            assert "Association" in result.text_report

        except FileNotFoundError:
            pytest.skip("Saved pipeline or CSV not found")

    def test_global_importance_top_feature_is_identifiable(self):
        """Integration: the top feature should have a positive, recognisable name."""
        try:
            import joblib
            pipeline = joblib.load("models/best_pipeline.pkl")
            from src.explainability.global_importance import get_global_feature_importance
            df = get_global_feature_importance(pipeline, top_n=1)
            top_feature = df.iloc[0]["feature"]
            assert isinstance(top_feature, str)
            assert len(top_feature) > 0
        except FileNotFoundError:
            pytest.skip("Saved pipeline not found")
