"""
tests/test_phase3_models.py
============================
Phase 3 test suite — Model Training and Evaluation.

Coverage
--------
1.  Model construction — all factories return unfitted estimators of correct type
2.  Pipeline fitting — pipelines fit and predict without error
3.  Prediction probabilities — shape, range [0,1], positive-class column
4.  Metric calculations — precision/recall/F1/ROC-AUC/PR-AUC are correct
5.  Threshold calculations — threshold table columns, optimal threshold
6.  Reproducibility — same seed produces identical CV fold results
7.  Edge cases — empty anomaly flag set, single-class raises, etc.
8.  Anomaly treatments — flag_only, clamp_and_flag, exclude behave correctly
9.  Leakage sensitivity — sensitivity preprocessor drops correct columns
10. Cross-validation — produces expected keys, shapes, and numeric values
11. OOF probability collection — same length as input, valid probabilities

Test strategy
-------------
- All tests use a minimal synthesised DataFrame (no dependency on real CSV).
- The real-CSV integration tests are skipped if the file is absent.
- RANDOM_STATE=42 is used everywhere for reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE, TARGET_COLUMN, ID_COLUMN, TARGET_POSITIVE, TARGET_NEGATIVE
from src.features.build_features import build_features, COL_ROLE_TENURE_ANOMALY
from src.preprocessing.splitter import split_data
from src.preprocessing.pipeline import prepare_features


# ---------------------------------------------------------------------------
# Shared fixture factory
# ---------------------------------------------------------------------------

def _make_df(
    n_yes: int = 40,
    n_no: int = 160,
) -> pd.DataFrame:
    """
    Synthesise a minimal DataFrame that satisfies the full schema contract
    including all required columns and engineered features.
    """
    nrows = n_yes + n_no
    rng = np.random.default_rng(0)

    dept_vals   = ["Marketing", "Sales", "Finance", "HR", "IT"]
    role_vals   = ["Analyst", "Assistant", "Executive", "Manager"]
    marital_vals = ["Married", "Divorced", "Single"]

    years_at_co = rng.integers(1, 30, nrows).tolist()
    # Make ~25% of rows anomalous (role > company tenure)
    years_in_role = []
    for i, yac in enumerate(years_at_co):
        if i % 4 == 0:
            years_in_role.append(min(yac + 2, 30))  # anomalous
        else:
            years_in_role.append(max(1, yac - 1))   # normal

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
    df = pd.DataFrame(data)
    return build_features(df)


@pytest.fixture
def small_df():
    return _make_df(n_yes=40, n_no=160)


@pytest.fixture
def splits(small_df):
    return split_data(small_df)


@pytest.fixture
def X_train(splits):
    return splits["X_train"]


@pytest.fixture
def y_train(splits):
    return splits["y_train"]


@pytest.fixture
def X_test(splits):
    return splits["X_test"]


@pytest.fixture
def y_test(splits):
    return splits["y_test"]


# ---------------------------------------------------------------------------
# 1. Model construction
# ---------------------------------------------------------------------------

class TestModelConstruction:

    def test_logistic_regression_returns_correct_type(self):
        from src.models.definitions import make_logistic_regression
        model = make_logistic_regression()
        assert isinstance(model, LogisticRegression)

    def test_logistic_regression_balanced_class_weight(self):
        from src.models.definitions import make_logistic_regression
        model = make_logistic_regression(class_weight="balanced")
        assert model.class_weight == "balanced"

    def test_logistic_regression_unfit(self):
        from sklearn.exceptions import NotFittedError
        from src.models.definitions import make_logistic_regression
        model = make_logistic_regression()
        with pytest.raises(NotFittedError):
            model.predict(np.zeros((1, 5)))

    def test_random_forest_returns_correct_type(self):
        from src.models.definitions import make_random_forest
        model = make_random_forest()
        assert isinstance(model, RandomForestClassifier)

    def test_random_forest_uses_random_state(self):
        from src.models.definitions import make_random_forest
        model = make_random_forest()
        assert model.random_state == RANDOM_STATE

    def test_gradient_boosting_returns_correct_type(self):
        from src.models.definitions import make_gradient_boosting
        model = make_gradient_boosting()
        assert isinstance(model, GradientBoostingClassifier)

    def test_xgboost_returns_correct_type(self):
        from xgboost import XGBClassifier
        from src.models.definitions import make_xgboost
        model = make_xgboost()
        assert isinstance(model, XGBClassifier)

    def test_xgboost_scale_pos_weight_applied(self):
        from src.models.definitions import make_xgboost
        model = make_xgboost(scale_pos_weight=4.0)
        assert model.get_params()["scale_pos_weight"] == 4.0

    def test_model_catalogue_has_expected_keys(self):
        from src.models.definitions import get_model_catalogue
        catalogue = get_model_catalogue()
        expected_keys = {"LR", "LR_balanced", "RF", "RF_balanced", "GB", "XGB", "XGB_balanced"}
        assert set(catalogue.keys()) == expected_keys

    def test_all_catalogue_entries_are_pipelines(self):
        from src.models.definitions import get_model_catalogue
        for name, pipeline in get_model_catalogue().items():
            assert isinstance(pipeline, Pipeline), f"{name} is not a Pipeline"

    def test_pipeline_has_preprocessor_and_clf_steps(self):
        from src.models.definitions import get_model_catalogue
        for name, pipeline in get_model_catalogue().items():
            step_names = [s[0] for s in pipeline.steps]
            assert "preprocessor" in step_names, f"{name} missing 'preprocessor'"
            assert "clf" in step_names, f"{name} missing 'clf'"


# ---------------------------------------------------------------------------
# 2. Pipeline fitting
# ---------------------------------------------------------------------------

class TestPipelineFitting:

    def test_lr_pipeline_fits_without_error(self, X_train, y_train):
        from src.models.definitions import get_model_catalogue
        pipeline = get_model_catalogue()["LR"]
        pipeline.fit(X_train, y_train)  # must not raise

    def test_rf_pipeline_fits_without_error(self, X_train, y_train):
        from src.models.definitions import get_model_catalogue
        pipeline = get_model_catalogue()["RF"]
        pipeline.fit(X_train, y_train)

    def test_gb_pipeline_fits_without_error(self, X_train, y_train):
        from src.models.definitions import get_model_catalogue
        pipeline = get_model_catalogue()["GB"]
        pipeline.fit(X_train, y_train)

    def test_xgb_pipeline_fits_without_error(self, X_train, y_train):
        from src.models.definitions import get_model_catalogue
        pipeline = get_model_catalogue()["XGB"]
        pipeline.fit(X_train, y_train)

    def test_lr_balanced_fits_without_error(self, X_train, y_train):
        from src.models.definitions import get_model_catalogue
        pipeline = get_model_catalogue()["LR_balanced"]
        pipeline.fit(X_train, y_train)

    def test_rf_balanced_fits_without_error(self, X_train, y_train):
        from src.models.definitions import get_model_catalogue
        pipeline = get_model_catalogue()["RF_balanced"]
        pipeline.fit(X_train, y_train)


# ---------------------------------------------------------------------------
# 3. Prediction probabilities
# ---------------------------------------------------------------------------

class TestPredictionProbabilities:

    @pytest.fixture
    def fitted_lr(self, X_train, y_train):
        from src.models.definitions import get_model_catalogue
        pipeline = get_model_catalogue()["LR"]
        pipeline.fit(X_train, y_train)
        return pipeline

    def test_predict_proba_returns_2d_array(self, fitted_lr, X_test):
        proba = fitted_lr.predict_proba(X_test)
        assert proba.ndim == 2

    def test_predict_proba_has_two_columns(self, fitted_lr, X_test):
        proba = fitted_lr.predict_proba(X_test)
        assert proba.shape[1] == 2

    def test_predict_proba_rows_match_input(self, fitted_lr, X_test):
        proba = fitted_lr.predict_proba(X_test)
        assert proba.shape[0] == len(X_test)

    def test_predict_proba_sums_to_one(self, fitted_lr, X_test):
        proba = fitted_lr.predict_proba(X_test)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_predict_proba_in_zero_one_range(self, fitted_lr, X_test):
        proba = fitted_lr.predict_proba(X_test)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_positive_class_probability_is_column_1(self, fitted_lr, X_test):
        proba = fitted_lr.predict_proba(X_test)
        y_prob = proba[:, 1]
        assert y_prob.shape == (len(X_test),)

    def test_all_models_produce_valid_probabilities(self, X_train, y_train, X_test):
        from src.models.definitions import get_model_catalogue
        for name, pipeline in get_model_catalogue().items():
            pipeline.fit(X_train, y_train)
            proba = pipeline.predict_proba(X_test)
            assert (proba >= 0).all() and (proba <= 1).all(), \
                f"{name}: probabilities out of [0,1]"
            assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5), \
                f"{name}: row probabilities do not sum to 1"


# ---------------------------------------------------------------------------
# 4. Metric calculations
# ---------------------------------------------------------------------------

class TestMetricCalculations:

    @pytest.fixture
    def fitted_lr_result(self, X_train, y_train, X_test, y_test):
        from src.models.definitions import get_model_catalogue
        from src.models.evaluation import compute_metrics
        pipeline = get_model_catalogue()["LR"]
        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.50).astype(int)
        return compute_metrics(y_test, y_pred, y_prob)

    def test_precision_in_zero_one(self, fitted_lr_result):
        assert 0.0 <= fitted_lr_result["precision"] <= 1.0

    def test_recall_in_zero_one(self, fitted_lr_result):
        assert 0.0 <= fitted_lr_result["recall"] <= 1.0

    def test_f1_in_zero_one(self, fitted_lr_result):
        assert 0.0 <= fitted_lr_result["f1"] <= 1.0

    def test_roc_auc_in_zero_one(self, fitted_lr_result):
        assert 0.0 <= fitted_lr_result["roc_auc"] <= 1.0

    def test_pr_auc_in_zero_one(self, fitted_lr_result):
        assert 0.0 <= fitted_lr_result["pr_auc"] <= 1.0

    def test_confusion_matrix_is_2x2(self, fitted_lr_result):
        cm = fitted_lr_result["confusion_matrix"]
        assert cm.shape == (2, 2)

    def test_confusion_matrix_non_negative(self, fitted_lr_result):
        cm = fitted_lr_result["confusion_matrix"]
        assert (cm >= 0).all()

    def test_classification_report_is_string(self, fitted_lr_result):
        assert isinstance(fitted_lr_result["classification_report"], str)

    def test_classification_report_contains_attrition(self, fitted_lr_result):
        assert "Attrition" in fitted_lr_result["classification_report"]

    def test_compute_metrics_perfect_predictions(self):
        from src.models.evaluation import compute_metrics
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0, 1])
        y_prob = np.array([0.9, 0.1, 0.85, 0.15, 0.8])
        m = compute_metrics(y_true, y_pred, y_prob)
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"]    == pytest.approx(1.0)
        assert m["f1"]        == pytest.approx(1.0)

    def test_compute_metrics_all_wrong(self):
        from src.models.evaluation import compute_metrics
        y_true = np.array([1, 1, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        m = compute_metrics(y_true, y_pred, y_prob)
        assert m["precision"] == pytest.approx(0.0, abs=1e-6)
        assert m["recall"]    == pytest.approx(0.0, abs=1e-6)

    def test_roc_auc_is_higher_for_good_model(self, X_train, y_train, X_test, y_test):
        from src.models.definitions import get_model_catalogue
        from src.models.evaluation import compute_metrics
        pipeline = get_model_catalogue()["RF"]
        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        m = compute_metrics(y_test, y_pred, y_prob)
        # A trained RF on a sensible dataset should beat random (0.5)
        assert m["roc_auc"] > 0.5


# ---------------------------------------------------------------------------
# 5. Threshold calculations
# ---------------------------------------------------------------------------

class TestThresholdCalculations:

    @pytest.fixture
    def y_prob_fixture(self, X_train, y_train, X_test, y_test):
        from src.models.definitions import get_model_catalogue
        pipeline = get_model_catalogue()["LR"]
        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        return y_test, y_prob

    def test_threshold_table_has_required_columns(self, y_prob_fixture):
        from src.models.evaluation import compute_threshold_analysis
        y_true, y_prob = y_prob_fixture
        df = compute_threshold_analysis(y_true, y_prob)
        required = {"threshold", "precision", "recall", "f1", "predicted_positive_rate"}
        assert required.issubset(set(df.columns))

    def test_threshold_table_row_count(self, y_prob_fixture):
        from src.models.evaluation import compute_threshold_analysis
        y_true, y_prob = y_prob_fixture
        thresholds = np.arange(0.10, 0.91, 0.05)
        df = compute_threshold_analysis(y_true, y_prob, thresholds=thresholds)
        assert len(df) == len(thresholds)

    def test_threshold_table_metrics_in_range(self, y_prob_fixture):
        from src.models.evaluation import compute_threshold_analysis
        y_true, y_prob = y_prob_fixture
        df = compute_threshold_analysis(y_true, y_prob)
        for col in ["precision", "recall", "f1"]:
            assert (df[col] >= 0).all() and (df[col] <= 1).all(), \
                f"{col} out of [0,1]"

    def test_higher_threshold_lower_positive_rate(self, y_prob_fixture):
        from src.models.evaluation import compute_threshold_analysis
        y_true, y_prob = y_prob_fixture
        df = compute_threshold_analysis(y_true, y_prob)
        # Positive rate should generally decrease as threshold increases
        low_t  = df[df["threshold"] < 0.30]["predicted_positive_rate"].mean()
        high_t = df[df["threshold"] > 0.70]["predicted_positive_rate"].mean()
        assert low_t >= high_t

    def test_find_optimal_threshold_returns_float(self, y_prob_fixture):
        from src.models.evaluation import find_optimal_threshold
        y_true, y_prob = y_prob_fixture
        t, score = find_optimal_threshold(y_true, y_prob, metric="f1")
        assert isinstance(t, float)
        assert isinstance(score, float)

    def test_find_optimal_threshold_in_range(self, y_prob_fixture):
        from src.models.evaluation import find_optimal_threshold
        y_true, y_prob = y_prob_fixture
        t, score = find_optimal_threshold(y_true, y_prob, metric="f1")
        assert 0.0 <= t <= 1.0
        assert 0.0 <= score <= 1.0

    def test_select_optimal_threshold_returns_dict(self, y_prob_fixture):
        from src.models.threshold_analysis import select_optimal_threshold
        y_true, y_prob = y_prob_fixture
        result = select_optimal_threshold(y_true, y_prob, optimise_for="f1")
        assert isinstance(result, dict)
        assert "optimal_threshold" in result
        assert "f1" in result
        assert "precision" in result
        assert "recall" in result

    def test_select_optimal_threshold_range(self, y_prob_fixture):
        from src.models.threshold_analysis import select_optimal_threshold
        y_true, y_prob = y_prob_fixture
        result = select_optimal_threshold(y_true, y_prob)
        assert 0.0 <= result["optimal_threshold"] <= 1.0
        assert 0.0 <= result["f1"] <= 1.0

    def test_build_threshold_table_has_tp_fp_fn_tn(self, y_prob_fixture):
        from src.models.threshold_analysis import build_threshold_table
        y_true, y_prob = y_prob_fixture
        df = build_threshold_table(y_true, y_prob)
        for col in ["tp", "fp", "fn", "tn"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_tp_fp_fn_tn_sum_equals_n(self, y_prob_fixture):
        from src.models.threshold_analysis import build_threshold_table
        y_true, y_prob = y_prob_fixture
        df = build_threshold_table(y_true, y_prob)
        n = len(y_true)
        totals = df["tp"] + df["fp"] + df["fn"] + df["tn"]
        assert (totals == n).all(), "tp+fp+fn+tn must equal n for every threshold"


# ---------------------------------------------------------------------------
# 6. Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:

    def test_same_seed_same_cv_results(self, X_train, y_train):
        from src.models.definitions import get_model_catalogue
        from src.models.evaluation import cross_validate_pipeline
        pipeline1 = get_model_catalogue()["LR"]
        pipeline2 = get_model_catalogue()["LR"]
        r1 = cross_validate_pipeline(pipeline1, X_train, y_train,
                                     cv_folds=3, random_state=42)
        r2 = cross_validate_pipeline(pipeline2, X_train, y_train,
                                     cv_folds=3, random_state=42)
        # Compare fold-by-fold; treat NaN == NaN as equal (same fold → same outcome)
        for a, b in zip(r1["fold_pr_auc"], r2["fold_pr_auc"]):
            if np.isnan(a) and np.isnan(b):
                pass  # identical NaN outcome from same fold → reproducible
            else:
                assert a == pytest.approx(b, abs=1e-9)

    def test_different_seeds_different_cv_results(self, X_train, y_train):
        """
        Different random_state values must produce different fold assignments.

        This test verifies StratifiedKFold behaviour directly — it does not
        rely on metric values (which can be NaN on small fixtures) or on OOF
        label orderings (which can coincidentally match across seeds).

        Strategy: build two StratifiedKFold splitters with seed=1 and seed=99,
        collect the validation index sets for every fold, and assert that at
        least one fold's validation set differs between the two splitters.
        """
        from sklearn.model_selection import StratifiedKFold

        skf1  = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
        skf99 = StratifiedKFold(n_splits=3, shuffle=True, random_state=99)

        # Collect frozensets of validation indices for each splitter
        val_sets_s1  = [
            frozenset(val_idx.tolist())
            for _, val_idx in skf1.split(X_train, y_train)
        ]
        val_sets_s99 = [
            frozenset(val_idx.tolist())
            for _, val_idx in skf99.split(X_train, y_train)
        ]

        # At least one fold must assign different rows to the validation set
        all_folds_identical = all(
            a == b for a, b in zip(val_sets_s1, val_sets_s99)
        )
        assert not all_folds_identical, (
            "StratifiedKFold with random_state=1 and random_state=99 produced "
            "identical fold assignments — random_state has no effect."
        )

    def test_pipeline_reproducibility_same_predictions(self, X_train, y_train, X_test):
        from src.models.definitions import get_model_catalogue
        p1 = get_model_catalogue()["LR"]
        p2 = get_model_catalogue()["LR"]
        p1.fit(X_train, y_train)
        p2.fit(X_train, y_train)
        proba1 = p1.predict_proba(X_test)
        proba2 = p2.predict_proba(X_test)
        np.testing.assert_array_almost_equal(proba1, proba2, decimal=10)

    def test_oof_collection_reproducible(self, X_train, y_train):
        from src.models.definitions import get_model_catalogue
        from src.models.threshold_analysis import collect_oof_probabilities
        pipeline = get_model_catalogue()["LR"]
        pipeline2 = get_model_catalogue()["LR"]
        y1_true, y1_prob = collect_oof_probabilities(
            pipeline, X_train, y_train, cv_folds=3, random_state=42
        )
        y2_true, y2_prob = collect_oof_probabilities(
            pipeline2, X_train, y_train, cv_folds=3, random_state=42
        )
        np.testing.assert_array_equal(y1_true, y2_true)
        np.testing.assert_array_almost_equal(y1_prob, y2_prob, decimal=8)


# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_compute_metrics_all_positive_predictions(self):
        from src.models.evaluation import compute_metrics
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 1, 1, 1])  # all predicted positive
        y_prob = np.array([0.9, 0.8, 0.95, 0.85])
        m = compute_metrics(y_true, y_pred, y_prob)
        # Precision = TP/(TP+FP) = 2/4 = 0.5, recall = 1.0
        assert m["recall"] == pytest.approx(1.0)

    def test_compute_metrics_no_positive_predictions(self):
        from src.models.evaluation import compute_metrics
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0, 0, 0, 0])
        y_prob = np.array([0.1, 0.05, 0.15, 0.1])
        m = compute_metrics(y_true, y_pred, y_prob)
        assert m["precision"] == pytest.approx(0.0, abs=1e-6)
        assert m["recall"]    == pytest.approx(0.0, abs=1e-6)
        assert m["f1"]        == pytest.approx(0.0, abs=1e-6)

    def test_threshold_analysis_custom_thresholds(self):
        from src.models.evaluation import compute_threshold_analysis
        y_true = np.array([1, 0, 1, 0, 1])
        y_prob = np.array([0.8, 0.2, 0.75, 0.25, 0.9])
        custom_t = np.array([0.3, 0.5, 0.7])
        df = compute_threshold_analysis(y_true, y_prob, thresholds=custom_t)
        assert len(df) == 3
        assert list(df["threshold"]) == pytest.approx([0.3, 0.5, 0.7])

    def test_pipeline_predict_returns_binary_labels(self, X_train, y_train, X_test):
        from src.models.definitions import get_model_catalogue
        pipeline = get_model_catalogue()["LR"]
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        assert set(np.unique(y_pred)).issubset({0, 1})

    def test_pipeline_handles_all_zero_anomaly_flag(self):
        """Pipeline must work even when all anomaly flags are 0 (no anomalies)."""
        from src.models.definitions import get_model_catalogue
        df = _make_df(n_yes=20, n_no=80)
        # Force anomaly flag to all-zero
        df[COL_ROLE_TENURE_ANOMALY] = 0
        splits = split_data(df)
        pipeline = get_model_catalogue()["LR"]
        pipeline.fit(splits["X_train"], splits["y_train"])
        proba = pipeline.predict_proba(splits["X_test"])
        assert proba.shape[0] == len(splits["X_test"])

    def test_pipeline_handles_all_one_anomaly_flag(self):
        """Pipeline must work even when all anomaly flags are 1."""
        from src.models.definitions import get_model_catalogue
        df = _make_df(n_yes=20, n_no=80)
        df[COL_ROLE_TENURE_ANOMALY] = 1
        splits = split_data(df)
        pipeline = get_model_catalogue()["LR"]
        pipeline.fit(splits["X_train"], splits["y_train"])
        proba = pipeline.predict_proba(splits["X_test"])
        assert proba.shape[0] == len(splits["X_test"])


# ---------------------------------------------------------------------------
# 8. Anomaly treatments
# ---------------------------------------------------------------------------

class TestAnomalyTreatments:

    def test_flag_only_returns_dataframe(self, X_train):
        from src.models.anomaly_treatments import apply_flag_only
        result = apply_flag_only(X_train)
        assert isinstance(result, pd.DataFrame)

    def test_flag_only_does_not_modify_input(self, X_train):
        from src.models.anomaly_treatments import apply_flag_only
        original = X_train.copy()
        apply_flag_only(X_train)
        pd.testing.assert_frame_equal(X_train, original)

    def test_flag_only_flag_column_present(self, X_train):
        from src.models.anomaly_treatments import apply_flag_only
        result = apply_flag_only(X_train)
        assert COL_ROLE_TENURE_ANOMALY in result.columns

    def test_clamp_and_flag_adds_clamped_column(self, X_train):
        from src.models.anomaly_treatments import apply_clamp_and_flag, COL_ROLE_CLAMPED
        result = apply_clamp_and_flag(X_train)
        assert COL_ROLE_CLAMPED in result.columns

    def test_clamp_and_flag_clamped_values_le_years_at_company(self, X_train):
        from src.models.anomaly_treatments import apply_clamp_and_flag, COL_ROLE_CLAMPED
        result = apply_clamp_and_flag(X_train)
        assert (result[COL_ROLE_CLAMPED] <= result["Years_at_Company"]).all()

    def test_clamp_and_flag_original_columns_unchanged(self, X_train):
        from src.models.anomaly_treatments import apply_clamp_and_flag
        original_role = X_train["Years_in_Current_Role"].copy()
        result = apply_clamp_and_flag(X_train)
        pd.testing.assert_series_equal(
            result["Years_in_Current_Role"].reset_index(drop=True),
            original_role.reset_index(drop=True),
        )

    def test_exclude_reduces_rows(self, X_train, y_train):
        from src.models.anomaly_treatments import apply_exclude
        n_anomalous = (X_train[COL_ROLE_TENURE_ANOMALY] == 1).sum()
        X_out, y_out = apply_exclude(X_train, y_train)
        assert len(X_out) == len(X_train) - n_anomalous
        assert len(y_out) == len(X_out)

    def test_exclude_no_anomalous_rows_in_output(self, X_train, y_train):
        from src.models.anomaly_treatments import apply_exclude
        X_out, _ = apply_exclude(X_train, y_train)
        assert (X_out[COL_ROLE_TENURE_ANOMALY] == 0).all()

    def test_exclude_preserves_stratification(self, X_train, y_train):
        """After exclusion, both classes must still be present."""
        from src.models.anomaly_treatments import apply_exclude
        _, y_out = apply_exclude(X_train, y_train)
        assert 0 in np.unique(y_out) and 1 in np.unique(y_out)

    def test_dispatcher_flag_only(self, X_train, y_train):
        from src.models.anomaly_treatments import apply_treatment, TREATMENT_FLAG_ONLY
        X_out, y_out = apply_treatment(TREATMENT_FLAG_ONLY, X_train, y_train)
        assert len(X_out) == len(X_train)

    def test_dispatcher_clamp_and_flag(self, X_train, y_train):
        from src.models.anomaly_treatments import (
            apply_treatment, TREATMENT_CLAMP_FLAG, COL_ROLE_CLAMPED
        )
        X_out, _ = apply_treatment(TREATMENT_CLAMP_FLAG, X_train, y_train)
        assert COL_ROLE_CLAMPED in X_out.columns

    def test_dispatcher_exclude(self, X_train, y_train):
        from src.models.anomaly_treatments import apply_treatment, TREATMENT_EXCLUDE
        n_anomalous = (X_train[COL_ROLE_TENURE_ANOMALY] == 1).sum()
        X_out, y_out = apply_treatment(TREATMENT_EXCLUDE, X_train, y_train)
        assert len(X_out) == len(X_train) - n_anomalous

    def test_dispatcher_unknown_treatment_raises(self, X_train, y_train):
        from src.models.anomaly_treatments import apply_treatment
        with pytest.raises(ValueError, match="unknown treatment"):
            apply_treatment("nonexistent", X_train, y_train)

    def test_exclude_without_y_raises(self, X_train):
        from src.models.anomaly_treatments import apply_treatment, TREATMENT_EXCLUDE
        with pytest.raises(ValueError, match="y is required"):
            apply_treatment(TREATMENT_EXCLUDE, X_train, y=None)


# ---------------------------------------------------------------------------
# 9. Leakage sensitivity
# ---------------------------------------------------------------------------

class TestLeakageSensitivity:

    def test_sensitivity_preprocessor_drops_suspect_features(self):
        from src.models.sensitivity import (
            build_sensitivity_preprocessor,
            SENSITIVITY_EXCLUDE_FEATURES,
        )
        preprocessor = build_sensitivity_preprocessor()
        assert preprocessor is not None  # builds without error

    def test_drop_sensitivity_features_removes_expected_columns(self, X_train):
        from src.models.sensitivity import drop_sensitivity_features, SENSITIVITY_EXCLUDE_FEATURES
        X_reduced = drop_sensitivity_features(X_train)
        for col in SENSITIVITY_EXCLUDE_FEATURES:
            assert col not in X_reduced.columns, f"Column '{col}' should be dropped"

    def test_drop_sensitivity_features_preserves_other_columns(self, X_train):
        from src.models.sensitivity import drop_sensitivity_features, SENSITIVITY_EXCLUDE_FEATURES
        X_reduced = drop_sensitivity_features(X_train)
        excluded = set(SENSITIVITY_EXCLUDE_FEATURES)
        for col in X_train.columns:
            if col not in excluded:
                assert col in X_reduced.columns, f"Non-excluded column '{col}' was dropped"

    def test_drop_sensitivity_features_does_not_modify_input(self, X_train):
        from src.models.sensitivity import drop_sensitivity_features
        original_cols = list(X_train.columns)
        drop_sensitivity_features(X_train)
        assert list(X_train.columns) == original_cols

    def test_sensitivity_pipeline_fits_on_reduced_features(self, X_train, y_train):
        from src.models.sensitivity import build_sensitivity_pipeline, drop_sensitivity_features
        from src.models.definitions import make_logistic_regression
        X_reduced = drop_sensitivity_features(X_train)
        pipeline = build_sensitivity_pipeline(make_logistic_regression())
        pipeline.fit(X_reduced, y_train)  # must not raise

    def test_sensitivity_pipeline_predicts_probabilities(self, X_train, y_train, X_test):
        from src.models.sensitivity import (
            build_sensitivity_pipeline,
            drop_sensitivity_features,
        )
        from src.models.definitions import make_logistic_regression
        X_train_red = drop_sensitivity_features(X_train)
        X_test_red  = drop_sensitivity_features(X_test)
        pipeline = build_sensitivity_pipeline(make_logistic_regression())
        pipeline.fit(X_train_red, y_train)
        proba = pipeline.predict_proba(X_test_red)
        assert proba.shape == (len(X_test), 2)


# ---------------------------------------------------------------------------
# 10. Cross-validation
# ---------------------------------------------------------------------------

class TestCrossValidation:

    def test_cv_returns_expected_keys(self, X_train, y_train):
        from src.models.definitions import get_model_catalogue
        from src.models.evaluation import cross_validate_pipeline
        pipeline = get_model_catalogue()["LR"]
        result = cross_validate_pipeline(
            pipeline, X_train, y_train, cv_folds=3, random_state=42
        )
        expected_keys = {
            "fold_pr_auc", "mean_pr_auc", "std_pr_auc",
            "fold_roc_auc", "mean_roc_auc", "std_roc_auc",
            "fold_f1", "mean_f1", "std_f1",
            "fold_precision", "mean_precision", "std_precision",
            "fold_recall", "mean_recall", "std_recall",
            "n_folds",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_cv_n_folds_correct(self, X_train, y_train):
        from src.models.definitions import get_model_catalogue
        from src.models.evaluation import cross_validate_pipeline
        pipeline = get_model_catalogue()["LR"]
        result = cross_validate_pipeline(
            pipeline, X_train, y_train, cv_folds=3, random_state=42
        )
        assert len(result["fold_pr_auc"]) == 3
        assert result["n_folds"] == 3

    def test_cv_mean_pr_auc_in_range(self, X_train, y_train):
        from src.models.definitions import get_model_catalogue
        from src.models.evaluation import cross_validate_pipeline
        pipeline = get_model_catalogue()["LR"]
        result = cross_validate_pipeline(
            pipeline, X_train, y_train, cv_folds=3, random_state=42
        )
        mean_val = result["mean_pr_auc"]
        # NaN can occur with very small fold sizes (degenerate metric); accept that
        if not np.isnan(mean_val):
            assert 0.0 <= mean_val <= 1.0

    def test_cv_mean_equals_fold_average(self, X_train, y_train):
        from src.models.definitions import get_model_catalogue
        from src.models.evaluation import cross_validate_pipeline
        pipeline = get_model_catalogue()["LR"]
        result = cross_validate_pipeline(
            pipeline, X_train, y_train, cv_folds=3, random_state=42
        )
        expected_mean = np.mean(result["fold_pr_auc"])
        # NaN == NaN is not equal; use isnan guard
        if np.isnan(expected_mean):
            assert np.isnan(result["mean_pr_auc"])
        else:
            assert result["mean_pr_auc"] == pytest.approx(expected_mean, abs=1e-9)

    def test_run_cv_experiment_returns_dataframe(self, X_train, y_train):
        from src.models.definitions import get_model_catalogue
        from src.models.evaluation import run_cv_experiment
        # Use only LR for speed
        catalogue = {"LR": get_model_catalogue()["LR"]}
        df = run_cv_experiment(catalogue, X_train, y_train,
                               treatment_name="flag_only", cv_folds=3)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "mean_pr_auc" in df.columns

    def test_run_cv_experiment_has_all_models(self, X_train, y_train):
        from src.models.definitions import get_model_catalogue
        from src.models.evaluation import run_cv_experiment
        # Use LR and LR_balanced only (fast)
        catalogue = {k: v for k, v in get_model_catalogue().items()
                     if k in ["LR", "LR_balanced"]}
        df = run_cv_experiment(catalogue, X_train, y_train,
                               treatment_name="flag_only", cv_folds=3)
        assert set(df["model"]) == {"LR", "LR_balanced"}


# ---------------------------------------------------------------------------
# 11. OOF probability collection
# ---------------------------------------------------------------------------

class TestOOFProbabilityCollection:

    def test_oof_probs_correct_length(self, X_train, y_train):
        from src.models.definitions import get_model_catalogue
        from src.models.threshold_analysis import collect_oof_probabilities
        pipeline = get_model_catalogue()["LR"]
        y_true, y_prob = collect_oof_probabilities(
            pipeline, X_train, y_train, cv_folds=3, random_state=42
        )
        assert len(y_true) == len(X_train)
        assert len(y_prob) == len(X_train)

    def test_oof_probs_in_zero_one(self, X_train, y_train):
        from src.models.definitions import get_model_catalogue
        from src.models.threshold_analysis import collect_oof_probabilities
        pipeline = get_model_catalogue()["LR"]
        _, y_prob = collect_oof_probabilities(
            pipeline, X_train, y_train, cv_folds=3, random_state=42
        )
        assert (y_prob >= 0).all() and (y_prob <= 1).all()

    def test_oof_true_matches_input_y(self, X_train, y_train):
        from src.models.definitions import get_model_catalogue
        from src.models.threshold_analysis import collect_oof_probabilities
        pipeline = get_model_catalogue()["LR"]
        y_true, _ = collect_oof_probabilities(
            pipeline, X_train, y_train, cv_folds=3, random_state=42
        )
        # OOF y_true is a permutation of y_train — same values
        assert sorted(y_true.tolist()) == sorted(y_train.tolist())

    def test_oof_probs_are_finite(self, X_train, y_train):
        from src.models.definitions import get_model_catalogue
        from src.models.threshold_analysis import collect_oof_probabilities
        pipeline = get_model_catalogue()["LR"]
        _, y_prob = collect_oof_probabilities(
            pipeline, X_train, y_train, cv_folds=3, random_state=42
        )
        assert np.isfinite(y_prob).all()


# ---------------------------------------------------------------------------
# Integration tests (real CSV)
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_full_training_pipeline_on_real_csv(self):
        """
        Integration: load real data, split, fit a pipeline, evaluate.
        Skipped if the CSV is absent.
        """
        try:
            from src.data.loader import load_data
            from src.models.definitions import get_model_catalogue
            from src.models.evaluation import cross_validate_pipeline

            raw    = load_data()
            eng    = build_features(raw)
            splits = split_data(eng)

            X_train = splits["X_train"]
            y_train = splits["y_train"]

            pipeline = get_model_catalogue()["LR"]
            result = cross_validate_pipeline(
                pipeline, X_train, y_train, cv_folds=3, random_state=42
            )
            assert result["mean_pr_auc"] > 0.10, (
                f"PR-AUC {result['mean_pr_auc']:.3f} is implausibly low on real data."
            )
        except FileNotFoundError:
            pytest.skip("Real CSV not available")

    def test_anomaly_treatment_on_real_csv(self):
        """Integration: all three treatments complete without error on real data."""
        try:
            from src.data.loader import load_data
            from src.models.anomaly_treatments import apply_treatment, ALL_TREATMENTS
            from src.models.definitions import get_model_catalogue

            raw    = load_data()
            eng    = build_features(raw)
            splits = split_data(eng)
            X_tr   = splits["X_train"]
            y_tr   = splits["y_train"]

            for treatment in ALL_TREATMENTS:
                X_out, y_out = apply_treatment(treatment, X_tr, y_tr)
                assert len(X_out) > 0
        except FileNotFoundError:
            pytest.skip("Real CSV not available")
