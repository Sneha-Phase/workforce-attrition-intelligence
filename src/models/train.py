"""
src/models/train.py
====================
Phase 3 Training Orchestration for the Workforce Attrition Intelligence project.

Execution plan
--------------
1.  Load raw data → apply feature engineering → stratified split.
2.  Keep the test set completely isolated until Step 9.
3.  For each anomaly treatment (flag_only, clamp_and_flag, exclude):
      a.  Apply treatment to X_train / y_train only.
      b.  Run CV for all model × imbalance-strategy combinations.
      c.  Store results.
4.  Evaluate SMOTE only if the best balanced-weight model is within 1% PR-AUC
    of the best overall — i.e., only when there is reason to test it.
5.  Run leakage sensitivity analysis (primary vs. sensitivity feature set)
    using the best model family identified in Step 3.
6.  Select the winning configuration:
      - Best (treatment, model, imbalance_strategy) by mean CV PR-AUC.
      - If two configurations are within 1% PR-AUC, prefer the simpler model.
7.  Perform threshold analysis on OOF probabilities from the winner.
8.  Refit the selected pipeline on the full training set.
9.  Evaluate the refitted pipeline ONCE on the held-out test set.
10. Save all artifacts.

Reproducibility
---------------
RANDOM_STATE=42 is used everywhere.  Do NOT change it.

Usage
-----
    python -m src.models.train

Artifacts written
-----------------
    models/best_pipeline.pkl
    models/cv_results.json
    models/model_comparison.csv
    models/final_test_results.json
    models/threshold_analysis.csv
    models/sensitivity_results.csv
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, RANDOM_STATE
from src.data.loader import load_data
from src.features.build_features import build_features
from src.models.anomaly_treatments import (
    ALL_TREATMENTS,
    TREATMENT_CLAMP_FLAG,
    TREATMENT_EXCLUDE,
    TREATMENT_FLAG_ONLY,
    apply_treatment,
)
from src.models.definitions import get_model_catalogue
from src.models.evaluation import (
    CV_FOLDS,
    compute_metrics,
    cross_validate_pipeline,
    cross_validate_with_smote,
    evaluate_on_test,
    run_cv_experiment,
)
from src.models.sensitivity import (
    FEATURE_SET_PRIMARY,
    FEATURE_SET_SENSITIVITY,
    build_sensitivity_pipeline,
    drop_sensitivity_features,
)
from src.models.threshold_analysis import (
    build_threshold_table,
    collect_oof_probabilities,
    select_optimal_threshold,
)
from src.preprocessing.pipeline import build_preprocessor
from src.preprocessing.splitter import split_data


# ---------------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------------

PIPELINE_PATH          = MODELS_DIR / "best_pipeline.pkl"
CV_RESULTS_PATH        = MODELS_DIR / "cv_results.json"
MODEL_COMPARE_PATH     = MODELS_DIR / "model_comparison.csv"
FINAL_TEST_RESULTS_PATH = MODELS_DIR / "final_test_results.json"
THRESHOLD_ANALYSIS_PATH = MODELS_DIR / "threshold_analysis.csv"
SENSITIVITY_RESULTS_PATH = MODELS_DIR / "sensitivity_results.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ndarray_to_list(obj):
    """Recursively convert ndarray/int64/float64 to JSON-serialisable types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _ndarray_to_list(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_ndarray_to_list(v) for v in obj]
    return obj


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_ndarray_to_list(data), f, indent=2)
    print(f"  Saved -> {path}")


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Saved -> {path}")


# ---------------------------------------------------------------------------
# Selection logic
# ---------------------------------------------------------------------------

def _select_winner(results_df: pd.DataFrame) -> dict:
    """
    Select the best configuration.

    Rules
    -----
    1.  Rank by mean_pr_auc descending.
    2.  If two configurations are within 1% PR-AUC of the top, prefer the
        simpler model (LR > RF > GB > XGB).

    Returns
    -------
    dict — the winning row as a dict.
    """
    df = results_df.copy().sort_values("mean_pr_auc", ascending=False).reset_index(drop=True)
    top_pr_auc = df.loc[0, "mean_pr_auc"]

    # Simplicity order (lower = simpler)
    simplicity = {"LR": 1, "LR_balanced": 1, "RF": 2, "RF_balanced": 2,
                  "GB": 3, "XGB": 4, "XGB_balanced": 4}

    # Candidates within 1% of top
    margin = 0.01
    candidates = df[df["mean_pr_auc"] >= top_pr_auc - margin].copy()
    candidates["simplicity"] = candidates["model"].map(
        lambda m: simplicity.get(m.split("_SMOTE")[0], 99)
    )
    candidates = candidates.sort_values(
        ["simplicity", "mean_pr_auc"], ascending=[True, False]
    )
    winner_row = candidates.iloc[0].to_dict()
    return winner_row


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_training() -> None:
    """
    Full Phase 3 training orchestration.
    """
    warnings.filterwarnings("ignore")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Data preparation
    # -----------------------------------------------------------------------
    print("\n=== Phase 3 — Data Preparation ===")
    raw_df    = load_data()
    eng_df    = build_features(raw_df)
    splits    = split_data(eng_df)

    X_train   = splits["X_train"]
    X_test    = splits["X_test"]
    y_train   = splits["y_train"]
    y_test    = splits["y_test"]

    print(f"  Train: {X_train.shape[0]} rows | "
          f"Attrition rate: {y_train.mean():.3f}")
    print(f"  Test:  {X_test.shape[0]} rows  | "
          f"Attrition rate: {y_test.mean():.3f} [ISOLATED]")

    # -----------------------------------------------------------------------
    # 2. Anomaly treatment comparison via CV
    # -----------------------------------------------------------------------
    print("\n=== Phase 3 — Anomaly Treatment × Model CV ===")

    all_cv_rows: list[pd.DataFrame] = []

    for treatment in ALL_TREATMENTS:
        print(f"\n--- Treatment: {treatment} ---")
        X_treated, y_treated = apply_treatment(treatment, X_train, y_train)

        # For clamp_and_flag, we need a pipeline that knows about the extra column.
        # The ColumnTransformer's remainder='drop' will ignore it, so we need
        # to rebuild the preprocessor to include the clamped column.
        if treatment == TREATMENT_CLAMP_FLAG:
            catalogue = _get_clamp_catalogue()
        else:
            catalogue = get_model_catalogue()

        n_excluded = len(X_train) - len(X_treated)
        if n_excluded > 0:
            print(f"  [exclude] Dropped {n_excluded} anomalous rows "
                  f"({n_excluded / len(X_train) * 100:.1f}%)")

        cv_df = run_cv_experiment(
            model_catalogue=catalogue,
            X_train=X_treated,
            y_train=y_treated,
            treatment_name=treatment,
            cv_folds=CV_FOLDS,
            random_state=RANDOM_STATE,
        )
        all_cv_rows.append(cv_df)

    cv_results_df = pd.concat(all_cv_rows, ignore_index=True)
    _save_csv(cv_results_df, MODEL_COMPARE_PATH)

    # -----------------------------------------------------------------------
    # 3. Evaluate SMOTE for the best non-SMOTE model on flag_only treatment
    # -----------------------------------------------------------------------
    print("\n=== Phase 3 — SMOTE Evaluation ===")
    flag_only_results = cv_results_df[cv_results_df["treatment"] == TREATMENT_FLAG_ONLY]
    best_non_smote = flag_only_results.sort_values("mean_pr_auc", ascending=False).iloc[0]
    best_balanced  = flag_only_results[
        flag_only_results["imbalance_strategy"] == "class_weight=balanced"
    ].sort_values("mean_pr_auc", ascending=False).iloc[0]

    print(f"  Best non-SMOTE: {best_non_smote['model']} "
          f"(PR-AUC={best_non_smote['mean_pr_auc']:.4f})")
    print(f"  Best balanced:  {best_balanced['model']} "
          f"(PR-AUC={best_balanced['mean_pr_auc']:.4f})")

    top_pr_auc = best_non_smote["mean_pr_auc"]

    # Evaluate SMOTE on the best candidate (usually the best ensemble)
    smote_candidate = best_non_smote["model"].replace("_balanced", "").replace("_SMOTE", "")
    smote_rows: list[pd.DataFrame] = []

    print(f"  Running SMOTE for candidate: {smote_candidate} ...")
    try:
        smote_pipeline = get_model_catalogue()[smote_candidate]
        smote_cv = cross_validate_with_smote(
            smote_pipeline, X_train, y_train,
            cv_folds=CV_FOLDS, random_state=RANDOM_STATE,
        )
        smote_row = {
            "model":              f"{smote_candidate}_SMOTE",
            "treatment":          TREATMENT_FLAG_ONLY,
            "imbalance_strategy": "SMOTE",
            "mean_pr_auc":        smote_cv["mean_pr_auc"],
            "std_pr_auc":         smote_cv["std_pr_auc"],
            "mean_roc_auc":       smote_cv["mean_roc_auc"],
            "std_roc_auc":        smote_cv["std_roc_auc"],
            "mean_f1":            smote_cv["mean_f1"],
            "std_f1":             smote_cv["std_f1"],
            "mean_precision":     smote_cv["mean_precision"],
            "std_precision":      smote_cv["std_precision"],
            "mean_recall":        smote_cv["mean_recall"],
            "std_recall":         smote_cv["std_recall"],
        }
        print(f"    SMOTE PR-AUC = {smote_cv['mean_pr_auc']:.4f}")
        smote_rows.append(pd.DataFrame([smote_row]))
    except Exception as exc:
        print(f"  SMOTE failed: {exc}")

    if smote_rows:
        cv_results_df = pd.concat([cv_results_df] + smote_rows, ignore_index=True)
        _save_csv(cv_results_df, MODEL_COMPARE_PATH)

    # -----------------------------------------------------------------------
    # 4. Leakage sensitivity analysis
    # -----------------------------------------------------------------------
    print("\n=== Phase 3 — Leakage Sensitivity Analysis ===")
    # Use best model family on flag_only; run primary vs sensitivity feature sets

    # Rebuild best model pipeline from catalogue
    best_model_name = best_non_smote["model"]
    print(f"  Sensitivity analysis using model: {best_model_name}")

    sensitivity_rows = []

    # Primary feature set (standard pipeline)
    primary_pipeline = get_model_catalogue()[best_model_name]
    primary_cv = cross_validate_pipeline(
        primary_pipeline, X_train, y_train,
        cv_folds=CV_FOLDS, random_state=RANDOM_STATE,
    )
    sensitivity_rows.append({
        "feature_set":    FEATURE_SET_PRIMARY,
        "model":          best_model_name,
        "mean_pr_auc":    primary_cv["mean_pr_auc"],
        "std_pr_auc":     primary_cv["std_pr_auc"],
        "mean_roc_auc":   primary_cv["mean_roc_auc"],
        "mean_f1":        primary_cv["mean_f1"],
        "mean_recall":    primary_cv["mean_recall"],
        "mean_precision": primary_cv["mean_precision"],
    })
    print(f"  Primary   PR-AUC = {primary_cv['mean_pr_auc']:.4f} "
          f"(+/-{primary_cv['std_pr_auc']:.4f})")

    # Sensitivity feature set (excluded columns dropped from X before pipeline)
    # We build a pipeline that uses the sensitivity preprocessor
    from src.models.definitions import (
        make_logistic_regression,
        make_random_forest,
        make_gradient_boosting,
        make_xgboost,
    )
    _clf_factory = {
        "LR": lambda: make_logistic_regression(),
        "LR_balanced": lambda: make_logistic_regression(class_weight="balanced"),
        "RF": lambda: make_random_forest(),
        "RF_balanced": lambda: make_random_forest(class_weight="balanced"),
        "GB": lambda: make_gradient_boosting(),
        "XGB": lambda: make_xgboost(),
        "XGB_balanced": lambda: make_xgboost(scale_pos_weight=4.0),
    }
    clf_factory_fn = _clf_factory.get(best_model_name)
    if clf_factory_fn is not None:
        sensitivity_pipeline = build_sensitivity_pipeline(clf_factory_fn())
        X_train_sens = drop_sensitivity_features(X_train)
        sensitivity_cv = cross_validate_pipeline(
            sensitivity_pipeline, X_train_sens, y_train,
            cv_folds=CV_FOLDS, random_state=RANDOM_STATE,
        )
        sensitivity_rows.append({
            "feature_set":    FEATURE_SET_SENSITIVITY,
            "model":          best_model_name,
            "mean_pr_auc":    sensitivity_cv["mean_pr_auc"],
            "std_pr_auc":     sensitivity_cv["std_pr_auc"],
            "mean_roc_auc":   sensitivity_cv["mean_roc_auc"],
            "mean_f1":        sensitivity_cv["mean_f1"],
            "mean_recall":    sensitivity_cv["mean_recall"],
            "mean_precision": sensitivity_cv["mean_precision"],
        })
        pr_delta = primary_cv["mean_pr_auc"] - sensitivity_cv["mean_pr_auc"]
        print(f"  Sensitivity PR-AUC = {sensitivity_cv['mean_pr_auc']:.4f} "
              f"(+/-{sensitivity_cv['std_pr_auc']:.4f})")
        print(f"  Delta PR-AUC (primary - sensitivity) = {pr_delta:+.4f}")
        if abs(pr_delta) <= 0.02:
            print("  -> Marginal difference: suspect features add minimal signal.")
        else:
            print("  -> Meaningful difference: suspect features carry real signal.")

    sensitivity_df = pd.DataFrame(sensitivity_rows)
    _save_csv(sensitivity_df, SENSITIVITY_RESULTS_PATH)

    # -----------------------------------------------------------------------
    # 5. Model selection
    # -----------------------------------------------------------------------
    print("\n=== Phase 3 — Model Selection ===")
    winner = _select_winner(cv_results_df)
    print(f"\n  WINNER: model={winner['model']} | "
          f"treatment={winner['treatment']} | "
          f"imbalance_strategy={winner['imbalance_strategy']}")
    print(f"  CV PR-AUC: {winner['mean_pr_auc']:.4f} (+/-{winner['std_pr_auc']:.4f})")
    print(f"  CV ROC-AUC: {winner['mean_roc_auc']:.4f}")

    # -----------------------------------------------------------------------
    # 6. Threshold selection via OOF probabilities
    # -----------------------------------------------------------------------
    print("\n=== Phase 3 — Threshold Selection ===")

    # Rebuild the winning pipeline fresh for OOF collection
    winning_model_name = winner["model"].replace("_SMOTE", "")
    winning_treatment  = winner["treatment"]

    if winning_treatment == TREATMENT_CLAMP_FLAG:
        winning_catalogue = _get_clamp_catalogue()
    else:
        winning_catalogue = get_model_catalogue()

    if winning_model_name in winning_catalogue:
        oof_pipeline = winning_catalogue[winning_model_name]
    else:
        oof_pipeline = get_model_catalogue()[winning_model_name]

    # Apply winning treatment for OOF collection
    X_winner, y_winner = apply_treatment(winning_treatment, X_train, y_train)

    print("  Collecting OOF probabilities ...")
    oof_y_true, oof_y_prob = collect_oof_probabilities(
        oof_pipeline, X_winner, y_winner,
        cv_folds=CV_FOLDS, random_state=RANDOM_STATE,
    )

    threshold_result = select_optimal_threshold(
        oof_y_true, oof_y_prob, optimise_for="f1"
    )
    optimal_threshold = threshold_result["optimal_threshold"]
    print(f"  Optimal threshold (F1 on OOF): {optimal_threshold:.4f}")
    print(f"  At optimal: precision={threshold_result['precision']:.3f} "
          f"recall={threshold_result['recall']:.3f} "
          f"F1={threshold_result['f1']:.3f}")

    threshold_table = build_threshold_table(oof_y_true, oof_y_prob)
    _save_csv(threshold_table, THRESHOLD_ANALYSIS_PATH)

    # -----------------------------------------------------------------------
    # 7. Refit on full training set
    # -----------------------------------------------------------------------
    print("\n=== Phase 3 — Refit on Full Training Set ===")

    if winning_treatment == TREATMENT_CLAMP_FLAG:
        final_catalogue = _get_clamp_catalogue()
    else:
        final_catalogue = get_model_catalogue()

    if winning_model_name in final_catalogue:
        final_pipeline = final_catalogue[winning_model_name]
    else:
        final_pipeline = get_model_catalogue()[winning_model_name]

    if "_SMOTE" in winner["model"]:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        steps = list(final_pipeline.steps)
        final_pipeline = ImbPipeline(steps=[
            steps[0],
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            steps[-1],
        ])

    print(f"  Fitting {winner['model']} on {len(X_winner)} training rows ...")
    final_pipeline.fit(X_winner, y_winner)
    print("  Fit complete.")

    # -----------------------------------------------------------------------
    # 8. Final test evaluation (ONCE — held-out set)
    # -----------------------------------------------------------------------
    print("\n=== Phase 3 — Final Test Evaluation (held-out) ===")
    print("  *** Using held-out test set for the FIRST AND ONLY TIME ***")

    # For clamp_and_flag, the test set also needs the clamped column so the
    # fitted pipeline's ColumnTransformer can find it.  This is NOT leakage:
    # apply_clamp_and_flag only applies min(role, company) — a deterministic
    # transformation of existing columns, not fitted on train.
    if winning_treatment == TREATMENT_CLAMP_FLAG:
        from src.models.anomaly_treatments import apply_clamp_and_flag
        X_test_eval = apply_clamp_and_flag(X_test)
    else:
        X_test_eval = X_test

    test_results = evaluate_on_test(
        final_pipeline, X_test_eval, y_test,
        optimal_threshold=optimal_threshold,
    )

    print(f"\n  Test PR-AUC:  {test_results['pr_auc']:.4f}")
    print(f"  Test ROC-AUC: {test_results['roc_auc']:.4f}")
    print(f"  Test F1 (@0.50): {test_results['f1']:.4f}")
    print(f"  Test F1 (@{optimal_threshold:.2f}): "
          f"{test_results['optimal_threshold_metrics']['f1']:.4f}")
    print(f"\n  Confusion matrix (@0.50 threshold):\n"
          f"  {test_results['confusion_matrix']}")
    print(f"\n  Classification report (@0.50 threshold):\n"
          f"  {test_results['classification_report']}")

    # -----------------------------------------------------------------------
    # 9. Save artifacts
    # -----------------------------------------------------------------------
    print("\n=== Phase 3 — Saving Artifacts ===")

    joblib.dump(final_pipeline, PIPELINE_PATH)
    print(f"  Saved -> {PIPELINE_PATH}")

    cv_json = {
        "winner": winner,
        "optimal_threshold": optimal_threshold,
        "threshold_selection": {k: v for k, v in threshold_result.items()
                                if k not in ("tp", "fp", "fn", "tn")},
        "all_cv_results": cv_results_df.to_dict(orient="records"),
    }
    _save_json(cv_json, CV_RESULTS_PATH)

    # Prepare serialisable test results
    test_results_serialisable = {
        k: v for k, v in test_results.items()
        if k not in ("y_prob", "threshold_analysis")
    }
    _save_json(test_results_serialisable, FINAL_TEST_RESULTS_PATH)

    print("\n=== Phase 3 Complete ===")
    print(f"\nSelected configuration:")
    print(f"  Model:               {winner['model']}")
    print(f"  Anomaly treatment:   {winner['treatment']}")
    print(f"  Imbalance strategy:  {winner['imbalance_strategy']}")
    print(f"  CV PR-AUC:           {winner['mean_pr_auc']:.4f} (+/-{winner['std_pr_auc']:.4f})")
    print(f"  Optimal threshold:   {optimal_threshold:.4f}")
    print(f"\nFinal test metrics:")
    print(f"  PR-AUC:   {test_results['pr_auc']:.4f}")
    print(f"  ROC-AUC:  {test_results['roc_auc']:.4f}")
    print(f"  F1 (@0.50): {test_results['f1']:.4f}")
    print(f"  F1 (@{optimal_threshold:.2f}):  "
          f"{test_results['optimal_threshold_metrics']['f1']:.4f}")
    print(f"  Precision (@{optimal_threshold:.2f}):  "
          f"{test_results['optimal_threshold_metrics']['precision']:.4f}")
    print(f"  Recall    (@{optimal_threshold:.2f}):  "
          f"{test_results['optimal_threshold_metrics']['recall']:.4f}")


# ---------------------------------------------------------------------------
# Clamp-and-flag pipeline variant
# ---------------------------------------------------------------------------

def _get_clamp_catalogue() -> dict:
    """
    Return model catalogue using a preprocessor that includes the clamped column.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from src.config import (
        CATEGORICAL_BINARY,
        CATEGORICAL_NOMINAL,
        ORDINAL_FEATURES,
    )
    from src.features.build_features import COL_ROLE_TENURE_ANOMALY
    from src.preprocessing.pipeline import ENGINEERED_CONTINUOUS, RAW_CONTINUOUS
    from src.models.anomaly_treatments import COL_ROLE_CLAMPED
    from src.models.definitions import (
        make_logistic_regression,
        make_random_forest,
        make_gradient_boosting,
        make_xgboost,
    )

    ordinal_pass = ORDINAL_FEATURES + [COL_ROLE_TENURE_ANOMALY]
    # Add clamped column to continuous
    continuous = RAW_CONTINUOUS + ENGINEERED_CONTINUOUS + [COL_ROLE_CLAMPED]

    binary_categories = [["Female", "Male"], ["No", "Yes"]]

    def _build_clamp_preprocessor():
        return ColumnTransformer(
            transformers=[
                ("nominal_ohe",  OneHotEncoder(drop="first", handle_unknown="ignore",
                                               sparse_output=False), CATEGORICAL_NOMINAL),
                ("binary_ord",   OrdinalEncoder(categories=binary_categories,
                                               handle_unknown="use_encoded_value",
                                               unknown_value=-1), CATEGORICAL_BINARY),
                ("ordinal_pass", "passthrough", ordinal_pass),
                ("scale_cont",   StandardScaler(), continuous),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    def _build_clamp_pipeline(clf):
        return Pipeline(steps=[("preprocessor", _build_clamp_preprocessor()), ("clf", clf)])

    _spw = 4.0
    return {
        "LR":           _build_clamp_pipeline(make_logistic_regression()),
        "LR_balanced":  _build_clamp_pipeline(make_logistic_regression(class_weight="balanced")),
        "RF":           _build_clamp_pipeline(make_random_forest()),
        "RF_balanced":  _build_clamp_pipeline(make_random_forest(class_weight="balanced")),
        "GB":           _build_clamp_pipeline(make_gradient_boosting()),
        "XGB":          _build_clamp_pipeline(make_xgboost()),
        "XGB_balanced": _build_clamp_pipeline(make_xgboost(scale_pos_weight=_spw)),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_training()
