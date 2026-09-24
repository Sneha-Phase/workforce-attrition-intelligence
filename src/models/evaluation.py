"""
src/models/evaluation.py
=========================
Cross-validation, metric computation, and model comparison utilities.

Design principles
-----------------
- Cross-validation is ALWAYS stratified on the training set only.
- The held-out test set is NEVER touched in any function here except
  ``evaluate_on_test``, which is called exactly once after selection.
- PR-AUC is the primary ranking metric (appropriate for 80/20 imbalance).
- All metric functions operate on already-computed predictions / probabilities
  so they are independently testable without a trained model.
- SMOTE is only evaluated when CV evidence shows measurable improvement over
  class_weight='balanced'; it is never the default.

Metrics computed
----------------
For each CV fold and for the final test evaluation:
    precision, recall, f1, roc_auc, pr_auc, confusion_matrix,
    classification_report, threshold_analysis

Primary ranking metric: PR-AUC
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CV_FOLDS: int = 5
PRIMARY_METRIC: str = "pr_auc"


# ---------------------------------------------------------------------------
# Metric helpers (operate on arrays — no model dependency)
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, Any]:
    """
    Compute the full set of evaluation metrics.

    Parameters
    ----------
    y_true : np.ndarray of int
        Ground-truth binary labels (0 / 1).
    y_pred : np.ndarray of int
        Predicted binary labels at the default 0.50 threshold.
    y_prob : np.ndarray of float
        Predicted probabilities for the positive class.

    Returns
    -------
    dict with keys:
        precision, recall, f1, roc_auc, pr_auc,
        confusion_matrix (2×2 ndarray), classification_report (str)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = {
            "precision":             precision_score(y_true, y_pred, zero_division=0),
            "recall":                recall_score(y_true, y_pred, zero_division=0),
            "f1":                    f1_score(y_true, y_pred, zero_division=0),
            "roc_auc":               roc_auc_score(y_true, y_prob),
            "pr_auc":                average_precision_score(y_true, y_prob),
            "confusion_matrix":      confusion_matrix(y_true, y_pred),
            "classification_report": classification_report(
                y_true, y_pred, target_names=["No Attrition", "Attrition"],
                zero_division=0,
            ),
        }
    return metrics


def compute_threshold_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Evaluate precision / recall / F1 across a range of classification thresholds.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels.
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    thresholds : np.ndarray | None
        Thresholds to evaluate.  Defaults to np.arange(0.10, 0.91, 0.05).

    Returns
    -------
    pd.DataFrame with columns:
        threshold, precision, recall, f1, predicted_positive_rate
    """
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.05)

    rows = []
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rows.append(
                {
                    "threshold":               round(float(t), 4),
                    "precision":               precision_score(y_true, y_pred_t, zero_division=0),
                    "recall":                  recall_score(y_true, y_pred_t, zero_division=0),
                    "f1":                      f1_score(y_true, y_pred_t, zero_division=0),
                    "predicted_positive_rate": float(y_pred_t.mean()),
                }
            )
    return pd.DataFrame(rows)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> tuple[float, float]:
    """
    Find the threshold that maximises the given metric on the provided data.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels.
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    metric : str
        One of 'f1', 'precision', 'recall'.  Default 'f1'.

    Returns
    -------
    tuple[float, float]
        (optimal_threshold, metric_value_at_optimal_threshold)
    """
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns n+1 precision/recall values for n thresholds
    # the last pair is (1.0, 0.0) with no corresponding threshold
    prec_arr = prec_arr[:-1]
    rec_arr  = rec_arr[:-1]

    if metric == "f1":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            denom = prec_arr + rec_arr
            f1_arr = np.where(denom > 0, 2 * prec_arr * rec_arr / denom, 0.0)
        scores = f1_arr
    elif metric == "precision":
        scores = prec_arr
    elif metric == "recall":
        scores = rec_arr
    else:
        raise ValueError(f"find_optimal_threshold: unknown metric '{metric}'")

    if len(scores) == 0:
        return 0.5, 0.0

    best_idx = int(np.argmax(scores))
    return float(thresh_arr[best_idx]), float(scores[best_idx])


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_pipeline(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """
    Run stratified K-fold cross-validation on the training set only.

    Each fold is evaluated manually so that ``predict_proba[:, 1]`` (the
    positive-class probability) is passed explicitly to probability-based
    metrics.  Using ``sklearn.cross_validate`` with ``make_scorer(…,
    needs_proba=True)`` passes the full 2-column matrix to
    ``average_precision_score`` / ``roc_auc_score``, which silently returns
    NaN for binary classification — this manual loop avoids that bug.

    Parameters
    ----------
    pipeline : Pipeline
        Unfitted sklearn Pipeline (preprocessor + classifier).
    X : pd.DataFrame
        Training feature matrix.
    y : np.ndarray
        Training binary labels.
    cv_folds : int
        Number of stratified folds.  Default CV_FOLDS (5).
    random_state : int
        Random state for StratifiedKFold.

    Returns
    -------
    dict with keys:
        fold_pr_auc   : list[float]  — per-fold PR-AUC
        fold_roc_auc  : list[float]  — per-fold ROC-AUC
        fold_f1       : list[float]  — per-fold F1 at 0.50 threshold
        fold_precision: list[float]  — per-fold precision at 0.50 threshold
        fold_recall   : list[float]  — per-fold recall at 0.50 threshold
        mean_pr_auc   : float
        std_pr_auc    : float
        mean_roc_auc  : float
        std_roc_auc   : float
        mean_f1       : float
        std_f1        : float
        mean_precision: float
        std_precision : float
        mean_recall   : float
        std_recall    : float
        n_folds       : int
    """
    import sklearn.base as skbase

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    fold_metrics: dict[str, list[float]] = {
        m: [] for m in ["pr_auc", "roc_auc", "f1", "precision", "recall"]
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for train_idx, val_idx in skf.split(X, y):
            X_tr  = X.iloc[train_idx]
            y_tr  = y[train_idx]
            X_val = X.iloc[val_idx]
            y_val = y[val_idx]

            clone = skbase.clone(pipeline)
            clone.fit(X_tr, y_tr)

            y_prob = clone.predict_proba(X_val)[:, 1]   # positive-class column
            y_pred = (y_prob >= 0.50).astype(int)

            fold_metrics["pr_auc"].append(
                float(average_precision_score(y_val, y_prob))
            )
            fold_metrics["roc_auc"].append(
                float(roc_auc_score(y_val, y_prob))
            )
            fold_metrics["f1"].append(
                float(f1_score(y_val, y_pred, zero_division=0))
            )
            fold_metrics["precision"].append(
                float(precision_score(y_val, y_pred, zero_division=0))
            )
            fold_metrics["recall"].append(
                float(recall_score(y_val, y_pred, zero_division=0))
            )

    result: dict[str, Any] = {}
    for metric, vals in fold_metrics.items():
        result[f"fold_{metric}"]  = vals
        result[f"mean_{metric}"]  = float(np.mean(vals))
        result[f"std_{metric}"]   = float(np.std(vals, ddof=1))

    result["n_folds"] = cv_folds
    return result


# ---------------------------------------------------------------------------
# SMOTE evaluation (conditional)
# ---------------------------------------------------------------------------

def cross_validate_with_smote(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """
    Evaluate a pipeline with SMOTE oversampling inside each CV fold.

    SMOTE is applied only inside the training portion of each fold — never
    to validation data — preserving leakage safety.

    Parameters
    ----------
    pipeline : Pipeline
        Unfitted sklearn Pipeline.
    X : pd.DataFrame
        Training feature matrix.
    y : np.ndarray
        Training binary labels.
    cv_folds : int
        Number of stratified folds.
    random_state : int
        Random state for StratifiedKFold and SMOTE.

    Returns
    -------
    dict
        Same structure as cross_validate_pipeline.
    """
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
    except ImportError:
        raise ImportError(
            "imbalanced-learn is required for SMOTE evaluation. "
            "Install with: pip install imbalanced-learn"
        )

    # Rebuild as imblearn Pipeline inserting SMOTE after preprocessor
    steps = list(pipeline.steps)  # [("preprocessor", ...), ("clf", ...)]
    imb_steps = [
        steps[0],  # preprocessor
        ("smote", SMOTE(random_state=random_state)),
        steps[-1], # classifier
    ]
    imb_pipeline = ImbPipeline(steps=imb_steps)

    return cross_validate_pipeline(
        imb_pipeline, X, y, cv_folds=cv_folds, random_state=random_state
    )


# ---------------------------------------------------------------------------
# Final test evaluation (called exactly once after selection)
# ---------------------------------------------------------------------------

def evaluate_on_test(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    optimal_threshold: float = 0.50,
) -> dict[str, Any]:
    """
    Evaluate a FITTED pipeline on the held-out test set.

    This function must be called exactly once, after model selection is
    complete.  The pipeline must already be fitted on X_train.

    Parameters
    ----------
    pipeline : Pipeline
        A *fitted* sklearn Pipeline.
    X_test : pd.DataFrame
        Held-out test features (never used for fitting or tuning).
    y_test : np.ndarray
        Held-out test labels.
    optimal_threshold : float
        Classification threshold (from threshold analysis on validation data).

    Returns
    -------
    dict with keys:
        all keys from compute_metrics (at default 0.50 threshold),
        optimal_threshold_metrics (dict at the selected threshold),
        threshold_analysis (pd.DataFrame),
        y_prob (np.ndarray)
    """
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred_default = (y_prob >= 0.50).astype(int)
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)

    result = compute_metrics(y_test, y_pred_default, y_prob)
    result["optimal_threshold"] = optimal_threshold
    result["optimal_threshold_metrics"] = compute_metrics(
        y_test, y_pred_optimal, y_prob
    )
    result["threshold_analysis"] = compute_threshold_analysis(y_test, y_prob)
    result["y_prob"] = y_prob
    return result


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_cv_experiment(
    model_catalogue: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    treatment_name: str = "flag_only",
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
    smote_candidates: list[str] | None = None,
) -> pd.DataFrame:
    """
    Run cross-validation for all models in the catalogue.

    Parameters
    ----------
    model_catalogue : dict[str, Pipeline]
        Name → unfitted Pipeline mapping.
    X_train : pd.DataFrame
        Training features (after treatment has been applied).
    y_train : np.ndarray
        Training labels.
    treatment_name : str
        Label for the anomaly treatment (used in results column only).
    cv_folds : int
        Number of folds.
    random_state : int
        Reproducibility seed.
    smote_candidates : list[str] | None
        Model names for which a SMOTE variant should also be evaluated.
        If None, SMOTE is not evaluated.

    Returns
    -------
    pd.DataFrame
        One row per model/variant with columns:
        model, treatment, imbalance_strategy,
        mean_pr_auc, std_pr_auc, mean_roc_auc, std_roc_auc,
        mean_f1, std_f1, mean_precision, std_precision, mean_recall, std_recall
    """
    rows = []

    for name, pipeline in model_catalogue.items():
        print(f"  [CV] {name} | treatment={treatment_name} ...")
        cv_result = cross_validate_pipeline(
            pipeline, X_train, y_train,
            cv_folds=cv_folds, random_state=random_state,
        )

        # Determine imbalance strategy from model name
        if name.startswith("XGB") and "balanced" in name:
            imbalance_strategy = "scale_pos_weight"
        elif "balanced" in name:
            imbalance_strategy = "class_weight=balanced"
        elif "SPW" in name:
            imbalance_strategy = "scale_pos_weight"
        else:
            imbalance_strategy = "none"

        rows.append(
            {
                "model":             name,
                "treatment":         treatment_name,
                "imbalance_strategy": imbalance_strategy,
                "mean_pr_auc":       cv_result["mean_pr_auc"],
                "std_pr_auc":        cv_result["std_pr_auc"],
                "mean_roc_auc":      cv_result["mean_roc_auc"],
                "std_roc_auc":       cv_result["std_roc_auc"],
                "mean_f1":           cv_result["mean_f1"],
                "std_f1":            cv_result["std_f1"],
                "mean_precision":    cv_result["mean_precision"],
                "std_precision":     cv_result["std_precision"],
                "mean_recall":       cv_result["mean_recall"],
                "std_recall":        cv_result["std_recall"],
            }
        )

        # Optionally evaluate SMOTE variant
        if smote_candidates and name in smote_candidates:
            smote_name = f"{name}_SMOTE"
            print(f"  [CV] {smote_name} | treatment={treatment_name} ...")
            try:
                smote_result = cross_validate_with_smote(
                    pipeline, X_train, y_train,
                    cv_folds=cv_folds, random_state=random_state,
                )
                rows.append(
                    {
                        "model":             smote_name,
                        "treatment":         treatment_name,
                        "imbalance_strategy": "SMOTE",
                        "mean_pr_auc":       smote_result["mean_pr_auc"],
                        "std_pr_auc":        smote_result["std_pr_auc"],
                        "mean_roc_auc":      smote_result["mean_roc_auc"],
                        "std_roc_auc":       smote_result["std_roc_auc"],
                        "mean_f1":           smote_result["mean_f1"],
                        "std_f1":            smote_result["std_f1"],
                        "mean_precision":    smote_result["mean_precision"],
                        "std_precision":     smote_result["std_precision"],
                        "mean_recall":       smote_result["mean_recall"],
                        "std_recall":        smote_result["std_recall"],
                    }
                )
            except Exception as exc:
                print(f"    SMOTE evaluation failed for {name}: {exc}")

    return pd.DataFrame(rows)
