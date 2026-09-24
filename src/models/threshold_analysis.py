"""
src/models/threshold_analysis.py
==================================
Threshold analysis utilities for the Workforce Attrition Intelligence project.

Background
----------
The default 0.50 classification threshold is rarely optimal for imbalanced
problems. With ~20% attrition, a lower threshold increases recall at the cost
of precision. This module provides tools to:

    1. Compute precision/recall/F1 across a range of thresholds.
    2. Find the threshold that maximises a target metric.
    3. Summarise threshold trade-offs for reporting.
    4. Compare threshold behaviour across models.

The final threshold selection is made on cross-validation OOF probabilities
to avoid test-set contamination.

Threshold selection strategy
-----------------------------
For attrition risk, recall (catching actual leavers) is typically more
valuable than precision. The default recommended threshold is selected to
maximise F1 on CV OOF data, but the full trade-off table is always reported
so the business can choose based on context.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.config import RANDOM_STATE


# ---------------------------------------------------------------------------
# OOF probability collection for threshold selection
# ---------------------------------------------------------------------------

def collect_oof_probabilities(
    pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    cv_folds: int = 5,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect out-of-fold (OOF) predicted probabilities via stratified K-fold.

    This is the correct way to select a threshold: evaluate thresholds on
    OOF predictions, not on test data or on training data.

    Parameters
    ----------
    pipeline : sklearn Pipeline
        Unfitted pipeline.
    X : pd.DataFrame
        Training features.
    y : np.ndarray
        Training binary labels.
    cv_folds : int
        Number of folds.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (oof_y_true, oof_y_prob) — arrays of length n_train, one entry per
        sample from its held-out fold.
    """
    import sklearn.base as skbase

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    oof_prob  = np.zeros(len(y), dtype=float)
    oof_ytrue = np.zeros(len(y), dtype=int)

    for train_idx, val_idx in skf.split(X, y):
        X_tr = X.iloc[train_idx]
        y_tr = y[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y[val_idx]

        clone = skbase.clone(pipeline)
        clone.fit(X_tr, y_tr)
        prob = clone.predict_proba(X_val)[:, 1]

        oof_prob[val_idx]  = prob
        oof_ytrue[val_idx] = y_val

    return oof_ytrue, oof_prob


# ---------------------------------------------------------------------------
# Threshold table
# ---------------------------------------------------------------------------

def build_threshold_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Build a table of precision / recall / F1 at each candidate threshold.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth labels.
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    thresholds : np.ndarray | None
        Candidate thresholds. Defaults to [0.10, 0.15, …, 0.90].

    Returns
    -------
    pd.DataFrame with columns:
        threshold, precision, recall, f1, predicted_positive_rate,
        tp, fp, fn, tn
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.10, 0.91, 0.05), 4)

    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = _confusion_values(y_true, y_pred)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rows.append(
                {
                    "threshold":               round(float(t), 4),
                    "precision":               precision_score(y_true, y_pred, zero_division=0),
                    "recall":                  recall_score(y_true, y_pred, zero_division=0),
                    "f1":                      f1_score(y_true, y_pred, zero_division=0),
                    "predicted_positive_rate": float(y_pred.mean()),
                    "tp":                      int(tp),
                    "fp":                      int(fp),
                    "fn":                      int(fn),
                    "tn":                      int(tn),
                }
            )
    return pd.DataFrame(rows)


def _confusion_values(y_true: np.ndarray, y_pred: np.ndarray):
    """Return (tn, fp, fn, tp) from confusion matrix."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    return tn, fp, fn, tp


# ---------------------------------------------------------------------------
# Optimal threshold selection
# ---------------------------------------------------------------------------

def select_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    optimise_for: str = "f1",
    min_recall: float | None = None,
    min_precision: float | None = None,
) -> dict:
    """
    Select the classification threshold that maximises the target metric,
    with optional constraints.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth labels.
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    optimise_for : str
        Metric to maximise.  One of 'f1', 'precision', 'recall'.
    min_recall : float | None
        Minimum acceptable recall.  If set, thresholds producing recall below
        this value are excluded.
    min_precision : float | None
        Minimum acceptable precision.  If set, thresholds producing precision
        below this value are excluded.

    Returns
    -------
    dict with keys:
        optimal_threshold, precision, recall, f1,
        predicted_positive_rate, tp, fp, fn, tn,
        pr_auc, roc_auc
    """
    import warnings

    prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_true, y_prob)
    # Drop the sentinel (1.0, 0.0) point with no threshold
    prec_arr = prec_arr[:-1]
    rec_arr  = rec_arr[:-1]

    # Apply constraints
    mask = np.ones(len(thresh_arr), dtype=bool)
    if min_recall is not None:
        mask &= (rec_arr >= min_recall)
    if min_precision is not None:
        mask &= (prec_arr >= min_precision)

    if not mask.any():
        # Fall back to best-f1 without constraints
        mask = np.ones(len(thresh_arr), dtype=bool)

    valid_thresh  = thresh_arr[mask]
    valid_prec    = prec_arr[mask]
    valid_rec     = rec_arr[mask]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if optimise_for == "f1":
            denom = valid_prec + valid_rec
            scores = np.where(denom > 0, 2 * valid_prec * valid_rec / denom, 0.0)
        elif optimise_for == "precision":
            scores = valid_prec
        elif optimise_for == "recall":
            scores = valid_rec
        else:
            raise ValueError(f"select_optimal_threshold: unknown metric '{optimise_for}'")

    best_idx  = int(np.argmax(scores))
    best_t    = float(valid_thresh[best_idx])
    y_pred    = (y_prob >= best_t).astype(int)
    tn, fp, fn, tp = _confusion_values(y_true, y_pred)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "optimal_threshold":        best_t,
            "precision":                float(precision_score(y_true, y_pred, zero_division=0)),
            "recall":                   float(recall_score(y_true, y_pred, zero_division=0)),
            "f1":                       float(f1_score(y_true, y_pred, zero_division=0)),
            "predicted_positive_rate":  float(y_pred.mean()),
            "tp":                       int(tp),
            "fp":                       int(fp),
            "fn":                       int(fn),
            "tn":                       int(tn),
            "pr_auc":                   float(average_precision_score(y_true, y_prob)),
            "roc_auc":                  float(roc_auc_score(y_true, y_prob)),
        }


# ---------------------------------------------------------------------------
# Model threshold comparison
# ---------------------------------------------------------------------------

def compare_thresholds_across_models(
    model_probs: dict[str, tuple[np.ndarray, np.ndarray]],
    optimise_for: str = "f1",
) -> pd.DataFrame:
    """
    Compare optimal thresholds across multiple models.

    Parameters
    ----------
    model_probs : dict[str, tuple[np.ndarray, np.ndarray]]
        Mapping of model_name → (y_true, y_prob).
    optimise_for : str
        Metric to maximise.

    Returns
    -------
    pd.DataFrame
        One row per model with optimal threshold and associated metrics.
    """
    rows = []
    for name, (y_true, y_prob) in model_probs.items():
        result = select_optimal_threshold(y_true, y_prob, optimise_for=optimise_for)
        result["model"] = name
        rows.append(result)
    df = pd.DataFrame(rows)
    cols = ["model", "optimal_threshold", "precision", "recall", "f1",
            "pr_auc", "roc_auc", "predicted_positive_rate", "tp", "fp", "fn", "tn"]
    return df[[c for c in cols if c in df.columns]]
