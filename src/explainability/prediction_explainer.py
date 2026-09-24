"""
src/explainability/prediction_explainer.py
============================================
High-level prediction explanation API that combines probability prediction,
SHAP local explanation, and formatted reporting into a single callable.

This module is the primary entry point for the Streamlit dashboard (Phase 5)
and any downstream consumers that need a complete explanation bundle for an
individual employee.

Design
------
``explain_prediction`` accepts a single-row DataFrame and returns a
structured ``PredictionExplanation`` object containing:
    - attrition_probability   (float, 0–1)
    - risk_label              (str: 'High Risk' / 'Moderate Risk' / 'Low Risk')
    - local_shap_df           (pd.DataFrame — per-feature SHAP contributions)
    - global_importance_df    (pd.DataFrame — global coefficient ranking)
    - text_report             (str — formatted human-readable summary)
    - base_value              (float — SHAP expected value)
    - feature_names           (list[str])
    - optimal_threshold       (float — from Phase 3 threshold analysis)

All returned objects are read-only data; no pipeline state is modified.

IMPORTANT
---------
- Risk labels are analytical signals only.  An employee labelled 'High Risk'
  is NOT described as certain to leave.
- Positive SHAP values indicate that a feature INCREASED predicted attrition
  probability for THIS employee relative to the population average.  This is
  NOT a causal claim.
- Derived from synthetic data; patterns may not apply to real-world settings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.explainability.feature_names import get_feature_names_from_pipeline
from src.explainability.global_importance import (
    CAUSATION_DISCLAIMER,
    SYNTHETIC_DATA_DISCLAIMER,
    get_global_feature_importance,
)
from src.explainability.local_explanation import (
    explain_local_shap,
    format_local_explanation_report,
)
from src.explainability.shap_explainer import compute_shap_values


# ---------------------------------------------------------------------------
# Risk-label thresholds
# ---------------------------------------------------------------------------

_RISK_HIGH_THRESHOLD = 0.40
_RISK_MODERATE_THRESHOLD = 0.20


def _classify_risk(probability: float, optimal_threshold: float = 0.40) -> str:
    """
    Classify a probability into a human-readable risk label.

    Thresholds
    ----------
    - probability >= optimal_threshold : 'High Risk'
    - 0.20 <= probability < optimal_threshold : 'Moderate Risk'
    - probability < 0.20 : 'Low Risk'

    Parameters
    ----------
    probability : float
        Predicted attrition probability (0–1).
    optimal_threshold : float
        The Phase 3 optimal F1 threshold.  Employees above this threshold
        would be flagged by the model at the selected operating point.

    Returns
    -------
    str : 'High Risk' | 'Moderate Risk' | 'Low Risk'

    Notes
    -----
    Risk labels are analytical signals only.  They do NOT imply certainty.
    """
    if probability >= optimal_threshold:
        return "High Risk"
    elif probability >= _RISK_MODERATE_THRESHOLD:
        return "Moderate Risk"
    else:
        return "Low Risk"


# ---------------------------------------------------------------------------
# Return structure
# ---------------------------------------------------------------------------

@dataclass
class PredictionExplanation:
    """
    Container for a complete prediction + explanation bundle for one employee.

    All fields are read-only analytical outputs.

    Fields
    ------
    attrition_probability : float
        Model's predicted probability that this employee will leave (0–1).
        This is an analytical signal, NOT a certainty.
    risk_label : str
        Human-readable risk tier: 'High Risk', 'Moderate Risk', or 'Low Risk'.
    optimal_threshold : float
        The F1-optimal probability threshold from Phase 3.
    local_shap_df : pd.DataFrame
        Per-feature SHAP contributions for this employee (sorted by |SHAP|).
    global_importance_df : pd.DataFrame
        Global coefficient-based feature ranking (entire model).
    text_report : str
        Formatted human-readable explanation report with disclaimers.
    base_value : float
        SHAP base value (population average log-odds prediction).
    feature_names : list[str]
        Ordered list of transformed feature names.
    employee_id : str | int | None
        Optional employee identifier passed in by the caller.
    disclaimers : list[str]
        Standardised disclaimer strings included in all outputs.
    """

    attrition_probability: float
    risk_label: str
    optimal_threshold: float
    local_shap_df: pd.DataFrame
    global_importance_df: pd.DataFrame
    text_report: str
    base_value: float
    feature_names: list
    employee_id: Optional[Union[str, int]] = None
    disclaimers: list = field(default_factory=lambda: [
        CAUSATION_DISCLAIMER,
        SYNTHETIC_DATA_DISCLAIMER,
    ])


# ---------------------------------------------------------------------------
# Primary explanation function
# ---------------------------------------------------------------------------

def explain_prediction(
    pipeline: Pipeline,
    explainer: shap.LinearExplainer,
    X_row: pd.DataFrame,
    *,
    employee_id: Optional[Union[str, int]] = None,
    optimal_threshold: float = 0.40,
    top_n: int = 10,
) -> PredictionExplanation:
    """
    Generate a complete prediction + explanation bundle for one employee.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        A *fitted* Pipeline.
    explainer : shap.LinearExplainer
        A fitted SHAP LinearExplainer (built from training data).
    X_row : pd.DataFrame
        A single-row DataFrame for the employee to explain.
        Must have the same schema as the training data (pre-engineered features
        such as Satisfaction_Composite must already be present).
    employee_id : str | int | None
        Optional identifier for the employee.
    optimal_threshold : float
        The Phase 3 F1-optimal threshold.  Defaults to 0.40 as a conservative
        fallback; pass ``cv_results["optimal_threshold"]`` for accuracy.
    top_n : int
        Number of top SHAP features to include in the local explanation.

    Returns
    -------
    PredictionExplanation
        Fully populated explanation bundle.

    Notes
    -----
    - This function does NOT modify the pipeline or explainer.
    - This function does NOT access the held-out test set.
    - Risk labels and SHAP values are for analytical purposes only.
    - Positive SHAP → feature INCREASED predicted probability for this employee.
    - Negative SHAP → feature DECREASED predicted probability for this employee.
    - These are statistical associations from synthetic data, NOT causal claims.

    Example
    -------
    >>> result = explain_prediction(pipeline, explainer, X_row=employee_df,
    ...                             employee_id=42, optimal_threshold=0.398)
    >>> print(result.text_report)
    >>> print(result.risk_label)
    """
    if len(X_row) != 1:
        raise ValueError(
            f"X_row must have exactly 1 row; got {len(X_row)} rows."
        )

    # 1. Predict probability
    prob_array = pipeline.predict_proba(X_row)
    attrition_prob = float(prob_array[0, 1])  # positive class probability

    # 2. Risk label
    risk_label = _classify_risk(attrition_prob, optimal_threshold)

    # 3. SHAP local explanation
    local_shap_df = explain_local_shap(
        pipeline, explainer, X_row, top_n=top_n
    )

    # 4. Global importance (coefficient-based)
    global_df = get_global_feature_importance(pipeline, top_n=top_n)

    # 5. Base value
    base_value = float(explainer.expected_value)

    # 6. Feature names
    feature_names = get_feature_names_from_pipeline(pipeline)

    # 7. Text report
    text_report = format_local_explanation_report(
        local_shap_df,
        risk_probability=attrition_prob,
        employee_id=employee_id,
        base_value=base_value,
        top_n=top_n,
        use_shap=True,
    )

    return PredictionExplanation(
        attrition_probability=attrition_prob,
        risk_label=risk_label,
        optimal_threshold=optimal_threshold,
        local_shap_df=local_shap_df,
        global_importance_df=global_df,
        text_report=text_report,
        base_value=base_value,
        feature_names=feature_names,
        employee_id=employee_id,
    )


# ---------------------------------------------------------------------------
# Batch explanation helper
# ---------------------------------------------------------------------------

def explain_batch(
    pipeline: Pipeline,
    explainer: shap.LinearExplainer,
    X: pd.DataFrame,
    *,
    optimal_threshold: float = 0.40,
    employee_ids: Optional[list] = None,
) -> pd.DataFrame:
    """
    Generate a summary explanation table for multiple employees.

    Returns a DataFrame with one row per employee containing:
        employee_id, attrition_probability, risk_label,
        top_feature_1, top_feature_1_shap, top_feature_2, ...

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
    explainer : shap.LinearExplainer
    X : pd.DataFrame
        Feature matrix for all employees to explain.
    optimal_threshold : float
    employee_ids : list | None
        Optional list of employee identifiers (length must match len(X)).

    Returns
    -------
    pd.DataFrame

    Notes
    -----
    - This function computes SHAP values for all rows at once (efficient).
    - Risk labels are analytical signals; no employee is described as certain
      to leave.
    - Synthetic-data limitation applies.

    Example
    -------
    >>> batch_df = explain_batch(pipeline, explainer, X_sample)
    >>> batch_df[["employee_id", "attrition_probability", "risk_label"]].head()
    """
    if employee_ids is not None and len(employee_ids) != len(X):
        raise ValueError(
            f"employee_ids length ({len(employee_ids)}) must match X length ({len(X)})."
        )

    # Probabilities for all rows
    proba = pipeline.predict_proba(X)[:, 1]  # shape: (n_samples,)

    # SHAP values for all rows
    shap_values = compute_shap_values(explainer, pipeline, X)  # (n_samples, n_features)
    feature_names = get_feature_names_from_pipeline(pipeline)

    rows = []
    for i in range(len(X)):
        emp_id = employee_ids[i] if employee_ids is not None else i
        prob = float(proba[i])
        risk = _classify_risk(prob, optimal_threshold)

        # Top 3 SHAP features for this employee
        sv = shap_values[i]
        top_idx = np.argsort(np.abs(sv))[::-1][:3]
        top_features = {
            f"top_feature_{j + 1}": feature_names[top_idx[j]] if j < len(top_idx) else ""
            for j in range(3)
        }
        top_shap = {
            f"top_shap_{j + 1}": float(sv[top_idx[j]]) if j < len(top_idx) else 0.0
            for j in range(3)
        }

        row = {"employee_id": emp_id, "attrition_probability": prob, "risk_label": risk}
        row.update(top_features)
        row.update(top_shap)
        rows.append(row)

    return pd.DataFrame(rows)
