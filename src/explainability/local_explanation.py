"""
src/explainability/local_explanation.py
=========================================
Local (per-employee) explanation functions for the Phase 3 Logistic Regression
pipeline.

Local explanations answer the question:
    "Why did this specific employee receive this particular attrition risk score?"

Two complementary approaches are provided:
    1. SHAP-based local explanation (preferred)
       Uses LinearExplainer SHAP values (log-odds contributions per feature).
       Each value shows how much a specific feature VALUE pushed THIS
       employee's prediction above or below the population baseline.

    2. Coefficient × deviation local explanation (fallback)
       Manual recreation of the SHAP decomposition without the shap library.
       contribution_i = coef_i × (x_i − mean_x_i)
       where mean_x_i is the training-set mean of feature i.

By default the SHAP approach is used when an explainer is available.

IMPORTANT
---------
- Local explanations are PER-EMPLOYEE analytical signals only.
- A high positive SHAP value for "Overtime=1" means that, for THIS employee,
  the overtime flag contributed to a higher predicted probability relative to
  the average employee.  It does NOT mean overtime causes attrition.
- No employee should be described as certain to leave.
- Derived from a synthetic dataset; patterns may not apply to real data.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.explainability.feature_names import (
    get_feature_names_from_pipeline,
    transform_input,
)
from src.explainability.global_importance import CAUSATION_DISCLAIMER


# ---------------------------------------------------------------------------
# SHAP-based local explanation
# ---------------------------------------------------------------------------

def explain_local_shap(
    pipeline: Pipeline,
    explainer: shap.LinearExplainer,
    X_row: pd.DataFrame,
    *,
    top_n: int | None = 10,
) -> pd.DataFrame:
    """
    Generate a local SHAP explanation for a single employee (one row).

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        A *fitted* Pipeline (used for preprocessing the input row).
    explainer : shap.LinearExplainer
        A fitted SHAP LinearExplainer.
    X_row : pd.DataFrame
        A single-row DataFrame with the same columns as the training data.
        Must contain exactly 1 row.
    top_n : int | None
        If provided, show only the top N contributing features by |SHAP value|.
        If None, show all features.

    Returns
    -------
    pd.DataFrame with columns:
        feature          : str   — transformed feature name
        shap_value       : float — SHAP value (log-odds contribution)
        abs_shap         : float — |SHAP value|
        direction        : str   — "(+) increases risk" / "(-) decreases risk"
        transformed_value: float — the preprocessed feature value for this row

    Notes
    -----
    - Rows are sorted by abs_shap descending (most influential first).
    - SHAP base_value (mean prediction over background) is NOT included in the
      returned DataFrame but is accessible via ``explainer.expected_value``.

    Raises
    ------
    ValueError
        If X_row has more or fewer than 1 row.

    Example
    -------
    >>> local_df = explain_local_shap(pipeline, explainer, X_row=employee_df)
    >>> print(local_df[["feature", "shap_value", "direction"]].head(5))
    """
    if len(X_row) != 1:
        raise ValueError(
            f"X_row must have exactly 1 row; got {len(X_row)} rows. "
            "Pass a single-row DataFrame."
        )

    X_transformed = transform_input(pipeline, X_row)  # shape: (1, n_features)
    shap_values = explainer.shap_values(X_transformed)  # shape: (1, n_features) or (n_features,)

    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)

    feature_names = get_feature_names_from_pipeline(pipeline)
    sv_row = shap_values[0]          # shape: (n_features,)
    x_row = X_transformed[0]         # shape: (n_features,)

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "shap_value": sv_row,
            "transformed_value": x_row,
        }
    )
    df["abs_shap"] = df["shap_value"].abs()
    df["direction"] = df["shap_value"].apply(
        lambda v: "(+) increases risk" if v > 0 else "(-) decreases risk"
    )
    df = df.sort_values("abs_shap", ascending=False).reset_index(drop=True)

    if top_n is not None:
        df = df.head(top_n).reset_index(drop=True)

    return df[["feature", "shap_value", "abs_shap", "direction", "transformed_value"]]


# ---------------------------------------------------------------------------
# Coefficient × deviation local explanation (no SHAP library required)
# ---------------------------------------------------------------------------

def explain_local_coef_deviation(
    pipeline: Pipeline,
    X_row: pd.DataFrame,
    X_background: Union[pd.DataFrame, np.ndarray],
    *,
    top_n: int | None = 10,
) -> pd.DataFrame:
    """
    Local explanation using coefficient × (value − mean) decomposition.

    This is the closed-form manual equivalent of SHAP for linear models:
        contribution_i = coef_i × (x_i − mean_train_i)

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        A *fitted* Pipeline with 'preprocessor' and 'clf' steps.
    X_row : pd.DataFrame
        A single-row DataFrame.
    X_background : pd.DataFrame or np.ndarray
        Training data (NOT test set), used to compute mean_train_i.
    top_n : int | None
        If provided, return only top N features by |contribution|.

    Returns
    -------
    pd.DataFrame with columns:
        feature          : str   — transformed feature name
        coef             : float — LR coefficient
        deviation        : float — x_i − mean_train_i
        contribution     : float — coef × deviation
        abs_contribution : float — |contribution|
        direction        : str   — "(+) increases risk" / "(-) decreases risk"

    Notes
    -----
    - This function does NOT use the shap library.
    - Results match SHAP LinearExplainer exactly under the 'interventional'
      assumption.
    - The held-out test set must NOT be passed as X_background.

    Example
    -------
    >>> local_df = explain_local_coef_deviation(pipeline, X_row, X_train)
    """
    if len(X_row) != 1:
        raise ValueError(
            f"X_row must have exactly 1 row; got {len(X_row)} rows."
        )

    clf = pipeline.named_steps["clf"]
    feature_names = get_feature_names_from_pipeline(pipeline)

    X_bg_transformed = transform_input(pipeline, X_background)
    X_row_transformed = transform_input(pipeline, X_row)

    mean_bg = X_bg_transformed.mean(axis=0)
    coefs = clf.coef_[0]
    x_vals = X_row_transformed[0]

    deviation = x_vals - mean_bg
    contribution = coefs * deviation

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "coef": coefs,
            "deviation": deviation,
            "contribution": contribution,
        }
    )
    df["abs_contribution"] = df["contribution"].abs()
    df["direction"] = df["contribution"].apply(
        lambda v: "(+) increases risk" if v > 0 else "(-) decreases risk"
    )
    df = df.sort_values("abs_contribution", ascending=False).reset_index(drop=True)

    if top_n is not None:
        df = df.head(top_n).reset_index(drop=True)

    return df[["feature", "coef", "deviation", "contribution", "abs_contribution", "direction"]]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_local_explanation_report(
    local_df: pd.DataFrame,
    risk_probability: float,
    employee_id: Optional[Union[str, int]] = None,
    *,
    base_value: Optional[float] = None,
    top_n: int = 10,
    use_shap: bool = True,
) -> str:
    """
    Format a local explanation as a human-readable text report.

    Parameters
    ----------
    local_df : pd.DataFrame
        Output of ``explain_local_shap()`` or ``explain_local_coef_deviation()``.
    risk_probability : float
        The model's predicted attrition probability for this employee (0–1).
    employee_id : str | int | None
        Optional employee identifier for display purposes.
    base_value : float | None
        SHAP base value (expected_value from the explainer), if available.
    top_n : int
        Number of top features to show.
    use_shap : bool
        If True, label the report as SHAP-based; otherwise as coefficient-based.

    Returns
    -------
    str
        A multi-line text report with disclaimers.

    Example
    -------
    >>> report = format_local_explanation_report(local_df, risk_prob, employee_id=42)
    >>> print(report)
    """
    method = "SHAP (LinearExplainer)" if use_shap else "Coefficient x Deviation"
    emp_label = f"Employee: {employee_id}" if employee_id is not None else "Employee: [unspecified]"

    lines = [
        "=" * 70,
        f"LOCAL EXPLANATION — {method}",
        emp_label,
        "=" * 70,
        "",
        f"Predicted attrition probability: {risk_probability:.1%}",
    ]

    if base_value is not None:
        lines.append(f"SHAP base value (population avg log-odds): {base_value:.4f}")

    lines += [
        "",
        "NOTE: This is an analytical signal only. The employee is NOT described",
        "  as certain to leave. Association != causation.",
        "",
        f"Top {top_n} contributing features (most influential first):",
        "",
    ]

    if use_shap and "shap_value" in local_df.columns:
        lines.append(f"  {'Feature':<38} {'SHAP value':>10}  {'Direction'}")
        lines.append("  " + "-" * 60)
        for _, row in local_df.head(top_n).iterrows():
            lines.append(
                f"  {row['feature']:<38} {row['shap_value']:>10.4f}  {row['direction']}"
            )
    else:
        lines.append(f"  {'Feature':<38} {'Contribution':>12}  {'Direction'}")
        lines.append("  " + "-" * 64)
        for _, row in local_df.head(top_n).iterrows():
            lines.append(
                f"  {row['feature']:<38} {row['contribution']:>12.4f}  {row['direction']}"
            )

    lines += [
        "",
        "-" * 70,
        "",
        CAUSATION_DISCLAIMER,
        "=" * 70,
    ]
    return "\n".join(lines)
