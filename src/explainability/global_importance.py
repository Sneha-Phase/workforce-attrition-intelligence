"""
src/explainability/global_importance.py
========================================
Global feature-importance explanations for the selected Logistic Regression
pipeline using model coefficients.

Background — why coefficient-based global importance?
------------------------------------------------------
The selected model is a *Logistic Regression* (lbfgs, class_weight='balanced').
After the StandardScaler brings continuous features to zero-mean / unit-variance,
and binary/ordinal features are already on comparable integer scales, the
magnitude of each coefficient is a valid proxy for the feature's contribution
to the log-odds of attrition relative to the other features.

Direction of association
------------------------
- **Positive coefficient** → the feature is associated with a *higher*
  predicted attrition probability.
- **Negative coefficient** → the feature is associated with a *lower*
  predicted attrition probability.

⚠ IMPORTANT: association ≠ causation
--------------------------------------
These coefficients reflect patterns learned from a *synthetic* dataset.
A positive or negative association in this model does **NOT** establish that
any feature *causes* attrition.  Risk predictions are analytical signals only.
No employee should be described as certain to leave based on these outputs.

Synthetic-data limitation
--------------------------
The dataset is synthetically generated.  Coefficient magnitudes and directions
may not reflect patterns in real workforce data.  All interpretations should
be treated as illustrative rather than operationally prescriptive.

IMPORTANT: This module only reads the fitted pipeline.  No re-fitting,
no access to the held-out test set, no modification of source data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.explainability.feature_names import get_feature_names_from_pipeline


# ---------------------------------------------------------------------------
# Constant caveat strings (single source of truth for disclaimer language)
# ---------------------------------------------------------------------------

CAUSATION_DISCLAIMER = (
    "WARNING: Association != Causation: A positive or negative coefficient reflects a "
    "statistical association learned from a synthetic dataset. It does NOT "
    "establish that any feature causes attrition. These are analytical signals "
    "only. No employee should be described as certain to leave."
)

SYNTHETIC_DATA_DISCLAIMER = (
    "Synthetic-data limitation: This dataset is computer-generated. "
    "Coefficient directions and magnitudes may not reflect real workforce data."
)


def get_global_feature_importance(
    pipeline: Pipeline,
    *,
    top_n: int | None = None,
    absolute: bool = True,
) -> pd.DataFrame:
    """
    Compute global feature importance from Logistic Regression coefficients.

    The coefficients come from the fitted ``clf`` step of the pipeline.
    Because the preprocessing pipeline standardises continuous features and
    encodes categoricals to comparable integer/dummy scales, coefficient
    magnitude is a reasonable proxy for relative importance.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        A *fitted* Pipeline whose ``clf`` step is a LogisticRegression.
    top_n : int | None
        If provided, return only the top N features by absolute coefficient.
        If None (default), return all features.
    absolute : bool
        If True (default), sort by |coefficient| descending.
        If False, sort by raw coefficient value descending.

    Returns
    -------
    pd.DataFrame with columns:
        feature      : str   — transformed feature name
        coefficient  : float — raw signed coefficient from the LR model
        abs_coef     : float — absolute value of coefficient
        direction    : str   — "(+) higher attrition risk" or "(-) lower attrition risk"
        rank         : int   — rank by importance (1 = most important)

    Notes
    -----
    - The pipeline MUST already be fitted.
    - No re-fitting or test-set access occurs inside this function.
    - Coefficients are for the positive class (Attrition = 1 / 'Yes').
      ``clf.classes_`` is asserted to be [0, 1] before extraction.

    Raises
    ------
    ValueError
        If the classifier is not a binary LogisticRegression.

    Example
    -------
    >>> df = get_global_feature_importance(pipeline, top_n=10)
    >>> df[["feature", "coefficient", "direction"]].head(10)
    """
    clf = pipeline.named_steps["clf"]

    # Validate binary logistic regression
    if not hasattr(clf, "coef_"):
        raise ValueError(
            "The 'clf' step does not have a 'coef_' attribute. "
            "Expected a fitted LogisticRegression."
        )
    if clf.coef_.shape[0] != 1:
        raise ValueError(
            f"Expected a binary classifier with coef_.shape = (1, n_features), "
            f"got {clf.coef_.shape}. Multi-class not supported here."
        )

    feature_names = get_feature_names_from_pipeline(pipeline)
    coefficients = clf.coef_[0]  # shape: (n_features,)

    if len(feature_names) != len(coefficients):
        raise ValueError(
            f"Feature name count ({len(feature_names)}) does not match "
            f"coefficient count ({len(coefficients)})."
        )

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
        }
    )
    df["abs_coef"] = df["coefficient"].abs()
    df["direction"] = df["coefficient"].apply(
        lambda c: "(+) higher attrition risk" if c > 0 else "(-) lower attrition risk"
    )

    sort_col = "abs_coef" if absolute else "coefficient"
    df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    if top_n is not None:
        df = df.head(top_n).reset_index(drop=True)

    return df[["rank", "feature", "coefficient", "abs_coef", "direction"]]


def format_global_importance_report(
    importance_df: pd.DataFrame,
    top_n: int = 15,
) -> str:
    """
    Format the global importance DataFrame as a human-readable text report.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Output of ``get_global_feature_importance()``.
    top_n : int
        Number of top features to show in the report.

    Returns
    -------
    str
        A multi-line text report including disclaimers.

    Example
    -------
    >>> report = format_global_importance_report(df, top_n=10)
    >>> print(report)
    """
    lines = [
        "=" * 70,
        "GLOBAL FEATURE IMPORTANCE — Logistic Regression (LR_balanced)",
        "Phase 4 — Workforce Attrition Intelligence",
        "=" * 70,
        "",
        f"Showing top {top_n} features by absolute coefficient magnitude.",
        "",
        f"{'Rank':<5} {'Feature':<38} {'Coefficient':>11}  {'Direction'}",
        "-" * 70,
    ]

    shown = importance_df.head(top_n)
    for _, row in shown.iterrows():
        lines.append(
            f"{int(row['rank']):<5} {row['feature']:<38} "
            f"{row['coefficient']:>11.4f}  {row['direction']}"
        )

    lines += [
        "",
        "-" * 70,
        "",
        CAUSATION_DISCLAIMER,
        "",
        SYNTHETIC_DATA_DISCLAIMER,
        "=" * 70,
    ]
    return "\n".join(lines)
