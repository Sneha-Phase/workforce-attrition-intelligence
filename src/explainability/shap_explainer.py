"""
src/explainability/shap_explainer.py
=====================================
SHAP-based global and local explanations for the Phase 3 Logistic Regression
pipeline.

Why SHAP is appropriate for this model
---------------------------------------
The selected model is a **Logistic Regression** (sklearn, lbfgs solver).
SHAP provides a ``LinearExplainer`` specifically designed for linear models.
It computes exact Shapley values in closed form using:

    φᵢ(x) = cᵢ · (xᵢ − E[xᵢ])

where ``cᵢ`` is the i-th coefficient and ``E[xᵢ]`` is the feature mean
from the background dataset.  For Logistic Regression, SHAP operates on
the *pre-activation* (log-odds) space; the returned SHAP values are additive
contributions to the model's log-odds output.

This is technically superior to raw coefficients for local explanations
because it accounts for the *deviation of each individual's feature value*
from the population baseline, giving per-row attribution rather than a
global average.

SHAP background dataset
------------------------
SHAP's LinearExplainer requires a background (reference) dataset to compute
E[xᵢ].  We use the **training portion** of the data (transformed by the
fitted preprocessor) to build this background.  This is NOT the held-out
test set — the test set is never touched here.  The explainer is built once
and serialised.

IMPORTANT
---------
- SHAP values reflect statistical associations, NOT causal relationships.
- A positive SHAP value for a feature means it pushes THIS employee's
  predicted log-odds *upward* relative to the population average.
- A negative SHAP value pushes predicted log-odds *downward*.
- This does NOT mean the feature causes attrition.
- These outputs are derived from a synthetic dataset and are illustrative only.

Synthetic-data limitation
--------------------------
The SHAP values are computed on synthetically generated data.  Magnitudes
and directions may not reflect patterns in real workforce populations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.explainability.feature_names import (
    get_feature_names_from_pipeline,
    transform_input,
)
from src.explainability.global_importance import CAUSATION_DISCLAIMER, SYNTHETIC_DATA_DISCLAIMER

# ---------------------------------------------------------------------------
# Path constant for the serialised explainer
# ---------------------------------------------------------------------------

_DEFAULT_EXPLAINER_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "shap_explainer.pkl"


# ---------------------------------------------------------------------------
# Explainer construction
# ---------------------------------------------------------------------------

def build_shap_explainer(
    pipeline: Pipeline,
    X_background: Union[pd.DataFrame, np.ndarray],
) -> shap.LinearExplainer:
    """
    Build a SHAP LinearExplainer for the fitted Logistic Regression pipeline.

    The background data is transformed through the pipeline's preprocessor
    first.  The explainer is then built on the transformed feature matrix,
    paired with the LR classifier's coefficients.

    Uses ``shap.maskers.Independent`` to compute interventional Shapley values —
    features are treated as statistically independent.  This is mathematically
    equivalent to ``feature_perturbation='interventional'`` in older SHAP
    versions and avoids the FutureWarning in SHAP >= 0.44.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        A *fitted* Pipeline with 'preprocessor' and 'clf' steps where 'clf'
        is a LogisticRegression.
    X_background : pd.DataFrame or np.ndarray
        The training feature matrix (NOT the held-out test set).
        Used to compute E[xᵢ] for the Shapley baseline.
        Typically: the full X_train DataFrame (before preprocessing).

    Returns
    -------
    shap.LinearExplainer
        A fitted SHAP LinearExplainer ready to call ``.shap_values()``.

    Notes
    -----
    - The held-out test set is NEVER passed to this function.
    - No re-fitting of the pipeline occurs here.
    - Background is transformed, not fitted — preprocessing is already frozen.

    Example
    -------
    >>> explainer = build_shap_explainer(pipeline, X_train)
    >>> shap_vals = explainer.shap_values(transform_input(pipeline, X_sample))
    """
    clf = pipeline.named_steps["clf"]
    X_bg_transformed = transform_input(pipeline, X_background)

    # Use shap.maskers.Independent for interventional SHAP — avoids the
    # feature_perturbation FutureWarning in SHAP >= 0.44.
    # Set max_samples to the full background size to avoid the subsampling
    # informational message (we pre-sample in generate_artifacts when needed).
    n_bg = X_bg_transformed.shape[0]
    masker = shap.maskers.Independent(X_bg_transformed, max_samples=n_bg)
    explainer = shap.LinearExplainer(clf, masker)
    return explainer


def save_shap_explainer(
    explainer: shap.LinearExplainer,
    path: Union[str, Path] = _DEFAULT_EXPLAINER_PATH,
) -> Path:
    """
    Persist the SHAP explainer to disk using joblib.

    Parameters
    ----------
    explainer : shap.LinearExplainer
        The fitted explainer to save.
    path : str or Path
        Destination file path.  Defaults to ``models/shap_explainer.pkl``.

    Returns
    -------
    Path
        The resolved path where the explainer was saved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(explainer, path)
    return path


def load_shap_explainer(
    path: Union[str, Path] = _DEFAULT_EXPLAINER_PATH,
) -> shap.LinearExplainer:
    """
    Load a previously saved SHAP explainer.

    Parameters
    ----------
    path : str or Path
        Path to the serialised explainer file.

    Returns
    -------
    shap.LinearExplainer

    Raises
    ------
    FileNotFoundError
        If the serialised file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"SHAP explainer not found at: {path}. "
            "Run generate_explainability_artifacts() to create it."
        )
    return joblib.load(path)


# ---------------------------------------------------------------------------
# SHAP value computation
# ---------------------------------------------------------------------------

def compute_shap_values(
    explainer: shap.LinearExplainer,
    pipeline: Pipeline,
    X: Union[pd.DataFrame, np.ndarray],
) -> np.ndarray:
    """
    Compute SHAP values for a set of observations.

    Parameters
    ----------
    explainer : shap.LinearExplainer
        A fitted SHAP LinearExplainer (from ``build_shap_explainer``).
    pipeline : sklearn.pipeline.Pipeline
        The fitted pipeline (used to transform X before SHAP computation).
    X : pd.DataFrame or np.ndarray
        Feature matrix for which to compute SHAP values.
        Must have the same columns as the training data.

    Returns
    -------
    numpy.ndarray, shape (n_samples, n_features)
        SHAP values (additive log-odds contributions) for each observation
        and each feature.  Positive → pushes prediction toward attrition.
        Negative → pushes prediction away from attrition.

    Notes
    -----
    - SHAP values are in log-odds space (pre-sigmoid) for Logistic Regression.
    - The sum of SHAP values + base_value ≈ the model's log-odds output.
    - No test-set data should be passed if the explainer was built on
      training data; this function does not enforce that constraint (it is
      the caller's responsibility).

    Example
    -------
    >>> shap_vals = compute_shap_values(explainer, pipeline, X_sample)
    >>> shap_vals.shape
    (n_samples, 36)
    """
    X_transformed = transform_input(pipeline, X)
    shap_values = explainer.shap_values(X_transformed)
    # LinearExplainer returns an ndarray; ensure it's 2D
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)
    return shap_values


def get_shap_global_importance(
    explainer: shap.LinearExplainer,
    pipeline: Pipeline,
    X_background: Union[pd.DataFrame, np.ndarray],
    *,
    top_n: int | None = None,
) -> pd.DataFrame:
    """
    Compute mean-absolute SHAP values over the background dataset to get a
    global feature importance ranking from the SHAP perspective.

    This is complementary to the coefficient-based global importance and
    weights each feature by how much it actually varies across the population.

    Parameters
    ----------
    explainer : shap.LinearExplainer
    pipeline : sklearn.pipeline.Pipeline
    X_background : pd.DataFrame or np.ndarray
        The same background data used to build the explainer.
    top_n : int | None
        If provided, return only the top N features.

    Returns
    -------
    pd.DataFrame with columns:
        rank             : int
        feature          : str
        mean_abs_shap    : float  — mean |SHAP value| across all background rows
        direction        : str    — based on mean (signed) SHAP value direction

    Notes
    -----
    - Mean-absolute SHAP is the standard SHAP global importance metric.
    - It accounts for both coefficient magnitude AND feature variation.
    - Purely statistical; does not establish causation.

    Example
    -------
    >>> shap_df = get_shap_global_importance(explainer, pipeline, X_train)
    """
    shap_values = compute_shap_values(explainer, pipeline, X_background)
    feature_names = get_feature_names_from_pipeline(pipeline)

    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs,
            "mean_signed_shap": mean_signed,
        }
    )
    df["direction"] = df["mean_signed_shap"].apply(
        lambda v: "(+) higher attrition risk" if v > 0 else "(-) lower attrition risk"
    )
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    if top_n is not None:
        df = df.head(top_n).reset_index(drop=True)

    return df[["rank", "feature", "mean_abs_shap", "direction"]]


def format_shap_global_report(shap_df: pd.DataFrame, top_n: int = 15) -> str:
    """
    Format the SHAP global importance DataFrame as a human-readable report.

    Parameters
    ----------
    shap_df : pd.DataFrame
        Output of ``get_shap_global_importance()``.
    top_n : int
        Number of features to include in the report.

    Returns
    -------
    str
    """
    lines = [
        "=" * 70,
        "SHAP GLOBAL FEATURE IMPORTANCE — Logistic Regression (LR_balanced)",
        "Method: Mean |SHAP value| (LinearExplainer, interventional)",
        "=" * 70,
        "",
        f"Showing top {top_n} features by mean absolute SHAP value.",
        "SHAP values are in log-odds space (additive contributions).",
        "",
        f"{'Rank':<5} {'Feature':<38} {'Mean|SHAP|':>10}  {'Direction'}",
        "-" * 70,
    ]

    for _, row in shap_df.head(top_n).iterrows():
        lines.append(
            f"{int(row['rank']):<5} {row['feature']:<38} "
            f"{row['mean_abs_shap']:>10.4f}  {row['direction']}"
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
