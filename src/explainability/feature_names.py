"""
src/explainability/feature_names.py
=====================================
Utility for extracting and mapping transformed feature names from the
fitted Phase 3 preprocessing pipeline.

The ColumnTransformer produces features in a specific order:
    1. nominal_ohe   → OHE dummy columns (one-per-category, drop='first')
    2. binary_ord    → Gender, Overtime (0/1 ordinal)
    3. ordinal_pass  → Ordinal/Likert + engineered binary flag (passthrough)
    4. scale_cont    → Scaled continuous features (raw + engineered)

verbose_feature_names_out=False means sklearn returns the raw feature names
(no "nominal_ohe__" prefix), which makes for cleaner readability.

This module is the single source of truth for feature-name-to-index mapping
used by both the global and local explainability modules.

IMPORTANT
---------
These utilities operate only on a *fitted* pipeline.  No fitting occurs here.
The source CSV is never touched.  The held-out test set is not used here.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def get_feature_names_from_pipeline(pipeline: Pipeline) -> list[str]:
    """
    Extract the ordered list of transformed feature names from a fitted
    sklearn Pipeline that has a 'preprocessor' step.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        A *fitted* Pipeline with a ColumnTransformer step named 'preprocessor'.

    Returns
    -------
    list[str]
        Feature names in the same column order as the transformed matrix.
        Length matches the number of coefficients in the classifier.

    Raises
    ------
    KeyError
        If the pipeline has no step named 'preprocessor'.
    sklearn.exceptions.NotFittedError
        If the preprocessor has not been fitted yet.

    Example
    -------
    >>> names = get_feature_names_from_pipeline(fitted_pipeline)
    >>> len(names)
    36
    """
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


def get_feature_name_index_map(pipeline: Pipeline) -> dict[str, int]:
    """
    Return a mapping of feature name → column index in the transformed matrix.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        A *fitted* Pipeline with a 'preprocessor' step.

    Returns
    -------
    dict[str, int]
        ``{feature_name: column_index, ...}``

    Example
    -------
    >>> idx_map = get_feature_name_index_map(fitted_pipeline)
    >>> idx_map["Overtime"]
    9
    """
    names = get_feature_names_from_pipeline(pipeline)
    return {name: idx for idx, name in enumerate(names)}


def extract_preprocessor(pipeline: Pipeline) -> ColumnTransformer:
    """
    Extract the fitted ColumnTransformer from the pipeline.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        A *fitted* Pipeline with a 'preprocessor' step.

    Returns
    -------
    sklearn.compose.ColumnTransformer (fitted)

    Raises
    ------
    KeyError
        If the pipeline has no step named 'preprocessor'.
    """
    return pipeline.named_steps["preprocessor"]


def transform_input(pipeline: Pipeline, X) -> "numpy.ndarray":  # type: ignore[name-defined]
    """
    Apply the preprocessing transform (only) to X — without running the
    classifier.  Returns a dense NumPy array of shape (n_samples, n_features).

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        A *fitted* Pipeline with a 'preprocessor' step.
    X : array-like or pd.DataFrame
        Feature matrix to transform.

    Returns
    -------
    numpy.ndarray, shape (n_samples, n_features)
    """
    preprocessor = extract_preprocessor(pipeline)
    return preprocessor.transform(X)
