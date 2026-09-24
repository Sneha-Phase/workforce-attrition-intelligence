"""
src/models/sensitivity.py
==========================
Leakage sensitivity analysis for the Workforce Attrition Intelligence project.

Purpose
-------
Evaluate whether excluding a set of potentially post-hoc / leaky features
materially changes CV performance. This is a SENSITIVITY ANALYSIS, not a
claim that any feature is leaked. The experiment compares:

    Primary feature set   : all engineered features present in the pipeline
    Sensitivity feature set : same as primary, but excluding:
        - Absenteeism
        - Training_Hours_Last_Year
        - Performance_Rating
        - Years_Since_Last_Promotion

If the model trained on the sensitivity set performs within a small margin
(e.g. 1–2% PR-AUC) of the primary set, we have evidence that the model is
not primarily riding on those features.  If performance drops substantially,
it suggests those features carry real signal and/or leakage should be
investigated further.

Design principles
-----------------
- The sensitivity set is constructed by dropping the suspect columns from X
  BEFORE building the preprocessor, so the pipeline is always internally
  consistent.
- A new ColumnTransformer is constructed with the reduced feature set.
- Results are stored separately and clearly labelled 'sensitivity'.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from src.config import (
    CATEGORICAL_BINARY,
    CATEGORICAL_NOMINAL,
    ORDINAL_FEATURES,
    RANDOM_STATE,
)
from src.preprocessing.pipeline import (
    ENGINEERED_CONTINUOUS,
    RAW_CONTINUOUS,
)
from src.features.build_features import COL_ROLE_TENURE_ANOMALY


# ---------------------------------------------------------------------------
# Features to exclude in the sensitivity analysis
# ---------------------------------------------------------------------------

SENSITIVITY_EXCLUDE_FEATURES = [
    "Absenteeism",
    "Training_Hours_Last_Year",
    "Performance_Rating",
    "Years_Since_Last_Promotion",
]

FEATURE_SET_PRIMARY     = "primary"
FEATURE_SET_SENSITIVITY = "sensitivity"


# ---------------------------------------------------------------------------
# Sensitivity preprocessor builder
# ---------------------------------------------------------------------------

_BINARY_CATEGORIES = [
    ["Female", "Male"],
    ["No", "Yes"],
]

ORDINAL_PASSTHROUGH = ORDINAL_FEATURES + [COL_ROLE_TENURE_ANOMALY]


def build_sensitivity_preprocessor(exclude: list[str] | None = None) -> ColumnTransformer:
    """
    Build a ColumnTransformer with the specified features excluded.

    Parameters
    ----------
    exclude : list[str] | None
        Feature names to exclude from the pipeline.  Defaults to
        SENSITIVITY_EXCLUDE_FEATURES.

    Returns
    -------
    ColumnTransformer (unfitted)
    """
    if exclude is None:
        exclude = SENSITIVITY_EXCLUDE_FEATURES

    exclude_set = set(exclude)

    # Filter ordinal passthrough (remove suspect ordinals like Performance_Rating)
    ordinal_pass = [f for f in ORDINAL_PASSTHROUGH if f not in exclude_set]

    # Filter continuous features
    continuous = [f for f in (RAW_CONTINUOUS + ENGINEERED_CONTINUOUS)
                  if f not in exclude_set]

    nominal_ohe = OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
        sparse_output=False,
    )
    binary_ord = OrdinalEncoder(
        categories=_BINARY_CATEGORIES,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("nominal_ohe",  nominal_ohe,     CATEGORICAL_NOMINAL),
            ("binary_ord",   binary_ord,       CATEGORICAL_BINARY),
            ("ordinal_pass", "passthrough",    ordinal_pass),
            ("scale_cont",   StandardScaler(), continuous),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def build_sensitivity_pipeline(classifier) -> Pipeline:
    """
    Build a leakage-safe pipeline using the sensitivity preprocessor.

    Parameters
    ----------
    classifier : sklearn estimator
        Unfitted classifier.

    Returns
    -------
    sklearn.pipeline.Pipeline (unfitted)
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_sensitivity_preprocessor()),
            ("clf", classifier),
        ]
    )


# ---------------------------------------------------------------------------
# Feature set dispatcher
# ---------------------------------------------------------------------------

def drop_sensitivity_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Return X with sensitivity features dropped (those that are present).

    Parameters
    ----------
    X : pd.DataFrame
        Training or evaluation feature matrix.

    Returns
    -------
    pd.DataFrame
        Copy with SENSITIVITY_EXCLUDE_FEATURES removed where present.
    """
    cols_to_drop = [c for c in SENSITIVITY_EXCLUDE_FEATURES if c in X.columns]
    return X.drop(columns=cols_to_drop).copy()
