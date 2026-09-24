"""
src/preprocessing/pipeline.py
==============================
Leakage-safe preprocessing pipeline for the Workforce Attrition Intelligence
project.

Design principles
-----------------
- The pipeline is BUILT here (``build_preprocessor``), never fitted here.
  Fitting happens in the training script on training data only.
- The ColumnTransformer is constructed from the feature-group constants
  defined in ``src.config`` so that the schema contract is a single source
  of truth.
- ``handle_unknown='ignore'`` on the OHE transformer makes inference safe
  when unseen categorical values appear at Streamlit runtime.
- Binary features are encoded via OrdinalEncoder with explicit category lists
  so the mapping is deterministic regardless of DataFrame sort order.
- Ordinal / Likert features are already integer-coded and are passed through
  untouched; StandardScaler is applied only to continuous numeric features.
- All transformations are standard sklearn estimators — the pipeline serialises
  cleanly with joblib for later model training and Streamlit inference.

Feature group summary (mirrors ``src.config``)
-----------------------------------------------
Nominal categorical   → OneHotEncoder  (drop='first', handle_unknown='ignore')
  Department, Job_Role, Marital_Status

Binary categorical    → OrdinalEncoder (explicit categories)
  Gender  : ['Female', 'Male']   → 0, 1
  Overtime: ['No', 'Yes']        → 0, 1

Ordinal / pass-through → 'passthrough'
  Job_Level, Work_Life_Balance, Job_Satisfaction, Performance_Rating,
  Work_Environment_Satisfaction, Relationship_with_Manager,
  Job_Involvement, Number_of_Companies_Worked,
  Role_Tenure_Anomaly (engineered binary flag)

Continuous numeric     → StandardScaler
  Age, Monthly_Income, Hourly_Rate, Years_at_Company,
  Years_in_Current_Role, Years_Since_Last_Promotion,
  Training_Hours_Last_Year, Project_Count,
  Average_Hours_Worked_Per_Week, Absenteeism, Distance_From_Home,
  Satisfaction_Composite, Tenure_Promotion_Ratio,
  Hours_Overload, Income_to_Role_Ratio

Target encoding (outside ColumnTransformer)
  Attrition : 'Yes' → 1, 'No' → 0   via encode_target()
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
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
    ID_COLUMN,
    ORDINAL_FEATURES,
    TARGET_COLUMN,
    TARGET_NEGATIVE,
    TARGET_POSITIVE,
)
from src.features.build_features import (
    COL_HOURS_OVERLOAD,
    COL_INCOME_TO_ROLE_RATIO,
    COL_ROLE_TENURE_ANOMALY,
    COL_SATISFACTION_COMPOSITE,
    COL_TENURE_PROMOTION_RATIO,
)


# ---------------------------------------------------------------------------
# Feature group constants
# ---------------------------------------------------------------------------

# Engineered features produced by build_features()
ENGINEERED_CONTINUOUS = [
    COL_SATISFACTION_COMPOSITE,
    COL_TENURE_PROMOTION_RATIO,
    COL_HOURS_OVERLOAD,
    COL_INCOME_TO_ROLE_RATIO,
]

# Binary flag — already 0/1; pass through without scaling
ENGINEERED_BINARY = [COL_ROLE_TENURE_ANOMALY]

# Ordinal / Likert integers (natural order preserved; no scaling needed)
ORDINAL_PASSTHROUGH = ORDINAL_FEATURES + ENGINEERED_BINARY  # list

# Continuous features from the raw dataset (need scaling)
RAW_CONTINUOUS = [
    "Age",
    "Monthly_Income",
    "Hourly_Rate",
    "Years_at_Company",
    "Years_in_Current_Role",
    "Years_Since_Last_Promotion",
    "Training_Hours_Last_Year",
    "Project_Count",
    "Average_Hours_Worked_Per_Week",
    "Absenteeism",
    "Distance_From_Home",
]

# All continuous features (raw + engineered)
CONTINUOUS_FEATURES = RAW_CONTINUOUS + ENGINEERED_CONTINUOUS

# Explicit category lists for binary OrdinalEncoder
_BINARY_CATEGORIES = [
    ["Female", "Male"],  # Gender  → 0, 1
    ["No", "Yes"],       # Overtime → 0, 1
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """
    Construct the ColumnTransformer preprocessing pipeline (unfitted).

    The returned transformer must be fitted on training data only:

        preprocessor = build_preprocessor()
        X_train_enc = preprocessor.fit_transform(X_train)
        X_test_enc  = preprocessor.transform(X_test)

    Transformers applied
    --------------------
    nominal_ohe  :  OneHotEncoder
        Columns : CATEGORICAL_NOMINAL (Department, Job_Role, Marital_Status)
        drop='first' removes one dummy per feature to prevent multicollinearity.
        handle_unknown='ignore' silently zeros out unseen categories at
        inference time (Streamlit safety).
        sparse_output=False returns a dense array throughout.

    binary_ord   :  OrdinalEncoder
        Columns : CATEGORICAL_BINARY (Gender, Overtime)
        Explicit category lists ensure Female=0/Male=1 and No=0/Yes=1
        regardless of the sort order of the input DataFrame.
        handle_unknown='use_encoded_value' with unknown_value=-1 tags
        any unrecognised value with −1 so it is detectable downstream.

    ordinal_pass :  'passthrough'
        Columns : ORDINAL_FEATURES + Role_Tenure_Anomaly
        These are already integer-coded with a meaningful order; no
        transformation is needed or desired.

    scale_cont   :  StandardScaler
        Columns : RAW_CONTINUOUS + ENGINEERED_CONTINUOUS
        Zero-mean / unit-variance scaling.  Required for Logistic Regression;
        tree-based models are unaffected (monotone scaling).

    Returns
    -------
    sklearn.compose.ColumnTransformer
        An unfitted ColumnTransformer with remainder='drop' (any column
        not listed above — including Employee_ID if accidentally present —
        is silently dropped).
    """
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
            ("nominal_ohe",  nominal_ohe,      CATEGORICAL_NOMINAL),
            ("binary_ord",   binary_ord,        CATEGORICAL_BINARY),
            ("ordinal_pass", "passthrough",     ORDINAL_PASSTHROUGH),
            ("scale_cont",   StandardScaler(),  CONTINUOUS_FEATURES),
        ],
        remainder="drop",   # silently drops Employee_ID and any unlisted col
        verbose_feature_names_out=False,
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """
    Return the ordered list of output feature names after fitting.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        A *fitted* ColumnTransformer returned by ``build_preprocessor()``.

    Returns
    -------
    list[str]
        Feature names in the same column order as the transformed matrix.

    Raises
    ------
    sklearn.exceptions.NotFittedError
        If the preprocessor has not been fitted yet.
    """
    return list(preprocessor.get_feature_names_out())


def encode_target(y: pd.Series) -> np.ndarray:
    """
    Encode the Attrition target column to binary integers.

    Mapping: 'Yes' → 1  (attrition / positive class)
             'No'  → 0  (retained / negative class)

    Parameters
    ----------
    y : pd.Series
        The raw Attrition column containing string values 'Yes' / 'No'.

    Returns
    -------
    np.ndarray of dtype int, shape (n_samples,)

    Raises
    ------
    ValueError
        If ``y`` contains any value other than 'Yes' or 'No'.
    """
    unexpected = set(y.unique()) - {TARGET_POSITIVE, TARGET_NEGATIVE}
    if unexpected:
        raise ValueError(
            f"encode_target: unexpected values in target column: {unexpected}. "
            f"Expected only '{TARGET_POSITIVE}' and '{TARGET_NEGATIVE}'."
        )
    return (y == TARGET_POSITIVE).astype(int).to_numpy()


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that must never enter the feature matrix.

    Specifically drops:
    - ``Employee_ID`` (surrogate key, zero predictive signal)
    - ``Attrition``   (the target — must be separated before this call)

    The ColumnTransformer's ``remainder='drop'`` provides a second line of
    defence, but explicit removal here makes the separation visible and
    testable.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that may still contain Employee_ID and/or Attrition.
        This function does NOT require them to be present — missing columns
        are silently ignored.

    Returns
    -------
    pd.DataFrame
        A copy with Employee_ID and Attrition removed (if present).
    """
    drop_cols = [c for c in [ID_COLUMN, TARGET_COLUMN] if c in df.columns]
    return df.drop(columns=drop_cols)
