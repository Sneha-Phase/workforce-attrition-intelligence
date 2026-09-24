"""
src/preprocessing/splitter.py
==============================
Stratified train/test splitting for the Workforce Attrition Intelligence
project.

Design principles
-----------------
- A single public function ``split_data`` encapsulates the full splitting
  contract so downstream callers (train.py, tests) cannot accidentally
  deviate from the approved parameters.
- The split is ALWAYS stratified on the encoded target to preserve the
  ~20% attrition class proportion in both partitions.
- ``RANDOM_STATE`` from ``src.config`` is the only source of randomness;
  passing a different seed is allowed only for testing purposes.
- The test set is returned as raw DataFrames / arrays.  It must NOT be
  touched until final model selection.
- Feature engineering (``build_features``) is applied BEFORE the split so
  that engineered columns are present in both partitions without any fit
  on test data.
- The preprocessor is NOT fitted here.  Fitting happens in the training
  script using ONLY ``X_train``.

Usage example (train.py)
------------------------
    from src.data.loader import load_data
    from src.features.build_features import build_features
    from src.preprocessing.pipeline import build_preprocessor, encode_target
    from src.preprocessing.splitter import split_data

    raw          = load_data()
    engineered   = build_features(raw)
    splits       = split_data(engineered)

    X_train, X_test = splits["X_train"], splits["X_test"]
    y_train, y_test = splits["y_train"], splits["y_test"]

    preprocessor = build_preprocessor()
    X_train_enc  = preprocessor.fit_transform(X_train)
    X_test_enc   = preprocessor.transform(X_test)   # ← never fit on test
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    ID_COLUMN,
    RANDOM_STATE,
    TARGET_COLUMN,
)
from src.preprocessing.pipeline import encode_target, prepare_features


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

class SplitResult(TypedDict):
    """
    Typed dict returned by ``split_data``.

    Keys
    ----
    X_train : pd.DataFrame
        Training features (Employee_ID and Attrition removed).
    X_test  : pd.DataFrame
        Test features (Employee_ID and Attrition removed).
        **Do not use for fitting or hyperparameter tuning.**
    y_train : np.ndarray of int, shape (n_train,)
        Binary-encoded training labels (1 = attrition, 0 = retained).
    y_test  : np.ndarray of int, shape (n_test,)
        Binary-encoded test labels.
        **Do not use for fitting or hyperparameter tuning.**
    """
    X_train: pd.DataFrame
    X_test:  pd.DataFrame
    y_train: np.ndarray
    y_test:  np.ndarray


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def split_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
) -> SplitResult:
    """
    Perform a stratified train/test split of the feature-engineered dataset.

    Steps performed (in order)
    --------------------------
    1. Separate the Attrition target from the feature matrix.
    2. Encode the target: 'Yes' → 1, 'No' → 0.
    3. Drop Employee_ID (and Attrition if still present) from features.
    4. Stratified split preserving the ~20% minority class proportion.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset *after* ``build_features()`` has been applied.
        Must contain ``Attrition`` and (optionally) ``Employee_ID``.
    test_size : float, optional
        Fraction of data allocated to the test set.  Default 0.20.
    random_state : int, optional
        Controls the shuffle before the split.
        Defaults to ``src.config.RANDOM_STATE`` (42).
        Override only in tests that verify reproducibility.

    Returns
    -------
    SplitResult
        A dict with keys ``X_train``, ``X_test``, ``y_train``, ``y_test``.

    Raises
    ------
    ValueError
        If ``Attrition`` is missing from ``df``, or if the target contains
        unexpected values (propagated from ``encode_target``).
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"split_data: '{TARGET_COLUMN}' column not found in DataFrame. "
            "Ensure the raw target has not been dropped before calling split_data."
        )

    # 1 & 2 — separate and encode target
    y_raw = df[TARGET_COLUMN]
    y = encode_target(y_raw)

    # 3 — drop ID and target from feature matrix
    X = prepare_features(df)

    # 4 — stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    return SplitResult(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train,
        y_test=y_test,
    )
