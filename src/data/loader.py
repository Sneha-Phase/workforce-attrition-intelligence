"""
src/data/loader.py
==================
Read-only data loading with schema validation.

Rules enforced here:
- The source CSV is NEVER modified (opened read-only via pandas).
- Employee_ID is retained in the returned DataFrame for audit purposes
  but callers are responsible for dropping it before modelling.
- All validation errors raise descriptive exceptions so failures are
  surfaced immediately rather than silently corrupting downstream steps.
"""

from __future__ import annotations

import pandas as pd

from src.config import (
    RAW_CSV,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    TARGET_POSITIVE,
    TARGET_NEGATIVE,
    EXPECTED_SHAPE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_data(path=None) -> pd.DataFrame:
    """
    Load the raw attrition CSV and validate its schema.

    Parameters
    ----------
    path : str or Path, optional
        Override the default RAW_CSV path (useful in tests).

    Returns
    -------
    pd.DataFrame
        The raw DataFrame, completely unchanged from the source CSV.
        No encoding, no dropping, no transformation is applied here.

    Raises
    ------
    FileNotFoundError
        If the CSV cannot be found at the resolved path.
    ValueError
        If any schema validation check fails.
    """
    resolved = path if path is not None else RAW_CSV

    try:
        df = pd.read_csv(resolved)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Source CSV not found at: {resolved}\n"
            "Ensure the file exists and the path in src/config.py is correct."
        )

    validate_schema(df)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """
    Assert that the DataFrame conforms to the expected schema contract.

    Checks performed (in order):
    1. All required columns are present.
    2. Shape matches expectation.
    3. No missing values anywhere.
    4. Target column contains only the two expected values.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate (typically the raw CSV load).

    Raises
    ------
    ValueError
        On the first failing check, with a descriptive message.
    """
    # 1. Required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Schema validation failed — missing columns: {missing_cols}"
        )

    # 2. Shape
    if df.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"Shape mismatch: expected {EXPECTED_SHAPE}, got {df.shape}. "
            "The source file may have been modified."
        )

    # 3. Missing values
    missing_counts = df.isnull().sum()
    cols_with_nulls = missing_counts[missing_counts > 0]
    if not cols_with_nulls.empty:
        raise ValueError(
            f"Missing values detected:\n{cols_with_nulls.to_string()}"
        )

    # 4. Target values
    actual_target_values = set(df[TARGET_COLUMN].unique())
    expected_target_values = {TARGET_POSITIVE, TARGET_NEGATIVE}
    if actual_target_values != expected_target_values:
        raise ValueError(
            f"Unexpected target values in '{TARGET_COLUMN}': "
            f"got {actual_target_values}, expected {expected_target_values}"
        )


def get_class_distribution(df: pd.DataFrame) -> pd.Series:
    """
    Return value counts (absolute) for the target column.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.Series
        Index: class labels ("Yes" / "No"), values: counts.
    """
    return df[TARGET_COLUMN].value_counts()


def get_attrition_rate(df: pd.DataFrame) -> float:
    """
    Return the overall attrition rate as a float in [0, 1].

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    float
        Proportion of rows where Attrition == "Yes".
    """
    return (df[TARGET_COLUMN] == TARGET_POSITIVE).mean()
