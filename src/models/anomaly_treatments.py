"""
src/models/anomaly_treatments.py
=================================
Anomaly treatment variants for the Years_in_Current_Role > Years_at_Company
issue identified during Phase 1 profiling.

Background
----------
22.84% of rows have Years_in_Current_Role > Years_at_Company, which is
logically impossible. Three treatment strategies are evaluated:

    1. flag_only       — the approved Phase 2 default: add a binary flag
                         column (Role_Tenure_Anomaly) and leave both original
                         columns unchanged.

    2. clamp_and_flag  — clamp Years_in_Current_Role to Years_at_Company where
                         anomalous, THEN add the binary flag.  The original
                         data is NOT overwritten; a new clamped column is used
                         in the feature set.

    3. exclude         — sensitivity check: drop all anomalous rows entirely
                         from the training set.  Useful for measuring whether
                         the anomaly rows systematically distort model learning.
                         NEVER applied to the test set.

Design principles
-----------------
- Treatments operate on COPIES only — originals are never mutated.
- Each function takes a DataFrame that has already passed through
  build_features() and returns a modified copy suitable for modelling.
- The preprocessor pipeline is constructed fresh for each treatment so
  column sets are consistent.
- Treatment is applied to X_train (and y_train for exclude) only;
  the test set always uses flag_only (the approved baseline).
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.config import ANOMALY_COLUMN, ANOMALY_BASELINE
from src.features.build_features import COL_ROLE_TENURE_ANOMALY


# ---------------------------------------------------------------------------
# Treatment names (single source of truth for keys)
# ---------------------------------------------------------------------------

TREATMENT_FLAG_ONLY    = "flag_only"
TREATMENT_CLAMP_FLAG   = "clamp_and_flag"
TREATMENT_EXCLUDE      = "exclude"

ALL_TREATMENTS = [TREATMENT_FLAG_ONLY, TREATMENT_CLAMP_FLAG, TREATMENT_EXCLUDE]

# New column created by clamp_and_flag treatment
COL_ROLE_CLAMPED = "Years_in_Current_Role_Clamped"


# ---------------------------------------------------------------------------
# Treatment functions
# ---------------------------------------------------------------------------

def apply_flag_only(X: pd.DataFrame) -> pd.DataFrame:
    """
    Flag-only treatment (approved Phase 2 default).

    The Role_Tenure_Anomaly column must already be present (added by
    build_features). This function is a pass-through that validates the
    flag is present.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix after build_features() and prepare_features().

    Returns
    -------
    pd.DataFrame
        Unchanged copy (flag column is already present).
    """
    X = X.copy()
    if COL_ROLE_TENURE_ANOMALY not in X.columns:
        raise ValueError(
            f"apply_flag_only: '{COL_ROLE_TENURE_ANOMALY}' column not found. "
            "Ensure build_features() was called before apply_flag_only()."
        )
    return X


def apply_clamp_and_flag(X: pd.DataFrame) -> pd.DataFrame:
    """
    Controlled clamp + anomaly flag treatment.

    Creates a new column 'Years_in_Current_Role_Clamped' where the clamped
    value = min(Years_in_Current_Role, Years_at_Company).  The original
    Years_in_Current_Role column is RETAINED unchanged alongside the clamped
    version; the preprocessor will include both.

    The Role_Tenure_Anomaly flag is preserved as-is.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix after build_features() and prepare_features().

    Returns
    -------
    pd.DataFrame
        Copy with an additional 'Years_in_Current_Role_Clamped' column.
    """
    X = X.copy()
    if ANOMALY_COLUMN not in X.columns or ANOMALY_BASELINE not in X.columns:
        raise ValueError(
            f"apply_clamp_and_flag: requires '{ANOMALY_COLUMN}' and "
            f"'{ANOMALY_BASELINE}' columns in X."
        )
    X[COL_ROLE_CLAMPED] = np.minimum(
        X[ANOMALY_COLUMN].values,
        X[ANOMALY_BASELINE].values,
    )
    return X


def apply_exclude(
    X: pd.DataFrame,
    y: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Sensitivity exclusion treatment.

    Drops all anomalous rows (Role_Tenure_Anomaly == 1) from the training
    set.  Must NEVER be applied to the test set.

    Parameters
    ----------
    X : pd.DataFrame
        Training feature matrix after build_features() and prepare_features().
    y : np.ndarray
        Corresponding binary target array.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        (X_clean, y_clean) with anomalous rows removed.

    Raises
    ------
    ValueError
        If Role_Tenure_Anomaly column is absent.
    """
    if COL_ROLE_TENURE_ANOMALY not in X.columns:
        raise ValueError(
            f"apply_exclude: '{COL_ROLE_TENURE_ANOMALY}' column not found. "
            "Ensure build_features() was called before apply_exclude()."
        )
    mask = X[COL_ROLE_TENURE_ANOMALY].values == 0
    X_clean = X.loc[mask].copy().reset_index(drop=True)
    y_clean = y[mask]
    return X_clean, y_clean


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def apply_treatment(
    treatment: str,
    X: pd.DataFrame,
    y: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """
    Dispatcher: apply the named treatment to X (and optionally y).

    Parameters
    ----------
    treatment : str
        One of TREATMENT_FLAG_ONLY, TREATMENT_CLAMP_FLAG, TREATMENT_EXCLUDE.
    X : pd.DataFrame
        Training feature matrix.
    y : np.ndarray | None
        Required for TREATMENT_EXCLUDE; ignored for others.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray | None]
        (X_treated, y_treated) — y_treated is None for non-exclude treatments.

    Raises
    ------
    ValueError
        If treatment name is unrecognised or y is missing for exclude.
    """
    if treatment == TREATMENT_FLAG_ONLY:
        return apply_flag_only(X), y

    if treatment == TREATMENT_CLAMP_FLAG:
        return apply_clamp_and_flag(X), y

    if treatment == TREATMENT_EXCLUDE:
        if y is None:
            raise ValueError(
                "apply_treatment: y is required for the 'exclude' treatment."
            )
        return apply_exclude(X, y)

    raise ValueError(
        f"apply_treatment: unknown treatment '{treatment}'. "
        f"Choose from {ALL_TREATMENTS}."
    )
