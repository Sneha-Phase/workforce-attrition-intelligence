"""
src/features/build_features.py
================================
Feature engineering for the Workforce Attrition Intelligence project.

Design principles
-----------------
- Operates on a COPY of the input DataFrame — the original is never mutated.
- Engineered features are additive; no existing columns are overwritten.
- Division-by-zero risks are explicitly guarded (see per-feature notes).
- The Role-Tenure anomaly is FLAGGED, not silently corrected.
- Each public function is independently testable.

Primary engineered features (from architecture plan, Section 2):
    1. Role_Tenure_Anomaly        — binary flag for the 22.84% anomaly
    2. Satisfaction_Composite     — mean of 5 Likert well-being scales
    3. Tenure_Promotion_Ratio     — career stagnation velocity
    4. Hours_Overload             — deviation from 40-hour standard week
    5. Income_to_Role_Ratio       — conditional; included but flagged as weak
                                    in this synthetic dataset

Excluded candidates and reasons
--------------------------------
    Role_Tenure_Ratio      — uses the anomalous column pair directly;
                             produces >1.0 ratios for 22.84% of rows.
    Absenteeism_per_Year   — compounds the leakage concern of Absenteeism.
    Training_per_Project   — no clear conceptual backing over raw inputs.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.config import ANOMALY_COLUMN, ANOMALY_BASELINE


# ---------------------------------------------------------------------------
# Names of engineered feature columns (single source of truth)
# ---------------------------------------------------------------------------
COL_ROLE_TENURE_ANOMALY    = "Role_Tenure_Anomaly"
COL_SATISFACTION_COMPOSITE = "Satisfaction_Composite"
COL_TENURE_PROMOTION_RATIO = "Tenure_Promotion_Ratio"
COL_HOURS_OVERLOAD         = "Hours_Overload"
COL_INCOME_TO_ROLE_RATIO   = "Income_to_Role_Ratio"

ENGINEERED_COLUMNS = [
    COL_ROLE_TENURE_ANOMALY,
    COL_SATISFACTION_COMPOSITE,
    COL_TENURE_PROMOTION_RATIO,
    COL_HOURS_OVERLOAD,
    COL_INCOME_TO_ROLE_RATIO,
]

# The 5 Likert-scale inputs for the satisfaction composite
_SATISFACTION_INPUTS = [
    "Job_Satisfaction",
    "Work_Life_Balance",
    "Work_Environment_Satisfaction",
    "Relationship_with_Manager",
    "Job_Involvement",
]


# ---------------------------------------------------------------------------
# Individual feature constructors (each accepts and returns a DataFrame copy)
# ---------------------------------------------------------------------------

def add_role_tenure_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag rows where Years_in_Current_Role > Years_at_Company.

    This is logically impossible (role tenure cannot exceed company tenure)
    and affects 22.84% of rows in the profiled dataset.  Rather than
    silently clamping the values, we create a binary flag and preserve
    both original columns unchanged.

    New column
    ----------
    Role_Tenure_Anomaly : int (0 or 1)
        1  →  anomalous row (Years_in_Current_Role > Years_at_Company)
        0  →  logically consistent row
    """
    df = df.copy()
    df[COL_ROLE_TENURE_ANOMALY] = (
        df[ANOMALY_COLUMN] > df[ANOMALY_BASELINE]
    ).astype(int)
    return df


def add_satisfaction_composite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a single well-being index as the row-wise mean of 5 Likert
    satisfaction scales.

    Inputs: Job_Satisfaction (1–5), Work_Life_Balance (1–4),
            Work_Environment_Satisfaction (1–4),
            Relationship_with_Manager (1–4), Job_Involvement (1–4).

    Division-by-zero risk: None.  The denominator is always 5 (constant).

    New column
    ----------
    Satisfaction_Composite : float
        Range: [1.0, 5.0] given the input ranges above.
        Higher values → better overall well-being.
    """
    df = df.copy()
    df[COL_SATISFACTION_COMPOSITE] = df[_SATISFACTION_INPUTS].mean(axis=1)
    return df


def add_tenure_promotion_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Career stagnation velocity: company tenure divided by years since last
    promotion (Laplace-smoothed by +1 to prevent division by zero when
    Years_Since_Last_Promotion == 0).

    Formula: Years_at_Company / (Years_Since_Last_Promotion + 1)

    Division-by-zero risk: Eliminated by the +1 offset.
        Years_Since_Last_Promotion min = 0  →  denominator min = 1.

    Interpretation
    --------------
    Higher ratio → employee has been at the company longer relative to
    their last promotion, suggesting either steady promotion or prolonged
    stagnation.  Combined with Years_at_Company this gives context.

    New column
    ----------
    Tenure_Promotion_Ratio : float ≥ 0
    """
    df = df.copy()
    df[COL_TENURE_PROMOTION_RATIO] = (
        df["Years_at_Company"] / (df["Years_Since_Last_Promotion"] + 1)
    )
    return df


def add_hours_overload(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deviation from a standard 40-hour working week.

    Formula: Average_Hours_Worked_Per_Week − 40

    Division-by-zero risk: None.  Simple subtraction.

    Range in this dataset: 30 − 40 = −10  to  59 − 40 = +19.

    New column
    ----------
    Hours_Overload : int
        Negative  →  part-time / under-utilised
        Zero      →  exactly standard hours
        Positive  →  overtime / overload
    """
    df = df.copy()
    df[COL_HOURS_OVERLOAD] = df["Average_Hours_Worked_Per_Week"] - 40
    return df


def add_income_to_role_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly income relative to job level.

    Formula: Monthly_Income / Job_Level

    Division-by-zero risk: Minimal.
        Job_Level range is 1–5; the minimum value is 1, so the denominator
        is always ≥ 1.

    Synthetic-data caveat (documented in architecture plan)
    -------------------------------------------------------
    In this dataset, Job_Level and Monthly_Income have no meaningful income
    gradient (median income is ~$11,400 across all 5 levels).  This feature
    may carry near-zero predictive signal.  It is included in the feature
    matrix but its SHAP importance is monitored; it will be dropped from the
    final model if consistently near zero.

    New column
    ----------
    Income_to_Role_Ratio : float > 0
    """
    df = df.copy()
    df[COL_INCOME_TO_ROLE_RATIO] = (
        df["Monthly_Income"] / df["Job_Level"]
    )
    return df


# ---------------------------------------------------------------------------
# Master builder: apply all feature engineering in one call
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all engineered features to a copy of the input DataFrame.

    This is the primary entry point for both the training pipeline and
    the Streamlit application.  It is idempotent: calling it twice on the
    same DataFrame produces the same result.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame as returned by ``load_data()``.  Must contain all
        columns listed in ``src.config.REQUIRED_COLUMNS``.

    Returns
    -------
    pd.DataFrame
        New DataFrame with 5 additional engineered columns appended.
        The original columns are unchanged.

    Notes
    -----
    - Employee_ID is retained here; drop it in the preprocessing pipeline.
    - Attrition (target) is retained here; encode it in the pipeline.
    """
    df = df.copy()
    df = add_role_tenure_anomaly(df)
    df = add_satisfaction_composite(df)
    df = add_tenure_promotion_ratio(df)
    df = add_hours_overload(df)
    df = add_income_to_role_ratio(df)
    return df


# ---------------------------------------------------------------------------
# Utility: return a summary of engineered feature statistics
# ---------------------------------------------------------------------------

def describe_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return descriptive statistics for the 5 engineered columns.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame that has already passed through ``build_features()``.

    Returns
    -------
    pd.DataFrame
        Standard .describe() output for the engineered columns only.
    """
    missing = [c for c in ENGINEERED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"The following engineered columns are missing: {missing}. "
            "Call build_features(df) before describe_engineered_features(df)."
        )
    return df[ENGINEERED_COLUMNS].describe()
