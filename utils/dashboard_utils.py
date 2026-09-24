"""
utils/dashboard_utils.py
=========================
Shared utilities for the Workforce Attrition Intelligence Streamlit dashboard.

All helper functions here are:
- Pure (no Streamlit side-effects) so they can be unit-tested.
- Free of any training or fitting code.
- Free of any CSV mutation.

Responsibilities
----------------
- Artifact loading with Streamlit caching wrappers.
- Metric formatting helpers.
- Chart builders (returning Plotly figures, not rendering them).
- Disclaimer / warning text constants.
- Employee input row construction for the Risk Simulator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    MODELS_DIR,
    PIPELINE_PATH,
    MINORITY_CLASS_APPROX_PCT,
)

# ---------------------------------------------------------------------------
# Disclaimer constants  (single source of truth)
# ---------------------------------------------------------------------------

SYNTHETIC_DISCLAIMER = (
    "⚠ **Synthetic data**: This dashboard uses a computer-generated dataset. "
    "All patterns, metrics, and associations are illustrative only and "
    "**do not represent real employees or any real organisation**."
)

ASSOCIATION_DISCLAIMER = (
    "⚠ **Association ≠ Causation**: Feature associations shown here are "
    "statistical patterns learned from synthetic data. They do **not** establish "
    "that any feature *causes* employee attrition."
)

RISK_SIGNAL_DISCLAIMER = (
    "⚠ **Analytical signal only**: Attrition risk scores are probabilistic "
    "estimates for analytical purposes. A 'High Risk' label does **not** mean "
    "an employee will or should leave. Human review and contextual judgment "
    "are always required."
)

MODEL_LIMITATION_DISCLAIMER = (
    "⚠ **Model limitation**: The selected model's PR-AUC (~0.20) is near the "
    f"class-prevalence baseline (~{MINORITY_CLASS_APPROX_PCT:.0%}). "
    "Predictions have **limited discriminative ability** and should not be "
    "used for consequential individual employment decisions."
)

NO_ADVERSE_ACTION_DISCLAIMER = (
    "🚫 **No adverse action**: This tool must **not** be used to fire, demote, "
    "penalise, or take any adverse employment action against any individual "
    "based on a risk score. Risk scores are exploratory analytical signals only."
)


# ---------------------------------------------------------------------------
# Artifact loading (cached so they load once per Streamlit session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model pipeline…")
def load_pipeline():
    """Load the serialised sklearn Pipeline from models/best_pipeline.pkl."""
    path = PIPELINE_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Pipeline not found at {path}. Run Phase 3 training first."
        )
    return joblib.load(path)


@st.cache_resource(show_spinner="Loading SHAP explainer…")
def load_explainer():
    """Load the serialised SHAP LinearExplainer from models/shap_explainer.pkl."""
    path = MODELS_DIR / "shap_explainer.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"SHAP explainer not found at {path}. Run Phase 4 explainability first."
        )
    return joblib.load(path)


@st.cache_data(show_spinner="Loading dataset…")
def load_dataset() -> pd.DataFrame:
    """Load and validate the raw attrition CSV."""
    from src.data.loader import load_data
    return load_data()


@st.cache_data(show_spinner="Loading CV results…")
def load_cv_results() -> dict:
    """Load cv_results.json from models/."""
    path = MODELS_DIR / "cv_results.json"
    with open(path, "r") as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading test results…")
def load_test_results() -> dict:
    """Load final_test_results.json from models/."""
    path = MODELS_DIR / "final_test_results.json"
    with open(path, "r") as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading global importance…")
def load_global_coef_importance() -> pd.DataFrame:
    """Load pre-computed coefficient importance CSV."""
    return pd.read_csv(MODELS_DIR / "global_importance_coef.csv")


@st.cache_data(show_spinner="Loading SHAP importance…")
def load_global_shap_importance() -> pd.DataFrame:
    """Load pre-computed SHAP importance CSV."""
    return pd.read_csv(MODELS_DIR / "global_importance_shap.csv")


# ---------------------------------------------------------------------------
# Metric formatters
# ---------------------------------------------------------------------------

def fmt_pct(value: float, decimals: int = 1) -> str:
    """Format a float [0-1] as a percentage string, e.g. 0.197 → '19.7%'."""
    return f"{value * 100:.{decimals}f}%"


def fmt_float(value: float, decimals: int = 3) -> str:
    """Format a float with fixed decimals."""
    return f"{value:.{decimals}f}"


def risk_color(risk_label: str) -> str:
    """Return a CSS-compatible color string for a risk label."""
    colors = {
        "High Risk": "#e74c3c",
        "Moderate Risk": "#f39c12",
        "Low Risk": "#27ae60",
    }
    return colors.get(risk_label, "#7f8c8d")


def risk_emoji(risk_label: str) -> str:
    """Return an emoji indicator for a risk label."""
    emojis = {
        "High Risk": "🔴",
        "Moderate Risk": "🟡",
        "Low Risk": "🟢",
    }
    return emojis.get(risk_label, "⚪")


# ---------------------------------------------------------------------------
# Attrition summary helpers
# ---------------------------------------------------------------------------

def compute_attrition_summary(df: pd.DataFrame) -> dict:
    """
    Compute top-level workforce summary statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset (must contain 'Attrition' column).

    Returns
    -------
    dict with keys:
        total_employees, attrition_count, retained_count,
        attrition_rate, retention_rate
    """
    total = len(df)
    attrition_count = int((df["Attrition"] == "Yes").sum())
    retained_count = total - attrition_count
    attrition_rate = attrition_count / total
    return {
        "total_employees": total,
        "attrition_count": attrition_count,
        "retained_count": retained_count,
        "attrition_rate": attrition_rate,
        "retention_rate": 1.0 - attrition_rate,
    }


def attrition_rate_by_group(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Compute attrition rate for each category in a column.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with 'Attrition' column.
    column : str
        Categorical or ordinal column to group by.

    Returns
    -------
    pd.DataFrame with columns: [column, 'attrition_rate', 'count', 'attrition_count']
        Sorted by attrition_rate descending.
    """
    grp = df.groupby(column).agg(
        count=("Attrition", "count"),
        attrition_count=("Attrition", lambda s: (s == "Yes").sum()),
    ).reset_index()
    grp["attrition_rate"] = grp["attrition_count"] / grp["count"]
    return grp.sort_values("attrition_rate", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plotly chart builders
# ---------------------------------------------------------------------------

def bar_chart_attrition_by_group(
    df: pd.DataFrame,
    column: str,
    title: str,
    color_scheme: str = "Reds",
) -> go.Figure:
    """
    Build a horizontal bar chart of attrition rate by group.

    Returns a Plotly Figure (does not call st.plotly_chart).
    """
    rate_df = attrition_rate_by_group(df, column)
    rate_df["attrition_pct"] = (rate_df["attrition_rate"] * 100).round(1)

    fig = px.bar(
        rate_df,
        x="attrition_pct",
        y=column,
        orientation="h",
        text="attrition_pct",
        color="attrition_pct",
        color_continuous_scale=color_scheme,
        labels={"attrition_pct": "Attrition Rate (%)", column: column},
        title=title,
        custom_data=["count", "attrition_count"],
    )
    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        hovertemplate=(
            f"<b>%{{y}}</b><br>"
            "Attrition Rate: %{x:.1f}%<br>"
            "Employees: %{customdata[0]}<br>"
            "Attrited: %{customdata[1]}"
            "<extra></extra>"
        ),
    )
    fig.update_layout(
        coloraxis_showscale=False,
        xaxis_title="Attrition Rate (%)",
        yaxis_title="",
        height=max(300, len(rate_df) * 45 + 100),
        margin=dict(l=10, r=80, t=50, b=40),
    )
    return fig


def histogram_chart(
    df: pd.DataFrame,
    column: str,
    title: str,
    color_col: Optional[str] = None,
    nbins: int = 30,
) -> go.Figure:
    """Build a histogram with optional attrition overlay."""
    if color_col:
        fig = px.histogram(
            df, x=column, color=color_col,
            nbins=nbins, barmode="overlay",
            color_discrete_map={"Yes": "#e74c3c", "No": "#3498db"},
            title=title,
            labels={column: column, "count": "Count"},
            opacity=0.75,
        )
    else:
        fig = px.histogram(
            df, x=column, nbins=nbins,
            title=title,
            color_discrete_sequence=["#3498db"],
        )
    fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=50, b=40),
        legend_title_text="Attrition",
    )
    return fig


def pie_chart(
    labels: list,
    values: list,
    title: str,
    colors: Optional[list] = None,
) -> go.Figure:
    """Build a donut pie chart."""
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=colors or px.colors.qualitative.Set2,
        textinfo="label+percent",
    ))
    fig.update_layout(
        title=title,
        height=350,
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=True,
    )
    return fig


def horizontal_bar_importance(
    df: pd.DataFrame,
    feature_col: str,
    value_col: str,
    direction_col: str,
    title: str,
    top_n: int = 20,
) -> go.Figure:
    """
    Build a horizontal bar chart for feature importance (coef or SHAP).

    Positive contributions are shown in red, negative in blue.
    """
    plot_df = df.head(top_n).copy()
    # Colour by direction
    plot_df["color"] = plot_df[direction_col].apply(
        lambda d: "#e74c3c" if "(+)" in d else "#3498db"
    )
    plot_df = plot_df.sort_values(value_col, ascending=True)  # ascending for horizontal

    fig = go.Figure(go.Bar(
        x=plot_df[value_col],
        y=plot_df[feature_col],
        orientation="h",
        marker_color=plot_df["color"],
        text=plot_df[value_col].apply(lambda v: f"{v:.4f}"),
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        xaxis_title=value_col.replace("_", " ").title(),
        yaxis_title="",
        height=max(400, top_n * 28 + 100),
        margin=dict(l=10, r=80, t=50, b=40),
    )
    return fig


def local_shap_waterfall(
    local_shap_df: pd.DataFrame,
    base_value: float,
    predicted_prob: float,
    title: str = "Local SHAP Explanation",
    top_n: int = 12,
) -> go.Figure:
    """
    Build a waterfall-style bar chart for local SHAP contributions.

    Red bars = features pushing risk up; blue bars = pushing risk down.
    """
    plot_df = local_shap_df.head(top_n).copy()
    plot_df = plot_df.sort_values("shap_value", ascending=True)

    colors = ["#e74c3c" if v > 0 else "#3498db" for v in plot_df["shap_value"]]

    fig = go.Figure(go.Bar(
        x=plot_df["shap_value"],
        y=plot_df["feature"],
        orientation="h",
        marker_color=colors,
        text=plot_df["shap_value"].apply(lambda v: f"{v:+.4f}"),
        textposition="outside",
    ))

    # Add a zero-line
    fig.add_vline(x=0, line_width=1.5, line_color="black")

    fig.update_layout(
        title=f"{title} — Predicted probability: {predicted_prob:.1%}",
        xaxis_title="SHAP value (log-odds contribution)",
        yaxis_title="",
        height=max(350, top_n * 32 + 100),
        margin=dict(l=10, r=80, t=60, b=40),
    )
    return fig


def confusion_matrix_heatmap(
    cm: list,
    title: str = "Confusion Matrix",
) -> go.Figure:
    """Build an annotated confusion matrix heatmap from a 2×2 list."""
    labels = ["No Attrition", "Attrition"]
    z = [[cm[0][0], cm[0][1]], [cm[1][0], cm[1][1]]]
    text = [[str(v) for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=["Predicted: No", "Predicted: Yes"],
        y=["Actual: No", "Actual: Yes"],
        text=text,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=True,
    ))
    fig.update_layout(
        title=title,
        height=350,
        margin=dict(l=10, r=10, t=50, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Employee profile construction for Risk Simulator
# ---------------------------------------------------------------------------

def build_employee_row(profile: dict) -> pd.DataFrame:
    """
    Build a single-row DataFrame from a simulator profile dict, applying
    feature engineering so it can be passed directly to the pipeline.

    The clamp_and_flag anomaly treatment is applied unconditionally because
    the production model (LR_balanced / clamp_and_flag) was trained with a
    preprocessor that includes the ``Years_in_Current_Role_Clamped`` column.
    For normal profiles (Years_in_Current_Role <= Years_at_Company) the
    clamp has no effect on the value, but the column must still be present.

    Parameters
    ----------
    profile : dict
        Raw employee feature values (pre-engineering) matching REQUIRED_COLUMNS
        minus Employee_ID and Attrition.

    Returns
    -------
    pd.DataFrame with 1 row, all engineered columns present,
    including ``Years_in_Current_Role_Clamped``.
    """
    from src.features.build_features import build_features
    from src.models.anomaly_treatments import apply_clamp_and_flag

    # Wrap in DataFrame (single row)
    df = pd.DataFrame([profile])
    df = build_features(df)

    # Apply the clamp_and_flag treatment to create Years_in_Current_Role_Clamped.
    # This mirrors the training-time transformation for the selected production
    # model (LR_balanced + clamp_and_flag treatment).  apply_clamp_and_flag
    # operates on a copy and never mutates the source data.
    df = apply_clamp_and_flag(df)

    # Drop Attrition / Employee_ID if accidentally present
    for col in ["Attrition", "Employee_ID"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df
