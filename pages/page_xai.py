"""
pages/page_xai.py
==================
Explainable AI — Page 5.

Shows:
  - Global coefficient importance (from pre-computed CSV)
  - Global SHAP importance (from pre-computed CSV)
  - Local SHAP explanation for a selected sample employee

All outputs carry association ≠ causation disclaimers.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.dashboard_utils import (
    ASSOCIATION_DISCLAIMER,
    MODEL_LIMITATION_DISCLAIMER,
    SYNTHETIC_DISCLAIMER,
    horizontal_bar_importance,
    load_dataset,
    load_explainer,
    load_global_coef_importance,
    load_global_shap_importance,
    load_pipeline,
)

# Local sample employees baked from models/local_explanation_sample.csv
# Probabilities reflect actual values from Phase 4 artifact generation
_SAMPLE_LABELS = [
    "Sample 1 (10th pct, p≈0.453)",
    "Sample 2 (30th pct, p≈0.481)",
    "Sample 3 (50th pct, p≈0.499)",
    "Sample 4 (70th pct, p≈0.518)",
    "Sample 5 (90th pct, p≈0.543)",
]


def _load_local_samples() -> dict[str, pd.DataFrame]:
    """Load the pre-computed local explanation sample CSV."""
    from src.config import MODELS_DIR
    path = MODELS_DIR / "local_explanation_sample.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    samples = {}
    for label, sample_key in zip(_SAMPLE_LABELS, df["employee_sample"].unique()):
        samples[label] = df[df["employee_sample"] == sample_key].copy()
    return samples


def _shap_waterfall_from_sample(sample_df: pd.DataFrame, title: str) -> go.Figure:
    """Build a waterfall-style bar chart from a local sample DataFrame."""
    prob = sample_df["predicted_probability"].iloc[0]
    plot_df = sample_df.sort_values("shap_value", ascending=True)
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in plot_df["shap_value"]]

    fig = go.Figure(go.Bar(
        x=plot_df["shap_value"],
        y=plot_df["feature"],
        orientation="h",
        marker_color=colors,
        text=plot_df["shap_value"].apply(lambda v: f"{v:+.4f}"),
        textposition="outside",
    ))
    fig.add_vline(x=0, line_width=1.5, line_color="black")
    fig.update_layout(
        title=f"{title} — Predicted probability: {prob:.1%}",
        xaxis_title="SHAP value (log-odds contribution)",
        yaxis_title="",
        height=max(350, len(plot_df) * 36 + 100),
        margin=dict(l=10, r=80, t=60, b=40),
    )
    return fig


def render() -> None:
    """Render the Explainable AI page."""

    st.title("🔍 Explainable AI")
    st.caption(
        "Feature importance and SHAP explanations for the selected Logistic "
        "Regression model. All outputs are statistical associations — "
        "**not causal claims**."
    )

    st.markdown(
        f'<div class="disclaimer-box">{ASSOCIATION_DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="disclaimer-box">{MODEL_LIMITATION_DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="disclaimer-box">{SYNTHETIC_DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs([
        "📊 Global Coefficient Importance",
        "🔢 Global SHAP Importance",
        "🔎 Local SHAP Explanation",
    ])

    # -------------------------------------------------------------------
    # Tab 1 — Global Coefficient Importance
    # -------------------------------------------------------------------
    with tab1:
        st.markdown(
            "#### Global Feature Importance — Logistic Regression Coefficients  \n"
            "After StandardScaler, coefficient magnitudes provide a relative "
            "importance ranking. Red = associated with higher predicted risk, "
            "Blue = associated with lower predicted risk."
        )

        try:
            coef_df = load_global_coef_importance()
        except Exception as e:
            st.error(f"Could not load coefficient importance: {e}")
            return

        top_n = st.slider("Show top N features", 5, 36, 20, key="coef_topn")

        fig_coef = horizontal_bar_importance(
            coef_df,
            feature_col="feature",
            value_col="abs_coef",
            direction_col="direction",
            title="Top Features by Absolute Coefficient Magnitude",
            top_n=top_n,
        )
        st.plotly_chart(fig_coef, use_container_width=True)

        # Signed coefficient chart
        with st.expander("Show signed coefficients (positive = higher risk, negative = lower risk)", expanded=False):
            plot_df = coef_df.head(top_n).copy()
            plot_df = plot_df.sort_values("coefficient", ascending=True)
            colors = ["#e74c3c" if v > 0 else "#3498db" for v in plot_df["coefficient"]]
            fig_signed = go.Figure(go.Bar(
                x=plot_df["coefficient"],
                y=plot_df["feature"],
                orientation="h",
                marker_color=colors,
                text=plot_df["coefficient"].apply(lambda v: f"{v:+.4f}"),
                textposition="outside",
            ))
            fig_signed.add_vline(x=0, line_width=1.5, line_color="black")
            fig_signed.update_layout(
                title="Signed LR Coefficients (positive → higher risk, negative → lower risk)",
                xaxis_title="Coefficient",
                height=max(350, top_n * 28 + 100),
                margin=dict(l=10, r=80, t=50, b=40),
            )
            st.plotly_chart(fig_signed, use_container_width=True)

        st.dataframe(coef_df.head(top_n), use_container_width=True, hide_index=True)

        st.caption(
            "⚠ **Association ≠ Causation**: Coefficient directions and magnitudes "
            "reflect statistical patterns in the synthetic training data. "
            "A positive coefficient does NOT mean the feature causes attrition."
        )

    # -------------------------------------------------------------------
    # Tab 2 — Global SHAP Importance
    # -------------------------------------------------------------------
    with tab2:
        st.markdown(
            "#### Global Feature Importance — Mean |SHAP Value|  \n"
            "Mean absolute SHAP value across the training background dataset. "
            "This weights each feature by how much it actually varied across employees, "
            "not just the raw coefficient magnitude."
        )

        try:
            shap_df = load_global_shap_importance()
        except Exception as e:
            st.error(f"Could not load SHAP importance: {e}")
            return

        top_n_shap = st.slider("Show top N features", 5, 36, 20, key="shap_topn")

        fig_shap = horizontal_bar_importance(
            shap_df,
            feature_col="feature",
            value_col="mean_abs_shap",
            direction_col="direction",
            title="Top Features by Mean |SHAP Value| (SHAP LinearExplainer, interventional)",
            top_n=top_n_shap,
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        st.dataframe(shap_df.head(top_n_shap), use_container_width=True, hide_index=True)

        st.caption(
            "⚠ SHAP values are in log-odds space. A higher mean |SHAP| means the feature "
            "contributed more to shifting predictions above or below the population average. "
            "This is a **statistical association** from a synthetic dataset — not causation."
        )

        st.info(
            "**How to read SHAP global importance:**  \n"
            "- Each bar shows how much (on average, across all training employees) "
            "  a feature pushed the predicted log-odds away from the baseline.  \n"
            "- Red = the mean effect was toward higher predicted attrition risk.  \n"
            "- Blue = the mean effect was toward lower predicted attrition risk.  \n"
            "- These directions can differ from the coefficient signs because SHAP "
            "  accounts for the actual distribution of feature values."
        )

    # -------------------------------------------------------------------
    # Tab 3 — Local SHAP Explanation
    # -------------------------------------------------------------------
    with tab3:
        st.markdown(
            "#### Local SHAP Explanation for Sample Employees  \n"
            "Shows how individual feature values pushed a specific employee's "
            "predicted probability **above or below the population baseline**."
        )
        st.info(
            "These are pre-computed sample employees from the Phase 4 artifact "
            "(`models/local_explanation_sample.csv`). Select a sample to view "
            "their local explanation."
        )

        samples = _load_local_samples()
        if not samples:
            st.warning("Local explanation sample file not found.")
            return

        selected_sample = st.selectbox(
            "Select sample employee",
            list(samples.keys()),
            key="local_sample_select",
        )
        sample_df = samples[selected_sample]
        prob = sample_df["predicted_probability"].iloc[0]
        risk_label = sample_df["risk_label"].iloc[0]

        # Risk badge
        risk_color_map = {
            "High Risk": "#e74c3c",
            "Moderate Risk": "#f39c12",
            "Low Risk": "#27ae60",
        }
        color = risk_color_map.get(risk_label, "#7f8c8d")
        st.markdown(
            f"**Predicted probability:** {prob:.1%} &nbsp;&nbsp;"
            f"<span class='risk-badge' style='background:{color};'>{risk_label}</span>",
            unsafe_allow_html=True,
        )

        st.warning(
            f"**Reminder**: A predicted probability of {prob:.1%} is an analytical "
            "signal only. This employee is NOT described as certain to leave. "
            "Model PR-AUC ≈ 0.20 means this score has very limited discriminative value."
        )

        fig_local = _shap_waterfall_from_sample(sample_df, f"Local SHAP — {selected_sample}")
        st.plotly_chart(fig_local, use_container_width=True)

        st.dataframe(
            sample_df[["feature", "shap_value", "direction", "transformed_value"]]
            .rename(columns={
                "shap_value": "SHAP Value",
                "direction": "Direction",
                "transformed_value": "Preprocessed Value",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            "**How to read local SHAP values:**  \n"
            "- Red bars (positive SHAP) → this feature value pushed the predicted "
            "  log-odds **upward** relative to the average employee.  \n"
            "- Blue bars (negative SHAP) → this feature value pushed the prediction "
            "  **downward** relative to the average.  \n"
            "- The SHAP values are in log-odds space (not probability space).  \n"
            "- These are associations from synthetic data — **not causal claims**."
        )

        st.markdown("---")
        st.caption(
            "⚠ **Association ≠ Causation**: A positive SHAP value for a feature "
            "does NOT mean that feature causes the employee to leave. "
            "It means that, for this specific employee, that feature's value "
            "shifted the model's prediction above the population average. "
            "These patterns are from synthetic data only."
        )
