"""
pages/page_attrition.py
========================
Attrition Analysis — Page 3.

Descriptive attrition rates across categorical and numeric groupings.
All associations are clearly labelled as descriptive; causation is not implied.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.dashboard_utils import (
    ASSOCIATION_DISCLAIMER,
    SYNTHETIC_DISCLAIMER,
    attrition_rate_by_group,
    bar_chart_attrition_by_group,
    load_dataset,
    fmt_pct,
)


def render() -> None:
    """Render the Attrition Analysis page."""

    st.title("📉 Attrition Analysis")
    st.caption(
        "Descriptive attrition associations across workforce segments. "
        "These figures describe patterns in the synthetic dataset — "
        "**they do not establish causation**."
    )

    st.markdown(
        f'<div class="disclaimer-box">{SYNTHETIC_DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="disclaimer-box">{ASSOCIATION_DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )

    try:
        df = load_dataset()
    except Exception as e:
        st.error(f"Could not load dataset: {e}")
        return

    tab1, tab2, tab3 = st.tabs([
        "🏢 Categorical Groups",
        "📊 Numeric Distributions",
        "🔗 Cross-tab Explorer",
    ])

    # -------------------------------------------------------------------
    # Tab 1 — Categorical Groups
    # -------------------------------------------------------------------
    with tab1:
        st.markdown(
            "#### Attrition Rate by Categorical Group  \n"
            "_Sorted by observed attrition rate (highest first). "
            "Differences are descriptive associations in the synthetic dataset._"
        )

        cat_cols = {
            "Department": "Department",
            "Job_Role": "Job Role",
            "Gender": "Gender",
            "Marital_Status": "Marital Status",
            "Overtime": "Overtime",
            "Job_Level": "Job Level",
            "Performance_Rating": "Performance Rating",
        }

        for col, label in cat_cols.items():
            with st.expander(f"Attrition Rate by {label}", expanded=(col == "Department")):
                fig = bar_chart_attrition_by_group(
                    df, col,
                    title=f"Observed Attrition Rate by {label} (descriptive association)",
                )
                st.plotly_chart(fig, use_container_width=True)

                rate_df = attrition_rate_by_group(df, col)
                rate_df["Attrition Rate"] = rate_df["attrition_rate"].apply(fmt_pct)
                st.dataframe(
                    rate_df[[col, "count", "attrition_count", "Attrition Rate"]].rename(
                        columns={"count": "Total", "attrition_count": "Attrited"}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(
                    f"⚠ The attrition rate difference across {label} values is a "
                    "descriptive association in this synthetic dataset. "
                    "It does not imply that this variable causes attrition."
                )

    # -------------------------------------------------------------------
    # Tab 2 — Numeric Distributions
    # -------------------------------------------------------------------
    with tab2:
        st.markdown(
            "#### Numeric Feature Distributions by Attrition Group  \n"
            "_Box plots compare the distribution of each numeric feature "
            "between employees who left and those who stayed._  \n"
            "_These are descriptive comparisons — not predictors._"
        )

        numeric_cols = [
            ("Age", "Age"),
            ("Monthly_Income", "Monthly Income"),
            ("Years_at_Company", "Years at Company"),
            ("Years_in_Current_Role", "Years in Current Role"),
            ("Years_Since_Last_Promotion", "Years Since Last Promotion"),
            ("Distance_From_Home", "Distance From Home"),
            ("Average_Hours_Worked_Per_Week", "Avg Hours Worked / Week"),
            ("Training_Hours_Last_Year", "Training Hours (Last Year)"),
            ("Absenteeism", "Absenteeism"),
            ("Project_Count", "Project Count"),
            ("Number_of_Companies_Worked", "No. of Previous Employers"),
        ]

        for col, label in numeric_cols:
            with st.expander(f"{label}", expanded=(label == "Age")):
                c1, c2 = st.columns([3, 1])
                with c1:
                    fig = px.box(
                        df,
                        x="Attrition",
                        y=col,
                        color="Attrition",
                        color_discrete_map={"Yes": "#e74c3c", "No": "#3498db"},
                        points="outliers",
                        title=f"{label} — Attrition vs Retained (descriptive)",
                        labels={"Attrition": "Attrition Status", col: label},
                    )
                    fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=30), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    stats = df.groupby("Attrition")[col].describe()[["mean", "50%", "std"]].T
                    stats.index = ["Mean", "Median", "Std Dev"]
                    st.dataframe(stats.round(2), use_container_width=True)

                st.caption(
                    f"⚠ Difference in {label} distributions between attrition groups "
                    "is a descriptive association, not a causal claim."
                )

    # -------------------------------------------------------------------
    # Tab 3 — Cross-tab Explorer
    # -------------------------------------------------------------------
    with tab3:
        st.markdown(
            "#### Cross-tabulation Explorer  \n"
            "_Select two categorical variables to see a heatmap of "
            "attrition rates._"
        )

        available_cats = [
            "Department", "Job_Role", "Gender", "Marital_Status",
            "Overtime", "Job_Level", "Performance_Rating", "Work_Life_Balance",
        ]

        c1, c2 = st.columns(2)
        with c1:
            x_col = st.selectbox("Row variable", available_cats, index=0, key="xtab_row")
        with c2:
            remaining = [c for c in available_cats if c != x_col]
            y_col = st.selectbox("Column variable", remaining, index=0, key="xtab_col")

        # Build attrition rate pivot
        ct = df.groupby([x_col, y_col]).agg(
            total=("Attrition", "count"),
            attrited=("Attrition", lambda s: (s == "Yes").sum()),
        ).reset_index()
        ct["rate"] = (ct["attrited"] / ct["total"] * 100).round(1)

        pivot = ct.pivot(index=x_col, columns=y_col, values="rate")

        fig_heat = px.imshow(
            pivot,
            title=f"Attrition Rate (%) — {x_col} × {y_col}",
            color_continuous_scale="Reds",
            text_auto=True,
            aspect="auto",
            labels={"color": "Attrition %"},
        )
        fig_heat.update_layout(
            height=max(350, len(pivot) * 50 + 100),
            margin=dict(l=10, r=10, t=60, b=80),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.caption(
            "⚠ Heatmap values are observed attrition percentages in the synthetic dataset. "
            "Higher percentages indicate a stronger **descriptive association** — "
            "**not** a causal relationship."
        )
