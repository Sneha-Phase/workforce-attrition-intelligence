"""
pages/page_workforce.py
========================
Workforce Analytics — Page 2.

Interactive distributions for all major workforce variables.
Uses Plotly charts throughout.  No predictions are shown here.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.dashboard_utils import (
    SYNTHETIC_DISCLAIMER,
    histogram_chart,
    load_dataset,
    pie_chart,
)


_NUMERIC_COLS = [
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

_SATISFACTION_COLS = [
    "Job_Satisfaction",
    "Work_Life_Balance",
    "Work_Environment_Satisfaction",
    "Relationship_with_Manager",
    "Job_Involvement",
    "Performance_Rating",
]

_CATEGORICAL_COLS = [
    "Department",
    "Job_Role",
    "Job_Level",
    "Gender",
    "Marital_Status",
    "Overtime",
]


def render() -> None:
    """Render the Workforce Analytics page."""

    st.title("👥 Workforce Analytics")
    st.caption(
        "Descriptive distributions for the synthetic workforce dataset. "
        "Use the tabs below to explore different dimensions."
    )

    st.markdown(
        f'<div class="disclaimer-box">{SYNTHETIC_DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )

    try:
        df = load_dataset()
    except Exception as e:
        st.error(f"Could not load dataset: {e}")
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "📐 Demographics & Tenure",
        "💰 Income & Work Patterns",
        "😊 Satisfaction Scales",
        "🗂 Categorical Breakdowns",
    ])

    # -------------------------------------------------------------------
    # Tab 1 — Demographics & Tenure
    # -------------------------------------------------------------------
    with tab1:
        st.markdown("#### Age Distribution")
        c1, c2 = st.columns(2)
        with c1:
            fig = histogram_chart(df, "Age", "Age Distribution", color_col="Attrition", nbins=25)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = histogram_chart(df, "Years_at_Company", "Years at Company", color_col="Attrition", nbins=20)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Role & Promotion Tenure")
        c3, c4 = st.columns(2)
        with c3:
            fig = histogram_chart(df, "Years_in_Current_Role", "Years in Current Role", color_col="Attrition", nbins=20)
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            fig = histogram_chart(df, "Years_Since_Last_Promotion", "Years Since Last Promotion", color_col="Attrition", nbins=15)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Distance From Home")
        fig = histogram_chart(df, "Distance_From_Home", "Distance From Home (km/miles)", color_col="Attrition", nbins=25)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Number of Companies Worked")
        fig = histogram_chart(df, "Number_of_Companies_Worked", "Number of Previous Employers", color_col="Attrition", nbins=10)
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Blue = No Attrition, Red = Attrition (overlaid histograms). "
            "These are descriptive distributions — not causal relationships."
        )

    # -------------------------------------------------------------------
    # Tab 2 — Income & Work Patterns
    # -------------------------------------------------------------------
    with tab2:
        st.markdown("#### Monthly Income")
        c1, c2 = st.columns(2)
        with c1:
            fig = histogram_chart(df, "Monthly_Income", "Monthly Income Distribution", color_col="Attrition", nbins=30)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.box(
                df,
                x="Department",
                y="Monthly_Income",
                color="Attrition",
                color_discrete_map={"Yes": "#e74c3c", "No": "#3498db"},
                title="Monthly Income by Department",
                points=False,
            )
            fig.update_layout(height=350, margin=dict(l=10, r=10, t=50, b=60))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Working Hours & Overtime")
        c3, c4 = st.columns(2)
        with c3:
            fig = histogram_chart(
                df, "Average_Hours_Worked_Per_Week",
                "Average Hours Worked Per Week",
                color_col="Attrition", nbins=20,
            )
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            overtime_counts = (
                df.groupby(["Overtime", "Attrition"]).size().reset_index(name="Count")
            )
            fig = px.bar(
                overtime_counts,
                x="Overtime",
                y="Count",
                color="Attrition",
                barmode="group",
                color_discrete_map={"Yes": "#e74c3c", "No": "#3498db"},
                title="Overtime vs Attrition Count",
            )
            fig.update_layout(height=350, margin=dict(l=10, r=10, t=50, b=40))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Training & Project Load")
        c5, c6 = st.columns(2)
        with c5:
            fig = histogram_chart(df, "Training_Hours_Last_Year", "Training Hours (Last Year)", color_col="Attrition", nbins=20)
            st.plotly_chart(fig, use_container_width=True)
        with c6:
            fig = histogram_chart(df, "Project_Count", "Project Count", color_col="Attrition", nbins=10)
            st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Income and work-pattern distributions across the synthetic workforce. "
            "Colour indicates attrition status (descriptive, not predictive)."
        )

    # -------------------------------------------------------------------
    # Tab 3 — Satisfaction Scales
    # -------------------------------------------------------------------
    with tab3:
        st.markdown(
            "#### Likert-Scale Well-being Indicators  \n"
            "_These are self-reported satisfaction scales in the synthetic dataset._"
        )

        # Radar chart: mean satisfaction by attrition group
        mean_yes = df[df["Attrition"] == "Yes"][_SATISFACTION_COLS].mean()
        mean_no  = df[df["Attrition"] == "No"][_SATISFACTION_COLS].mean()
        cats_clean = [c.replace("_", " ") for c in _SATISFACTION_COLS]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=mean_yes.values.tolist() + [mean_yes.values[0]],
            theta=cats_clean + [cats_clean[0]],
            fill="toself",
            name="Attrition = Yes",
            line_color="#e74c3c",
            opacity=0.6,
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=mean_no.values.tolist() + [mean_no.values[0]],
            theta=cats_clean + [cats_clean[0]],
            fill="toself",
            name="Attrition = No",
            line_color="#3498db",
            opacity=0.6,
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[1, 5])),
            title="Mean Satisfaction Scales by Attrition Group",
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.caption(
            "⚠ The radar chart shows **mean values by group**, not individual predictions. "
            "Differences between groups in this synthetic dataset do not imply causation."
        )

        st.markdown("---")

        # Individual distributions
        for col in _SATISFACTION_COLS:
            val_counts = df.groupby([col, "Attrition"]).size().reset_index(name="Count")
            fig = px.bar(
                val_counts,
                x=col,
                y="Count",
                color="Attrition",
                barmode="group",
                color_discrete_map={"Yes": "#e74c3c", "No": "#3498db"},
                title=f"{col.replace('_', ' ')} Distribution",
            )
            fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=30))
            st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------
    # Tab 4 — Categorical Breakdowns
    # -------------------------------------------------------------------
    with tab4:
        st.markdown("#### Categorical Variable Distributions")

        for col in _CATEGORICAL_COLS:
            c1, c2 = st.columns([2, 1])
            with c1:
                val_counts = df.groupby([col, "Attrition"]).size().reset_index(name="Count")
                fig = px.bar(
                    val_counts,
                    x=col,
                    y="Count",
                    color="Attrition",
                    barmode="group",
                    color_discrete_map={"Yes": "#e74c3c", "No": "#3498db"},
                    title=f"{col.replace('_', ' ')} vs Attrition",
                )
                fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=50))
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                raw_counts = df[col].value_counts().reset_index()
                raw_counts.columns = [col, "Count"]
                st.dataframe(raw_counts, use_container_width=True, hide_index=True)

        st.caption(
            "All categorical breakdowns are descriptive statistics from the "
            "synthetic dataset. Colour indicates attrition status for context only."
        )
