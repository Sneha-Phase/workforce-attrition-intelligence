"""
pages/page_executive.py
========================
Executive HR Dashboard — Page 1.

Shows top-level workforce KPIs, attrition overview,
department/role summaries, and a prominent synthetic-data disclaimer.

All figures are descriptive statistics from the dataset.
No predictions are presented as facts.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.dashboard_utils import (
    ASSOCIATION_DISCLAIMER,
    MODEL_LIMITATION_DISCLAIMER,
    SYNTHETIC_DISCLAIMER,
    compute_attrition_summary,
    fmt_pct,
    load_dataset,
    pie_chart,
)


def render() -> None:
    """Render the Executive HR Dashboard page."""

    st.title("📊 Executive HR Dashboard")
    st.caption(
        "High-level workforce overview based on the synthetic attrition dataset. "
        "All figures are descriptive statistics — not predictions."
    )

    # -----------------------------------------------------------------------
    # Prominent synthetic-data disclaimer
    # -----------------------------------------------------------------------
    st.markdown(
        f'<div class="disclaimer-box">{SYNTHETIC_DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="disclaimer-box">{MODEL_LIMITATION_DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    try:
        df = load_dataset()
    except Exception as e:
        st.error(f"Could not load dataset: {e}")
        return

    summary = compute_attrition_summary(df)

    # -----------------------------------------------------------------------
    # Top KPI row
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown('<div class="section-header">Workforce KPIs</div>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Workforce", f"{summary['total_employees']:,}")
    col2.metric("Attrition Count", f"{summary['attrition_count']:,}")
    col3.metric("Retained", f"{summary['retained_count']:,}")
    col4.metric(
        "Attrition Rate",
        fmt_pct(summary["attrition_rate"]),
        delta=None,
        help="Proportion of employees recorded as having left.",
    )
    col5.metric(
        "Retention Rate",
        fmt_pct(summary["retention_rate"]),
        help="Proportion of employees recorded as retained.",
    )

    st.caption(
        "These figures are from the synthetic dataset and do not represent "
        "any real organisation's workforce."
    )

    # -----------------------------------------------------------------------
    # Attrition vs Retained breakdown
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown('<div class="section-header">Workforce Composition</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        fig_attrition_pie = pie_chart(
            labels=["Retained", "Attrited"],
            values=[summary["retained_count"], summary["attrition_count"]],
            title="Attrition vs Retained (Overall)",
            colors=["#3498db", "#e74c3c"],
        )
        st.plotly_chart(fig_attrition_pie, use_container_width=True)

    with c2:
        dept_counts = df.groupby("Department")["Attrition"].count().reset_index()
        dept_counts.columns = ["Department", "Count"]
        fig_dept_pie = px.pie(
            dept_counts,
            names="Department",
            values="Count",
            title="Workforce by Department",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_dept_pie.update_layout(height=350, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_dept_pie, use_container_width=True)

    # -----------------------------------------------------------------------
    # Department attrition summary table
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown(
        '<div class="section-header">Attrition by Department (Descriptive)</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Shows observed attrition rates per department in this synthetic dataset. "
        "These are associations — not causal claims."
    )

    dept_summary = (
        df.groupby("Department")
        .agg(
            Total=("Attrition", "count"),
            Attrited=("Attrition", lambda s: (s == "Yes").sum()),
        )
        .reset_index()
    )
    dept_summary["Retained"] = dept_summary["Total"] - dept_summary["Attrited"]
    dept_summary["Attrition Rate"] = (dept_summary["Attrited"] / dept_summary["Total"]).apply(fmt_pct)
    dept_summary = dept_summary.sort_values("Attrited", ascending=False).reset_index(drop=True)

    st.dataframe(dept_summary, use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------------
    # Job Role summary
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown(
        '<div class="section-header">Workforce by Job Role</div>',
        unsafe_allow_html=True,
    )

    role_counts = (
        df.groupby("Job_Role")["Attrition"]
        .agg(["count", lambda s: (s == "Yes").sum()])
        .reset_index()
    )
    role_counts.columns = ["Job Role", "Total", "Attrited"]
    role_counts["Retention Rate"] = ((role_counts["Total"] - role_counts["Attrited"]) / role_counts["Total"]).apply(fmt_pct)
    role_counts = role_counts.sort_values("Total", ascending=False).reset_index(drop=True)

    fig_role = px.bar(
        role_counts,
        x="Job Role",
        y="Total",
        color="Attrited",
        title="Headcount by Job Role (coloured by attrition count)",
        labels={"Total": "Total Employees", "Attrited": "Attrited"},
        color_continuous_scale="Reds",
        text="Total",
    )
    fig_role.update_traces(textposition="outside")
    fig_role.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=50, b=60),
        coloraxis_colorbar_title="Attrited",
    )
    st.plotly_chart(fig_role, use_container_width=True)

    # -----------------------------------------------------------------------
    # Key workforce stats
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown(
        '<div class="section-header">Key Workforce Statistics</div>',
        unsafe_allow_html=True,
    )

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Median Age", f"{df['Age'].median():.0f} yrs")
    s2.metric("Median Tenure", f"{df['Years_at_Company'].median():.0f} yrs")
    s3.metric("Median Monthly Income", f"${df['Monthly_Income'].median():,.0f}")
    s4.metric("Overtime Proportion", fmt_pct((df["Overtime"] == "Yes").mean()))

    s5, s6, s7, s8 = st.columns(4)
    s5.metric("Avg Job Satisfaction", f"{df['Job_Satisfaction'].mean():.2f} / 5")
    s6.metric("Avg Work-Life Balance", f"{df['Work_Life_Balance'].mean():.2f} / 4")
    s7.metric("Avg Relationship w/ Manager", f"{df['Relationship_with_Manager'].mean():.2f} / 4")
    s8.metric("Avg Job Involvement", f"{df['Job_Involvement'].mean():.2f} / 4")

    # -----------------------------------------------------------------------
    # Footer disclaimer
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.info(
        "**Note:** All statistics on this page are purely descriptive. "
        "They describe patterns observed in the synthetic dataset and should "
        "not be interpreted as predictions or as evidence of causation."
    )
