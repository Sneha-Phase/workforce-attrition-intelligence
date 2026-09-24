"""
pages/page_simulator.py
========================
Attrition Risk Simulator — Page 6.

Allows the user to enter a hypothetical employee profile and generates
a predicted attrition probability using the saved pipeline.

Output is presented as an ANALYTICAL RISK SIGNAL, not a prediction
that a real person will leave. Multiple disclaimers prevent misuse.

IMPORTANT ETHICAL CONSTRAINTS (enforced in UI and copy):
- Do NOT recommend firing, demoting, or penalising any employee.
- Do NOT present the score as a certainty.
- Do NOT hide the model's limited performance.
- Require explicit user acknowledgement before showing results.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.dashboard_utils import (
    ASSOCIATION_DISCLAIMER,
    MODEL_LIMITATION_DISCLAIMER,
    NO_ADVERSE_ACTION_DISCLAIMER,
    RISK_SIGNAL_DISCLAIMER,
    SYNTHETIC_DISCLAIMER,
    build_employee_row,
    local_shap_waterfall,
    load_cv_results,
    load_explainer,
    load_pipeline,
    risk_color,
    risk_emoji,
)


# ---------------------------------------------------------------------------
# Categorical choice lists (match pipeline's OHE categories)
# ---------------------------------------------------------------------------
_DEPARTMENTS   = ["Finance", "HR", "IT", "Marketing", "Sales"]
_JOB_ROLES     = ["Analyst", "Assistant", "Executive", "Manager"]
_MARITAL       = ["Divorced", "Married", "Single"]
_GENDERS       = ["Female", "Male"]
_OVERTIME      = ["No", "Yes"]


def _build_sidebar_form() -> dict:
    """Render the employee profile input widgets in the sidebar and return values."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Simulator Profile")

    with st.sidebar.expander("Demographics", expanded=True):
        age = st.slider("Age", 18, 65, 35, key="sim_age")
        gender = st.selectbox("Gender", _GENDERS, key="sim_gender")
        marital = st.selectbox("Marital Status", _MARITAL, key="sim_marital")
        distance = st.slider("Distance From Home (km)", 1, 50, 10, key="sim_dist")
        n_companies = st.slider("Previous Employers (count)", 1, 9, 2, key="sim_ncomp")

    with st.sidebar.expander("Job Details", expanded=True):
        dept = st.selectbox("Department", _DEPARTMENTS, key="sim_dept")
        role = st.selectbox("Job Role", _JOB_ROLES, key="sim_role")
        job_level = st.slider("Job Level (1–5)", 1, 5, 2, key="sim_jlevel")
        overtime = st.selectbox("Overtime", _OVERTIME, key="sim_ot")

    with st.sidebar.expander("Tenure & Compensation", expanded=True):
        years_company = st.slider("Years at Company", 0, 30, 5, key="sim_yrsco")
        years_role = st.slider("Years in Current Role", 0, 30, 3, key="sim_yrsrole")
        years_promo = st.slider("Years Since Last Promotion", 0, 15, 2, key="sim_yrspromo")
        monthly_income = st.slider("Monthly Income ($)", 2000, 25000, 8000, step=500, key="sim_income")
        hourly_rate = st.slider("Hourly Rate ($)", 10, 120, 40, key="sim_hrly")

    with st.sidebar.expander("Performance & Engagement", expanded=True):
        perf_rating = st.slider("Performance Rating (1–4)", 1, 4, 3, key="sim_perf")
        job_sat = st.slider("Job Satisfaction (1–5)", 1, 5, 3, key="sim_jobsat")
        wlb = st.slider("Work-Life Balance (1–4)", 1, 4, 3, key="sim_wlb")
        env_sat = st.slider("Work Env. Satisfaction (1–4)", 1, 4, 3, key="sim_envsat")
        rel_mgr = st.slider("Relationship w/ Manager (1–4)", 1, 4, 3, key="sim_relmgr")
        job_inv = st.slider("Job Involvement (1–4)", 1, 4, 3, key="sim_jobinv")
        project_count = st.slider("Project Count", 1, 10, 3, key="sim_proj")
        training_hrs = st.slider("Training Hours Last Year", 0, 100, 20, key="sim_train")
        avg_hrs = st.slider("Avg Hours Worked / Week", 30, 60, 42, key="sim_avghrs")
        absenteeism = st.slider("Absenteeism (days/year)", 0, 30, 3, key="sim_absent")

    return {
        "Age": age,
        "Gender": gender,
        "Marital_Status": marital,
        "Department": dept,
        "Job_Role": role,
        "Job_Level": job_level,
        "Monthly_Income": monthly_income,
        "Hourly_Rate": hourly_rate,
        "Years_at_Company": years_company,
        "Years_in_Current_Role": years_role,
        "Years_Since_Last_Promotion": years_promo,
        "Work_Life_Balance": wlb,
        "Job_Satisfaction": job_sat,
        "Performance_Rating": perf_rating,
        "Training_Hours_Last_Year": training_hrs,
        "Overtime": overtime,
        "Project_Count": project_count,
        "Average_Hours_Worked_Per_Week": avg_hrs,
        "Absenteeism": absenteeism,
        "Work_Environment_Satisfaction": env_sat,
        "Relationship_with_Manager": rel_mgr,
        "Job_Involvement": job_inv,
        "Distance_From_Home": distance,
        "Number_of_Companies_Worked": n_companies,
    }


def render() -> None:
    """Render the Attrition Risk Simulator page."""

    st.title("⚙️ Attrition Risk Simulator")
    st.caption(
        "Enter a hypothetical employee profile to generate an attrition risk signal "
        "using the trained pipeline. Read all disclaimers before proceeding."
    )

    # -----------------------------------------------------------------------
    # Mandatory disclaimers
    # -----------------------------------------------------------------------
    st.error(NO_ADVERSE_ACTION_DISCLAIMER, icon="🚫")
    st.warning(RISK_SIGNAL_DISCLAIMER)
    st.markdown(
        f'<div class="disclaimer-box">{MODEL_LIMITATION_DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="disclaimer-box">{SYNTHETIC_DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------------
    # Acknowledgement gate
    # -----------------------------------------------------------------------
    st.markdown("---")
    acknowledged = st.checkbox(
        "I understand that this is an analytical tool trained on synthetic data, "
        "the model has very limited discriminative ability (PR-AUC ≈ 0.20), "
        "risk scores are not predictions that an employee will leave, "
        "and I will not use this tool to take adverse action against any individual.",
        value=False,
        key="sim_ack",
    )

    if not acknowledged:
        st.info(
            "Please check the acknowledgement box above to enable the simulator. "
            "The simulator is for educational and analytical exploration only."
        )
        _build_sidebar_form()  # still render form so it's visible
        return

    # -----------------------------------------------------------------------
    # Profile form (sidebar)
    # -----------------------------------------------------------------------
    profile = _build_sidebar_form()

    # -----------------------------------------------------------------------
    # Load model artifacts
    # -----------------------------------------------------------------------
    st.markdown("---")
    with st.spinner("Loading model…"):
        try:
            pipeline = load_pipeline()
            explainer = load_explainer()
            cv = load_cv_results()
            optimal_threshold = cv.get("optimal_threshold", 0.40)
        except Exception as e:
            st.error(f"Could not load model artifacts: {e}")
            return

    # -----------------------------------------------------------------------
    # Run prediction on button click
    # -----------------------------------------------------------------------
    run_btn = st.button("🔍 Compute Risk Signal", type="primary", use_container_width=True)

    if not run_btn:
        st.info(
            "Configure the hypothetical employee profile in the sidebar, "
            "then click **Compute Risk Signal** to generate the analytical output."
        )
        return

    # Build feature row
    try:
        X_row = build_employee_row(profile)
    except Exception as e:
        st.error(f"Error building feature row: {e}")
        return

    # Generate prediction and explanation
    try:
        from src.explainability.prediction_explainer import explain_prediction
        result = explain_prediction(
            pipeline,
            explainer,
            X_row,
            optimal_threshold=optimal_threshold,
            top_n=15,
        )
    except Exception as e:
        st.error(f"Error running prediction: {e}")
        return

    prob  = result.attrition_probability
    label = result.risk_label
    color = risk_color(label)
    emoji = risk_emoji(label)

    # -----------------------------------------------------------------------
    # Risk result display
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📊 Risk Signal Output")

    r1, r2, r3 = st.columns(3)
    r1.metric(
        "Predicted Probability",
        f"{prob:.1%}",
        help=(
            "The model's estimated probability of attrition for this profile. "
            f"Class prevalence ≈ {0.1997:.0%}. "
            "Model PR-AUC ≈ 0.20 — very limited discrimination."
        ),
    )
    r2.metric(
        "Risk Tier",
        f"{emoji} {label}",
        help=(
            "High Risk: prob ≥ optimal threshold. "
            "Moderate: 0.20 ≤ prob < threshold. "
            "Low Risk: prob < 0.20."
        ),
    )
    r3.metric(
        "Optimal Threshold",
        f"{optimal_threshold:.4f}",
        help="F1-optimal threshold from Phase 3 CV analysis.",
    )

    # Coloured risk banner
    st.markdown(
        f"""
        <div style="
            background-color: {color}20;
            border-left: 5px solid {color};
            padding: 1rem;
            border-radius: 4px;
            margin: 0.5rem 0;
        ">
            <strong style="color:{color}; font-size:1.1rem;">{emoji} {label}</strong> —
            Predicted attrition probability: <strong>{prob:.1%}</strong><br>
            <small>This is an <em>analytical risk signal</em> only.
            The model cannot reliably predict whether any individual employee will leave.
            Human review and contextual judgment are always required.</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------------
    # SHAP local explanation chart
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🔍 Local SHAP Explanation")
    st.caption(
        "The chart below shows which features pushed this employee's predicted "
        "probability **above** (red, positive SHAP) or **below** (blue, negative SHAP) "
        "the population baseline. These are statistical associations — not causes."
    )

    try:
        fig_local = local_shap_waterfall(
            result.local_shap_df,
            base_value=result.base_value,
            predicted_prob=prob,
            title="Local SHAP — Hypothetical Employee Profile",
            top_n=15,
        )
        st.plotly_chart(fig_local, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render SHAP chart: {e}")

    # SHAP table
    with st.expander("📋 Full SHAP explanation table", expanded=False):
        shap_display = result.local_shap_df.rename(columns={
            "shap_value": "SHAP Value",
            "abs_shap": "|SHAP|",
            "direction": "Direction",
            "transformed_value": "Preprocessed Value",
        })
        st.dataframe(shap_display, use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------------
    # Input profile summary
    # -----------------------------------------------------------------------
    with st.expander("📝 Input Profile Summary", expanded=False):
        profile_df = pd.DataFrame(list(profile.items()), columns=["Feature", "Value"])
        st.dataframe(profile_df, use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------------
    # Disclaimers footer
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown(
        f'<div class="disclaimer-box">{ASSOCIATION_DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="disclaimer-box disclaimer-danger">{NO_ADVERSE_ACTION_DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Base value (population mean log-odds): "
        f"{result.base_value:.4f}. "
        "SHAP values are additive log-odds contributions relative to this baseline. "
        "All figures are from a model trained on synthetic data."
    )
