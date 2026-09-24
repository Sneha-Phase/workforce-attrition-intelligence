"""
pages/page_about.py
====================
Responsible Use / About — Page 7.

Comprehensive responsible-AI documentation:
- Synthetic dataset limitation
- Model performance limitation
- Association ≠ causation
- Risk score as analytical signal
- Human review requirement
- No adverse action policy
"""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render() -> None:
    """Render the Responsible Use / About page."""

    st.title("📋 Responsible Use & About")
    st.caption(
        "Important information about how this tool should (and should not) be used."
    )

    # -----------------------------------------------------------------------
    # Hero notice
    # -----------------------------------------------------------------------
    st.error(
        "🚫 **This tool must not be used to fire, demote, penalise, or take "
        "any adverse employment action against any individual employee.** "
        "Risk scores are exploratory analytical signals only.",
        icon="🚫",
    )

    # -----------------------------------------------------------------------
    # Tabs for organised content
    # -----------------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "⚠ Limitations",
        "📖 About the Model",
        "🧭 Responsible Use Guide",
        "🔬 Methodology",
        "ℹ About This Project",
    ])

    # -------------------------------------------------------------------
    # Tab 1 — Limitations
    # -------------------------------------------------------------------
    with tab1:
        st.markdown("### 1. Synthetic Dataset Limitation")
        st.warning(
            "**All data in this dashboard is computer-generated (synthetic).** "
            "The dataset (`employee_attrition_dataset_10000.csv`) was algorithmically "
            "created with 10,000 simulated employee records. It does not represent "
            "any real organisation, any real employees, or any real workforce population. "
            "All patterns, correlations, and statistics are artefacts of the "
            "data-generation process.",
            icon="⚠️",
        )
        st.markdown(
            "**Implications:**\n"
            "- Feature importance rankings may not reflect real-world driver of attrition.\n"
            "- Departments, roles, and salaries are simulated and have no connection to "
            "any real company.\n"
            "- SHAP values and coefficient patterns are learned from synthetic noise.\n"
            "- The tool should be treated as a **learning exercise** in HR analytics "
            "methodology, not as an operational decision-support tool."
        )

        st.markdown("---")
        st.markdown("### 2. Model Performance Limitation")
        st.error(
            "**The selected model has very limited discriminative ability.**\n\n"
            "- Cross-validation PR-AUC ≈ 0.199 (class-prevalence baseline ≈ 0.200)\n"
            "- Test-set ROC-AUC ≈ 0.504 (random classifier = 0.500)\n"
            "- At the F1-optimal threshold, ~99% of employees are flagged as "
            "'high risk', making individual discrimination practically impossible.\n\n"
            "**A model this close to the random baseline cannot reliably identify "
            "which specific individuals are more likely to leave.**",
            icon="⚠️",
        )
        st.markdown(
            "This outcome is expected and appropriate for synthetic data. "
            "The dataset was not designed with a realistic attrition signal, "
            "so no model can achieve meaningful discrimination. "
            "The project demonstrates the *methodology* of HR attrition analytics — "
            "not a deployable prediction system."
        )

        st.markdown("---")
        st.markdown("### 3. Known Data Quality Issue")
        st.info(
            "The dataset contains a logical anomaly: `Years_in_Current_Role > Years_at_Company` "
            "for ~22.84% of employees. This is impossible (role tenure cannot exceed company tenure). "
            "The project handles this anomaly by flagging it (`Role_Tenure_Anomaly`) and clamping "
            "the value rather than silently correcting it. This is documented throughout the codebase."
        )

    # -------------------------------------------------------------------
    # Tab 2 — About the Model
    # -------------------------------------------------------------------
    with tab2:
        st.markdown("### Model Architecture")
        st.markdown(
            "The selected model is a **Logistic Regression** with balanced class weights "
            "(`class_weight='balanced'`), trained on the `clamp_and_flag` anomaly treatment. "
            "It was selected after 5-fold stratified cross-validation across 22 combinations "
            "of:\n"
            "- 8 model types (LR, LR_balanced, RF, RF_balanced, GB, XGB, XGB_balanced, RF_SMOTE)\n"
            "- 3 anomaly treatments (flag_only, clamp_and_flag, exclude)\n"
            "  *(RF_SMOTE evaluated with flag_only only)*\n\n"
            "The winning configuration was selected by highest mean PR-AUC, with simplicity "
            "as a tiebreaker."
        )

        st.markdown("### Why Logistic Regression?")
        st.markdown(
            "- **Interpretability**: Coefficients are directly interpretable as "
            "  log-odds contributors.\n"
            "- **SHAP compatibility**: SHAP `LinearExplainer` provides exact Shapley "
            "  values for linear models.\n"
            "- **Comparable performance**: All 21 model combinations produced similar "
            "  PR-AUC values near the baseline, so the simplest model was preferred.\n"
            "- **No evidence of benefit from complexity**: More complex models (RF, XGB) "
            "  did not consistently outperform LR on this synthetic dataset."
        )

        st.markdown("### Preprocessing Pipeline")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                "**Nominal categorical** (OneHotEncoder, drop=first):\n"
                "- Department\n- Job_Role\n- Marital_Status\n\n"
                "**Binary categorical** (OrdinalEncoder):\n"
                "- Gender (Female=0, Male=1)\n- Overtime (No=0, Yes=1)"
            )
        with col2:
            st.markdown(
                "**Ordinal/pass-through** (no scaling):\n"
                "- Job_Level, Satisfaction scales, Role_Tenure_Anomaly\n\n"
                "**Continuous** (StandardScaler):\n"
                "- Age, Income, Tenure fields, Hours, etc."
            )

        st.markdown("### Engineered Features")
        st.markdown(
            "Five features were engineered from raw inputs:\n\n"
            "| Feature | Formula | Notes |\n"
            "|---|---|---|\n"
            "| Role_Tenure_Anomaly | `Years_in_Current_Role > Years_at_Company` | Binary anomaly flag |\n"
            "| Satisfaction_Composite | Mean of 5 Likert scales | Well-being index |\n"
            "| Tenure_Promotion_Ratio | `Years_at_Company / (Years_Since_Last_Promotion + 1)` | Stagnation proxy |\n"
            "| Hours_Overload | `Avg_Hours - 40` | Deviation from 40-hr standard |\n"
            "| Income_to_Role_Ratio | `Monthly_Income / Job_Level` | Compensation relative to level |"
        )

    # -------------------------------------------------------------------
    # Tab 3 — Responsible Use Guide
    # -------------------------------------------------------------------
    with tab3:
        st.markdown("### What This Tool Is")
        st.success(
            "✅ An **educational demonstration** of HR analytics methodology.  \n"
            "✅ A **portfolio project** showing data science skills.  \n"
            "✅ A **learning resource** for understanding attrition modelling.  \n"
            "✅ An **illustration** of responsible AI practices in HR contexts."
        )

        st.markdown("### What This Tool Is Not")
        st.error(
            "🚫 NOT a deployment-ready HR decision tool.  \n"
            "🚫 NOT a system for identifying employees who should be fired or penalised.  \n"
            "🚫 NOT a replacement for human judgment, 1-on-1 conversations, or "
            "exit interview data.  \n"
            "🚫 NOT applicable to any real organisation without proper validation "
            "on that organisation's own data."
        )

        st.markdown("### Association ≠ Causation")
        st.info(
            "**Critical reminder:** Every feature importance value, SHAP value, "
            "and attrition rate shown in this dashboard reflects a **statistical "
            "association** in the synthetic training data.  \n\n"
            "Association means: in this dataset, employees with characteristic X "
            "were more/less frequently observed to have left.  \n\n"
            "**Causation** would require controlled experiments, natural experiments, "
            "or longitudinal studies specifically designed to test causal mechanisms. "
            "None of that evidence exists here — this is a cross-sectional synthetic dataset.  \n\n"
            "Do not say or imply that any feature *causes* attrition based on these outputs."
        )

        st.markdown("### Human Review Is Always Required")
        st.warning(
            "Any analytical output from this tool — including risk scores — must be "
            "reviewed by qualified HR professionals with:  \n"
            "- Full context about the individual employee.  \n"
            "- Knowledge of the team, manager, and organisational environment.  \n"
            "- Understanding of legal and ethical obligations to employees.  \n"
            "- Awareness of recency bias, survivorship bias, and other confounders.  \n\n"
            "A risk score is one potential data point among many — never a decision."
        )

        st.markdown("### No Adverse Action Policy")
        st.error(
            "**This tool must NEVER be used to:**  \n"
            "- Fire or terminate an employee based on their risk score.  \n"
            "- Demote, reassign, or reduce compensation based on a risk signal.  \n"
            "- Deny promotions or development opportunities based on predictions.  \n"
            "- Create a hostile or surveilled work environment.  \n"
            "- Discriminate against any protected class (the model was not validated "
            "  for fairness across demographic groups).  \n\n"
            "Violation of these principles is unethical and may be illegal in many jurisdictions."
        )

    # -------------------------------------------------------------------
    # Tab 4 — Methodology
    # -------------------------------------------------------------------
    with tab4:
        st.markdown("### Project Phases")
        phases = {
            "Phase 1 — Data Profiling": (
                "Loaded and validated the 10,000-row synthetic CSV. "
                "Identified the 22.84% role-tenure anomaly. "
                "Profiled class imbalance (~20% attrition)."
            ),
            "Phase 2 — Feature Engineering": (
                "Engineered 5 new features (Role_Tenure_Anomaly, Satisfaction_Composite, "
                "Tenure_Promotion_Ratio, Hours_Overload, Income_to_Role_Ratio). "
                "Applied clamp_and_flag anomaly treatment. "
                "Built preprocessing pipeline (OHE, OrdinalEncoder, StandardScaler)."
            ),
            "Phase 3 — Model Selection": (
                "5-fold stratified CV across 22 model+treatment combinations. "
                "Selected LR_balanced + clamp_and_flag by PR-AUC. "
                "Performed threshold analysis. "
                "Evaluated on held-out test set (20%)."
            ),
            "Phase 4 — Explainability": (
                "Computed coefficient-based global importance. "
                "Built SHAP LinearExplainer on training background. "
                "Computed SHAP global and local explanations. "
                "Generated sample explanations for 5 representative employees."
            ),
            "Phase 5 — Dashboard": (
                "Built the Streamlit multi-page application. "
                "Integrated all artifacts with caching. "
                "Added comprehensive responsible-use disclaimers throughout. "
                "Included interactive Plotly charts and the Risk Simulator. "
                "Fixed Years_in_Current_Role_Clamped regression in the Risk Simulator."
            ),
            "Phase 6 — Final Dashboard Validation & Portfolio Polish": (
                "End-to-end review and validation of all dashboard pages. "
                "Fixed broken f-string in Model Performance page. "
                "Corrected model count (22 combinations, 8 model types) throughout. "
                "Updated phase label from Phase 5 to Phase 6 in sidebar. "
                "Added this comprehensive README.md and portfolio documentation."
            ),
        }
        for phase, desc in phases.items():
            with st.expander(phase, expanded=False):
                st.markdown(desc)

        st.markdown("### Technology Stack")
        tech = {
            "Data": "pandas, numpy",
            "Modelling": "scikit-learn, xgboost, imbalanced-learn",
            "Explainability": "shap (LinearExplainer)",
            "Visualisation": "plotly",
            "Dashboard": "streamlit",
            "Testing": "pytest",
            "Model persistence": "joblib",
        }
        tech_df = pd.DataFrame(list(tech.items()), columns=["Category", "Library"])
        st.dataframe(tech_df, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------
    # Tab 5 — About This Project
    # -------------------------------------------------------------------
    with tab5:
        st.markdown("### Workforce Attrition Intelligence & Retention Analytics Platform")
        st.markdown(
            "This is a portfolio-grade, end-to-end data science project demonstrating "
            "responsible HR analytics. The project was built in five phases:\n\n"
            "1. **Data Profiling** — Understanding the synthetic dataset.\n"
            "2. **Feature Engineering** — Creating informative features.\n"
            "3. **Model Selection** — Rigorous cross-validation and evaluation.\n"
            "4. **Explainability** — SHAP and coefficient-based interpretation.\n"
            "5. **Dashboard** — This Streamlit application.\n\n"
            "The project explicitly acknowledges and communicates:\n"
            "- The synthetic data limitation.\n"
            "- The model's near-baseline performance.\n"
            "- The distinction between association and causation.\n"
            "- The ethical constraints on using risk signals in employment decisions.\n\n"
            "**Purpose:** Education, portfolio demonstration, and responsible AI advocacy."
        )

        st.markdown("---")
        st.markdown("### Dataset Information")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                "**File:** `data/employee_attrition_dataset_10000.csv`  \n"
                "**Records:** 10,000 synthetic employees  \n"
                "**Features:** 26 columns  \n"
                "**Target:** `Attrition` (Yes/No)  \n"
                "**Class balance:** ~80% No / ~20% Yes"
            )
        with col2:
            st.markdown(
                "**Split:** 80% train / 20% test (stratified)  \n"
                "**Anomaly:** 22.84% role-tenure anomaly  \n"
                "**Missing values:** None  \n"
                "**Data source:** Synthetic generation"
            )

        st.markdown("---")
        st.caption(
            "This project was developed as part of an AICTE internship in "
            "Data Analytics. All data is synthetic. No real employee data was used."
        )
