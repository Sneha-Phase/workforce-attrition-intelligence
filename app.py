"""
app.py
======
Workforce Attrition Intelligence & Retention Analytics Platform
Streamlit multi-page application — Phase 5.

Run locally:
    streamlit run app.py

Pages
-----
1. Executive HR Dashboard      — workforce overview and KPIs
2. Workforce Analytics         — distribution charts
3. Attrition Analysis          — descriptive attrition associations
4. Model Performance           — CV and test-set metrics
5. Explainable AI              — global + local SHAP explanations
6. Attrition Risk Simulator    — hypothetical employee risk signal
7. Responsible Use / About     — disclaimers and responsible AI guidance
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Workforce Attrition Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for a clean, professional look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global font */
    html, body, [class*="css"] {
        font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
    }

    /* Sidebar nav header */
    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1f2328;
        padding-bottom: 0.25rem;
        border-bottom: 2px solid #3b82d4;
        margin-bottom: 0.75rem;
    }

    /* KPI metric cards */
    div[data-testid="metric-container"] {
        background: #f7f8fa;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.6rem 1rem;
    }

    /* Disclaimer boxes */
    .disclaimer-box {
        background: #fff8e1;
        border-left: 4px solid #f39c12;
        padding: 0.6rem 1rem;
        border-radius: 0 4px 4px 0;
        font-size: 0.88rem;
        margin: 0.5rem 0;
    }

    /* Danger disclaimer */
    .disclaimer-danger {
        background: #fdecea;
        border-left: 4px solid #e74c3c;
    }

    /* Section headers */
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1f2328;
        margin-top: 1.25rem;
        margin-bottom: 0.5rem;
        padding-bottom: 0.25rem;
        border-bottom: 1px solid #e5e7eb;
    }

    /* Pill badge */
    .risk-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        color: white;
    }

    /* Footer */
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Page registry (order determines sidebar display)
# ---------------------------------------------------------------------------

PAGE_EXECUTIVE   = "📊 Executive HR Dashboard"
PAGE_WORKFORCE   = "👥 Workforce Analytics"
PAGE_ATTRITION   = "📉 Attrition Analysis"
PAGE_PERFORMANCE = "🎯 Model Performance"
PAGE_XAI         = "🔍 Explainable AI"
PAGE_SIMULATOR   = "⚙️ Attrition Risk Simulator"
PAGE_ABOUT       = "📋 Responsible Use / About"

PAGES = [
    PAGE_EXECUTIVE,
    PAGE_WORKFORCE,
    PAGE_ATTRITION,
    PAGE_PERFORMANCE,
    PAGE_XAI,
    PAGE_SIMULATOR,
    PAGE_ABOUT,
]

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div class="sidebar-title">Workforce Attrition Intelligence</div>',
        unsafe_allow_html=True,
    )
    st.caption("Retention Analytics Platform · Phase 6")
    st.divider()

    selected_page = st.radio(
        "Navigate to:",
        PAGES,
        index=0,
        label_visibility="collapsed",
    )

    st.divider()
    st.caption(
        "⚠ Synthetic dataset · Educational use only\n\n"
        "Model PR-AUC ~0.20 — near class-prevalence baseline.\n"
        "Predictions are analytical signals, not facts."
    )

# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------

if selected_page == PAGE_EXECUTIVE:
    from pages.page_executive import render
    render()

elif selected_page == PAGE_WORKFORCE:
    from pages.page_workforce import render
    render()

elif selected_page == PAGE_ATTRITION:
    from pages.page_attrition import render
    render()

elif selected_page == PAGE_PERFORMANCE:
    from pages.page_performance import render
    render()

elif selected_page == PAGE_XAI:
    from pages.page_xai import render
    render()

elif selected_page == PAGE_SIMULATOR:
    from pages.page_simulator import render
    render()

elif selected_page == PAGE_ABOUT:
    from pages.page_about import render
    render()
