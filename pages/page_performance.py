"""
pages/page_performance.py
==========================
Model Performance — Page 4.

Shows CV and held-out test metrics, confusion matrix, and an honest
explanation of why the model's performance is near the class-prevalence
baseline.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.dashboard_utils import (
    MODEL_LIMITATION_DISCLAIMER,
    SYNTHETIC_DISCLAIMER,
    confusion_matrix_heatmap,
    fmt_float,
    fmt_pct,
    load_cv_results,
    load_test_results,
)

# Class-prevalence baseline for reference lines
_CLASS_PREVALENCE = 0.1997


def render() -> None:
    """Render the Model Performance page."""

    st.title("🎯 Model Performance")
    st.caption(
        "Honest evaluation of the selected model (LR_balanced, clamp_and_flag) "
        "against held-out test data. Read the limitation notice below before "
        "interpreting these results."
    )

    # -----------------------------------------------------------------------
    # Limitation notices
    # -----------------------------------------------------------------------
    st.error(
        "**Model limitation**: The selected model's PR-AUC and ROC-AUC are near "
        f"the class-prevalence baseline (~{_CLASS_PREVALENCE:.0%} positive class). "
        "The model has **limited discriminative power**. Predictions should not be "
        "used for consequential individual employment decisions.",
        icon="⚠️",
    )
    st.markdown(
        f'<div class="disclaimer-box">{SYNTHETIC_DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )

    try:
        cv = load_cv_results()
        test = load_test_results()
    except Exception as e:
        st.error(f"Could not load model results: {e}")
        return

    winner = cv["winner"]
    threshold = cv["optimal_threshold"]
    default_test = test  # default threshold (0.5) metrics

    # -----------------------------------------------------------------------
    # Model identity
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown('<div class="section-header">Selected Model</div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Model", winner["model"])
    m2.metric("Anomaly Treatment", winner["treatment"])
    m3.metric("Imbalance Strategy", winner["imbalance_strategy"])

    st.info(
        "This is the winning model selected after 5-fold stratified cross-validation "
        "across 22 model+treatment combinations. See the CV Results section for details."
    )

    # -----------------------------------------------------------------------
    # Cross-validation metrics
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown('<div class="section-header">Cross-Validation Metrics (5-fold, Stratified)</div>', unsafe_allow_html=True)

    cv_cols = st.columns(6)
    cv_cols[0].metric(
        "CV PR-AUC",
        fmt_float(winner["mean_pr_auc"]),
        delta=f"±{winner['std_pr_auc']:.4f}",
        help="Mean ± std across 5 folds. Class-prevalence baseline ≈ 0.200",
    )
    cv_cols[1].metric(
        "CV ROC-AUC",
        fmt_float(winner["mean_roc_auc"]),
        delta=f"±{winner['std_roc_auc']:.4f}",
        help="Mean ± std across 5 folds. Random baseline = 0.500",
    )
    cv_cols[2].metric(
        "CV F1",
        fmt_float(winner["mean_f1"]),
        delta=f"±{winner['std_f1']:.4f}",
    )
    cv_cols[3].metric(
        "CV Precision",
        fmt_float(winner["mean_precision"]),
        delta=f"±{winner['std_precision']:.4f}",
    )
    cv_cols[4].metric(
        "CV Recall",
        fmt_float(winner["mean_recall"]),
        delta=f"±{winner['std_recall']:.4f}",
    )
    cv_cols[5].metric(
        "Class Prevalence",
        fmt_pct(_CLASS_PREVALENCE),
        help="The PR-AUC baseline a random classifier achieves.",
    )

    st.caption(
        f"The PR-AUC baseline for a random classifier is ~{_CLASS_PREVALENCE:.0%} "
        "(equal to class prevalence). The model's CV PR-AUC of "
        f"{winner['mean_pr_auc']:.4f} is **only marginally above** this baseline. "
        f"The ROC-AUC of {winner['mean_roc_auc']:.4f} is near the random-classifier value of 0.5. "
        "This confirms the model has very limited discriminative power."
    )

    # -----------------------------------------------------------------------
    # Held-out test metrics
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown(
        '<div class="section-header">Held-Out Test Set Metrics (Default Threshold = 0.5)</div>',
        unsafe_allow_html=True,
    )

    t_cols = st.columns(5)
    t_cols[0].metric("Test PR-AUC",  fmt_float(default_test["pr_auc"]))
    t_cols[1].metric("Test ROC-AUC", fmt_float(default_test["roc_auc"]))
    t_cols[2].metric("Test F1",      fmt_float(default_test["f1"]))
    t_cols[3].metric("Test Precision", fmt_float(default_test["precision"]))
    t_cols[4].metric("Test Recall",    fmt_float(default_test["recall"]))

    # -----------------------------------------------------------------------
    # Optimal threshold metrics
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown(
        f'<div class="section-header">Optimal Threshold Metrics (Threshold = {threshold:.4f})</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "The optimal threshold was selected to maximise F1 on the CV folds. "
        "At this threshold, nearly all employees are flagged as 'high risk', "
        "which again illustrates the model's limited discrimination."
    )

    opt = default_test.get("optimal_threshold_metrics", {})
    if opt:
        ot_cols = st.columns(5)
        ot_cols[0].metric("Opt. Threshold", f"{threshold:.4f}")
        ot_cols[1].metric("F1 @ Opt.", fmt_float(opt.get("f1", 0)))
        ot_cols[2].metric("Precision @ Opt.", fmt_float(opt.get("precision", 0)))
        ot_cols[3].metric("Recall @ Opt.", fmt_float(opt.get("recall", 0)))
        ot_cols[4].metric(
            "Flagged Rate",
            fmt_pct(cv["threshold_selection"].get("predicted_positive_rate", 0)),
            help="Fraction of employees flagged as at-risk at this threshold.",
        )

        st.warning(
            f"At the optimal threshold ({threshold:.4f}), the model flags "
            f"≈{cv['threshold_selection']['predicted_positive_rate']:.0%} of employees as high risk. "
            "This is essentially a near-universal flag and has no practical utility "
            "as a discriminator between individuals."
        )

    # -----------------------------------------------------------------------
    # Confusion matrices side by side
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown('<div class="section-header">Confusion Matrices</div>', unsafe_allow_html=True)

    cm_col1, cm_col2 = st.columns(2)
    with cm_col1:
        st.markdown("**Default threshold (0.5)**")
        fig_cm1 = confusion_matrix_heatmap(
            default_test["confusion_matrix"],
            title="Confusion Matrix — Threshold 0.5",
        )
        st.plotly_chart(fig_cm1, use_container_width=True)
    with cm_col2:
        if opt and opt.get("confusion_matrix"):
            st.markdown(f"**Optimal threshold ({threshold:.4f})**")
            fig_cm2 = confusion_matrix_heatmap(
                opt["confusion_matrix"],
                title=f"Confusion Matrix — Threshold {threshold:.4f}",
            )
            st.plotly_chart(fig_cm2, use_container_width=True)

    # -----------------------------------------------------------------------
    # Classification report
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown('<div class="section-header">Classification Report (Threshold = 0.5)</div>', unsafe_allow_html=True)
    st.code(default_test.get("classification_report", "Not available"), language=None)

    # -----------------------------------------------------------------------
    # All CV results table
    # -----------------------------------------------------------------------
    st.markdown("---")
    with st.expander("📋 All Cross-Validation Results (22 combinations)", expanded=False):
        all_cv = pd.DataFrame(cv.get("all_cv_results", []))
        if not all_cv.empty:
            all_cv = all_cv.sort_values("mean_pr_auc", ascending=False).reset_index(drop=True)
            all_cv.insert(0, "rank", range(1, len(all_cv) + 1))
            # Format floats
            float_cols = [c for c in all_cv.columns if "mean_" in c or "std_" in c]
            for c in float_cols:
                all_cv[c] = all_cv[c].apply(lambda v: f"{v:.4f}")
            st.dataframe(all_cv, use_container_width=True, hide_index=True)
        else:
            st.info("No CV results available.")

    # -----------------------------------------------------------------------
    # PR-AUC comparison bar chart
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown('<div class="section-header">PR-AUC Across All Models</div>', unsafe_allow_html=True)
    all_cv_raw = pd.DataFrame(cv.get("all_cv_results", []))
    if not all_cv_raw.empty:
        all_cv_raw["label"] = all_cv_raw["model"] + " / " + all_cv_raw["treatment"]
        all_cv_raw = all_cv_raw.sort_values("mean_pr_auc", ascending=True)
        fig_cv_bar = go.Figure()
        fig_cv_bar.add_trace(go.Bar(
            x=all_cv_raw["mean_pr_auc"],
            y=all_cv_raw["label"],
            orientation="h",
            error_x=dict(type="data", array=all_cv_raw["std_pr_auc"]),
            marker_color="#3b82d4",
            name="Mean CV PR-AUC",
        ))
        # Add baseline reference line
        fig_cv_bar.add_vline(
            x=_CLASS_PREVALENCE,
            line_dash="dash",
            line_color="#e74c3c",
            annotation_text=f"Class prevalence baseline ≈ {_CLASS_PREVALENCE:.0%}",
            annotation_position="top right",
        )
        fig_cv_bar.update_layout(
            title="5-Fold CV PR-AUC — All Models (red dashed line = class prevalence baseline)",
            xaxis_title="Mean PR-AUC",
            height=max(400, len(all_cv_raw) * 28 + 100),
            margin=dict(l=10, r=20, t=60, b=40),
        )
        st.plotly_chart(fig_cv_bar, use_container_width=True)
        st.caption(
            "All models perform close to the class-prevalence baseline (dashed red line). "
            "The marginal improvements are within noise. "
            "This confirms the model has very limited real-world discriminative utility."
        )

    # -----------------------------------------------------------------------
    # Footer
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.error(MODEL_LIMITATION_DISCLAIMER)
