"""
src/explainability/__init__.py
================================
Public API for the Phase 4 explainability layer.

Primary entry points
---------------------
explain_prediction(pipeline, explainer, X_row, ...)
    → PredictionExplanation  (local + global explanation bundle)

get_global_feature_importance(pipeline, ...)
    → pd.DataFrame  (coefficient-based global ranking)

build_shap_explainer(pipeline, X_background)
    → shap.LinearExplainer

get_feature_names_from_pipeline(pipeline)
    → list[str]

All outputs carry the following guarantees:
- No pipeline re-training occurs.
- The source CSV is never modified.
- The held-out test set is not used here.
- Risk predictions are analytical signals only (association ≠ causation).
- Results are derived from a synthetic dataset.
"""

from src.explainability.feature_names import (
    get_feature_names_from_pipeline,
    get_feature_name_index_map,
    extract_preprocessor,
    transform_input,
)
from src.explainability.global_importance import (
    get_global_feature_importance,
    format_global_importance_report,
    CAUSATION_DISCLAIMER,
    SYNTHETIC_DATA_DISCLAIMER,
)
from src.explainability.shap_explainer import (
    build_shap_explainer,
    save_shap_explainer,
    load_shap_explainer,
    compute_shap_values,
    get_shap_global_importance,
    format_shap_global_report,
)
from src.explainability.local_explanation import (
    explain_local_shap,
    explain_local_coef_deviation,
    format_local_explanation_report,
)
from src.explainability.prediction_explainer import (
    PredictionExplanation,
    explain_prediction,
    explain_batch,
    _classify_risk,
)

__all__ = [
    # feature_names
    "get_feature_names_from_pipeline",
    "get_feature_name_index_map",
    "extract_preprocessor",
    "transform_input",
    # global_importance
    "get_global_feature_importance",
    "format_global_importance_report",
    "CAUSATION_DISCLAIMER",
    "SYNTHETIC_DATA_DISCLAIMER",
    # shap_explainer
    "build_shap_explainer",
    "save_shap_explainer",
    "load_shap_explainer",
    "compute_shap_values",
    "get_shap_global_importance",
    "format_shap_global_report",
    # local_explanation
    "explain_local_shap",
    "explain_local_coef_deviation",
    "format_local_explanation_report",
    # prediction_explainer
    "PredictionExplanation",
    "explain_prediction",
    "explain_batch",
    "_classify_risk",
]
