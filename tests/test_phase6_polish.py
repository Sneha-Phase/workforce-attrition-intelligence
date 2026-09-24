"""
tests/test_phase6_polish.py
============================
Phase 6 test suite — Final Dashboard Validation & Portfolio Polish.

Coverage
--------
1.  README.md exists and contains required sections
2.  app.py sidebar caption says "Phase 6"
3.  page_performance.py — f-string fix: ROC-AUC caption uses f-string interpolation
4.  page_about.py — model count says 22 combinations and 8 model types
5.  page_about.py — Phase 6 entry present in Methodology tab phases dict
6.  page_xai.py — _SAMPLE_LABELS contain 3-decimal probabilities (not 2-decimal)
7.  cv_results.json — actual combination count is 22
8.  cv_results.json — actual model types count is 8
9.  final_test_results.json — required keys present
10. Risk Simulator: normal profile (Years_in_Current_Role <= Years_at_Company)
    produces a valid prediction (regression test — Phase 5 fix still holds)
11. Risk Simulator: anomalous profile (Years_in_Current_Role > Years_at_Company)
    produces a valid prediction without KeyError
12. Risk Simulator: clamped column value is correct for normal profile
13. Risk Simulator: clamped column value is correct for anomalous profile
14. Disclaimer constants — all five present and non-empty in dashboard_utils
15. local_explanation_sample.csv — 5 distinct employee_sample keys
16. local_explanation_sample.csv — all required columns present
17. global_importance_coef.csv — required columns present, 36 features
18. global_importance_shap.csv — required columns present, 36 features
19. dataset_profile.md exists
20. requirements.txt exists and non-empty

Test strategy
--------------
- Source file checks (existence, content) are pure Python — no Streamlit session.
- Pipeline integration tests skip gracefully when model artifact is absent.
- RANDOM_STATE=42 throughout.
- Fixtures are module-level helpers (not class-scoped instance methods) to
  avoid the PytestRemovedIn10Warning about class-scoped fixture deprecation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Project root helper
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Module-level data loaders (used instead of class-scoped fixtures to
# avoid PytestRemovedIn10Warning about instance-method class-scope fixtures)
# ---------------------------------------------------------------------------

def _load_cv_json() -> dict:
    path = _ROOT / "models" / "cv_results.json"
    if not path.exists():
        pytest.skip("cv_results.json not found.")
    with open(path, "r") as f:
        return json.load(f)


def _load_test_results_json() -> dict:
    path = _ROOT / "models" / "final_test_results.json"
    if not path.exists():
        pytest.skip("final_test_results.json not found.")
    with open(path, "r") as f:
        return json.load(f)


def _load_local_sample_csv() -> pd.DataFrame:
    path = _ROOT / "models" / "local_explanation_sample.csv"
    if not path.exists():
        pytest.skip("local_explanation_sample.csv not found.")
    return pd.read_csv(path)


def _make_normal_profile() -> dict:
    """Profile where Years_in_Current_Role <= Years_at_Company (non-anomalous)."""
    return {
        "Age": 35, "Gender": "Male", "Marital_Status": "Married",
        "Department": "IT", "Job_Role": "Analyst", "Job_Level": 2,
        "Monthly_Income": 7000, "Hourly_Rate": 40,
        "Years_at_Company": 5, "Years_in_Current_Role": 3,   # 3 <= 5 → normal
        "Years_Since_Last_Promotion": 2,
        "Work_Life_Balance": 3, "Job_Satisfaction": 3, "Performance_Rating": 3,
        "Training_Hours_Last_Year": 20, "Overtime": "No", "Project_Count": 3,
        "Average_Hours_Worked_Per_Week": 42, "Absenteeism": 3,
        "Work_Environment_Satisfaction": 3, "Relationship_with_Manager": 3,
        "Job_Involvement": 3, "Distance_From_Home": 10,
        "Number_of_Companies_Worked": 2,
    }


def _make_anomalous_profile() -> dict:
    """Profile where Years_in_Current_Role > Years_at_Company (anomalous)."""
    profile = _make_normal_profile()
    profile["Years_at_Company"] = 2
    profile["Years_in_Current_Role"] = 9   # 9 > 2 → anomalous
    return profile


# ---------------------------------------------------------------------------
# 1. README.md exists and contains required sections
# ---------------------------------------------------------------------------

class TestReadme:

    def test_readme_exists(self):
        assert (_ROOT / "README.md").exists(), "README.md is missing from project root."

    def test_readme_non_empty(self):
        content = (_ROOT / "README.md").read_text(encoding="utf-8")
        assert len(content) > 500, "README.md appears to be too short."

    def test_readme_has_project_overview(self):
        content = (_ROOT / "README.md").read_text(encoding="utf-8").lower()
        assert "workforce attrition" in content

    def test_readme_has_dataset_section(self):
        content = (_ROOT / "README.md").read_text(encoding="utf-8").lower()
        assert "dataset" in content

    def test_readme_has_methodology_or_phases(self):
        content = (_ROOT / "README.md").read_text(encoding="utf-8").lower()
        assert "phase" in content or "methodology" in content

    def test_readme_has_model_summary(self):
        content = (_ROOT / "README.md").read_text(encoding="utf-8").lower()
        assert "model" in content and ("pr-auc" in content or "roc-auc" in content)

    def test_readme_has_explainability(self):
        content = (_ROOT / "README.md").read_text(encoding="utf-8").lower()
        assert "shap" in content or "explainab" in content

    def test_readme_has_responsible_use(self):
        content = (_ROOT / "README.md").read_text(encoding="utf-8").lower()
        assert "responsible" in content or "adverse" in content or "synthetic" in content

    def test_readme_has_run_instructions(self):
        content = (_ROOT / "README.md").read_text(encoding="utf-8").lower()
        assert "streamlit run app.py" in content

    def test_readme_has_testing_instructions(self):
        content = (_ROOT / "README.md").read_text(encoding="utf-8").lower()
        assert "pytest" in content

    def test_readme_has_synthetic_disclaimer(self):
        content = (_ROOT / "README.md").read_text(encoding="utf-8").lower()
        assert "synthetic" in content

    def test_readme_has_no_adverse_action(self):
        content = (_ROOT / "README.md").read_text(encoding="utf-8").lower()
        assert "adverse" in content or "no adverse" in content or "must not" in content


# ---------------------------------------------------------------------------
# 2. app.py sidebar says Phase 6
# ---------------------------------------------------------------------------

class TestAppPhaseLabel:

    def test_sidebar_caption_says_phase_6(self):
        content = (_ROOT / "app.py").read_text(encoding="utf-8")
        assert "Phase 6" in content, (
            "app.py sidebar caption should reference Phase 6 after Phase 6 polish."
        )

    def test_sidebar_caption_not_phase_5(self):
        content = (_ROOT / "app.py").read_text(encoding="utf-8")
        # The Phase 5 label in the module docstring is fine, but the displayed
        # st.caption line must say Phase 6, not Phase 5
        caption_lines = [
            ln for ln in content.splitlines()
            if "st.caption" in ln and "Phase 5" in ln
        ]
        assert not caption_lines, (
            f"st.caption line still references Phase 5: {caption_lines}"
        )


# ---------------------------------------------------------------------------
# 3. page_performance.py — f-string fix
# ---------------------------------------------------------------------------

class TestPerformanceFString:

    def test_roc_auc_caption_uses_fstring_not_format(self):
        content = (_ROOT / "pages" / "page_performance.py").read_text(encoding="utf-8")
        # The broken version used .format() for the ROC-AUC value.
        # The fixed version uses f-string interpolation {winner['mean_roc_auc']:.4f}
        assert (
            "winner['mean_roc_auc']" in content
            or 'winner["mean_roc_auc"]' in content
        ), "ROC-AUC caption should reference winner['mean_roc_auc'] via f-string."

    def test_no_orphan_format_call_for_roc(self):
        content = (_ROOT / "pages" / "page_performance.py").read_text(encoding="utf-8")
        # Broken pattern produced a literal "{:.4f}" string without a variable name
        assert '"{:.4f}' not in content, (
            "Broken .format() pattern for ROC-AUC still present in page_performance.py."
        )


# ---------------------------------------------------------------------------
# 4 & 5. page_about.py — model counts and Phase 6 entry
# ---------------------------------------------------------------------------

class TestAboutPageAccuracy:

    def test_22_combinations_mentioned(self):
        content = (_ROOT / "pages" / "page_about.py").read_text(encoding="utf-8")
        assert "22 combinations" in content or "22 model" in content, (
            "page_about.py should mention 22 model+treatment combinations."
        )

    def test_8_model_types_mentioned(self):
        content = (_ROOT / "pages" / "page_about.py").read_text(encoding="utf-8")
        assert "8 model types" in content, (
            "page_about.py should reference 8 model types."
        )

    def test_phase_6_in_methodology(self):
        content = (_ROOT / "pages" / "page_about.py").read_text(encoding="utf-8")
        assert "Phase 6" in content, (
            "page_about.py should include a Phase 6 entry in the methodology section."
        )

    def test_21_combinations_removed(self):
        content = (_ROOT / "pages" / "page_about.py").read_text(encoding="utf-8")
        assert "21 combinations" not in content, (
            "page_about.py should not reference 21 combinations (correct count is 22)."
        )


# ---------------------------------------------------------------------------
# 6. page_xai.py — sample labels contain 3-decimal probabilities
# ---------------------------------------------------------------------------

class TestXaiSampleLabels:

    def test_sample_labels_have_3_decimal_probs(self):
        content = (_ROOT / "pages" / "page_xai.py").read_text(encoding="utf-8")
        # The corrected labels use ≈0.453 style (3 decimals) not ≈0.45 (2 decimals)
        assert "p≈0.453" in content or "0.453" in content, (
            "XAI page sample labels should show 3-decimal probabilities matching the CSV."
        )


# ---------------------------------------------------------------------------
# 7 & 8. cv_results.json — actual combination and model counts
# ---------------------------------------------------------------------------

class TestCVResultsJson:

    def test_22_combinations_in_json(self):
        cv = _load_cv_json()
        n = len(cv.get("all_cv_results", []))
        assert n == 22, f"Expected 22 CV combinations, found {n}."

    def test_8_model_types_in_json(self):
        cv = _load_cv_json()
        model_types = {r["model"] for r in cv.get("all_cv_results", [])}
        assert len(model_types) == 8, (
            f"Expected 8 model types, found {len(model_types)}: {model_types}"
        )

    def test_winner_model_is_lr_balanced(self):
        cv = _load_cv_json()
        assert cv["winner"]["model"] == "LR_balanced"

    def test_winner_treatment_is_clamp_and_flag(self):
        cv = _load_cv_json()
        assert cv["winner"]["treatment"] == "clamp_and_flag"

    def test_optimal_threshold_in_range(self):
        cv = _load_cv_json()
        t = cv["optimal_threshold"]
        assert 0.0 < t < 1.0, f"Threshold {t} out of (0,1) range."


# ---------------------------------------------------------------------------
# 9. final_test_results.json — required keys present
# ---------------------------------------------------------------------------

class TestFinalTestResultsJson:

    def test_required_keys_present(self):
        results = _load_test_results_json()
        for key in ("precision", "recall", "f1", "roc_auc", "pr_auc",
                    "confusion_matrix", "classification_report"):
            assert key in results, f"Missing key in final_test_results.json: {key}"

    def test_pr_auc_in_range(self):
        results = _load_test_results_json()
        assert 0.0 < results["pr_auc"] < 1.0

    def test_roc_auc_in_range(self):
        results = _load_test_results_json()
        assert 0.0 < results["roc_auc"] < 1.0

    def test_confusion_matrix_shape(self):
        results = _load_test_results_json()
        cm = results["confusion_matrix"]
        assert len(cm) == 2 and len(cm[0]) == 2, "Confusion matrix must be 2×2."


# ---------------------------------------------------------------------------
# 10-13. Risk Simulator — normal and anomalous profile regression tests
# ---------------------------------------------------------------------------

class TestRiskSimulatorProfiles:
    """Phase 6 validation of Risk Simulator with both profile types."""

    def test_normal_profile_clamped_col_present(self):
        """Normal profile must produce Years_in_Current_Role_Clamped column."""
        from utils.dashboard_utils import build_employee_row
        from src.models.anomaly_treatments import COL_ROLE_CLAMPED
        row = build_employee_row(_make_normal_profile())
        assert COL_ROLE_CLAMPED in row.columns

    def test_anomalous_profile_clamped_col_present(self):
        """Anomalous profile must produce Years_in_Current_Role_Clamped column."""
        from utils.dashboard_utils import build_employee_row
        from src.models.anomaly_treatments import COL_ROLE_CLAMPED
        row = build_employee_row(_make_anomalous_profile())
        assert COL_ROLE_CLAMPED in row.columns

    def test_normal_profile_clamped_value_unchanged(self):
        """For normal profile, clamped == original Years_in_Current_Role."""
        from utils.dashboard_utils import build_employee_row
        from src.models.anomaly_treatments import COL_ROLE_CLAMPED
        profile = _make_normal_profile()
        row = build_employee_row(profile)
        assert row[COL_ROLE_CLAMPED].iloc[0] == profile["Years_in_Current_Role"]

    def test_anomalous_profile_clamped_value_is_company_tenure(self):
        """For anomalous profile, clamped == Years_at_Company."""
        from utils.dashboard_utils import build_employee_row
        from src.models.anomaly_treatments import COL_ROLE_CLAMPED
        profile = _make_anomalous_profile()
        row = build_employee_row(profile)
        assert row[COL_ROLE_CLAMPED].iloc[0] == profile["Years_at_Company"]

    def test_normal_profile_pipeline_no_error(self):
        """Normal profile must pass through pipeline without error."""
        import joblib
        from utils.dashboard_utils import build_employee_row
        from src.config import PIPELINE_PATH
        if not PIPELINE_PATH.exists():
            pytest.skip("Pipeline artifact not found.")
        pipeline = joblib.load(PIPELINE_PATH)
        row = build_employee_row(_make_normal_profile())
        proba = pipeline.predict_proba(row)
        assert proba.shape == (1, 2)
        assert 0.0 <= proba[0, 1] <= 1.0

    def test_anomalous_profile_pipeline_no_error(self):
        """Anomalous profile must pass through pipeline without KeyError."""
        import joblib
        from utils.dashboard_utils import build_employee_row
        from src.config import PIPELINE_PATH
        if not PIPELINE_PATH.exists():
            pytest.skip("Pipeline artifact not found.")
        pipeline = joblib.load(PIPELINE_PATH)
        row = build_employee_row(_make_anomalous_profile())
        proba = pipeline.predict_proba(row)
        assert proba.shape == (1, 2)
        assert 0.0 <= proba[0, 1] <= 1.0


# ---------------------------------------------------------------------------
# 14. Disclaimer constants — all five present in dashboard_utils
# ---------------------------------------------------------------------------

class TestDisclaimerConstants:

    def test_all_disclaimer_constants_present(self):
        from utils.dashboard_utils import (
            SYNTHETIC_DISCLAIMER,
            ASSOCIATION_DISCLAIMER,
            RISK_SIGNAL_DISCLAIMER,
            MODEL_LIMITATION_DISCLAIMER,
            NO_ADVERSE_ACTION_DISCLAIMER,
        )
        for name, val in [
            ("SYNTHETIC_DISCLAIMER", SYNTHETIC_DISCLAIMER),
            ("ASSOCIATION_DISCLAIMER", ASSOCIATION_DISCLAIMER),
            ("RISK_SIGNAL_DISCLAIMER", RISK_SIGNAL_DISCLAIMER),
            ("MODEL_LIMITATION_DISCLAIMER", MODEL_LIMITATION_DISCLAIMER),
            ("NO_ADVERSE_ACTION_DISCLAIMER", NO_ADVERSE_ACTION_DISCLAIMER),
        ]:
            assert isinstance(val, str) and len(val) > 20, (
                f"{name} is missing or too short."
            )

    def test_synthetic_disclaimer_mentions_synthetic(self):
        from utils.dashboard_utils import SYNTHETIC_DISCLAIMER
        assert "synthetic" in SYNTHETIC_DISCLAIMER.lower()

    def test_no_adverse_action_mentions_adverse(self):
        from utils.dashboard_utils import NO_ADVERSE_ACTION_DISCLAIMER
        assert "adverse" in NO_ADVERSE_ACTION_DISCLAIMER.lower()


# ---------------------------------------------------------------------------
# 15 & 16. local_explanation_sample.csv
# ---------------------------------------------------------------------------

class TestLocalExplanationCSV:

    def test_five_distinct_samples(self):
        df = _load_local_sample_csv()
        n = df["employee_sample"].nunique()
        assert n == 5, f"Expected 5 sample employees, found {n}."

    def test_required_columns_present(self):
        df = _load_local_sample_csv()
        for col in ("employee_sample", "predicted_probability", "risk_label",
                    "feature", "shap_value", "direction", "transformed_value"):
            assert col in df.columns, f"Missing column: {col}"

    def test_probabilities_in_range(self):
        df = _load_local_sample_csv()
        assert (df["predicted_probability"] >= 0).all()
        assert (df["predicted_probability"] <= 1).all()


# ---------------------------------------------------------------------------
# 17 & 18. Importance CSVs
# ---------------------------------------------------------------------------

class TestImportanceCSVs:

    def test_coef_csv_required_columns(self):
        path = _ROOT / "models" / "global_importance_coef.csv"
        if not path.exists():
            pytest.skip("global_importance_coef.csv not found.")
        df = pd.read_csv(path)
        for col in ("rank", "feature", "coefficient", "abs_coef", "direction"):
            assert col in df.columns, f"Missing column in coef CSV: {col}"
        assert len(df) == 36, f"Expected 36 features, found {len(df)}."

    def test_shap_csv_required_columns(self):
        path = _ROOT / "models" / "global_importance_shap.csv"
        if not path.exists():
            pytest.skip("global_importance_shap.csv not found.")
        df = pd.read_csv(path)
        for col in ("rank", "feature", "mean_abs_shap", "direction"):
            assert col in df.columns, f"Missing column in SHAP CSV: {col}"
        assert len(df) == 36, f"Expected 36 features, found {len(df)}."


# ---------------------------------------------------------------------------
# 19 & 20. Supporting files exist
# ---------------------------------------------------------------------------

class TestSupportingFiles:

    def test_dataset_profile_md_exists(self):
        assert (_ROOT / "dataset_profile.md").exists(), (
            "dataset_profile.md is missing from project root."
        )

    def test_requirements_txt_exists_and_non_empty(self):
        path = _ROOT / "requirements.txt"
        assert path.exists(), "requirements.txt is missing."
        content = path.read_text(encoding="utf-8").strip()
        assert len(content) > 0, "requirements.txt is empty."

    def test_requirements_contains_streamlit(self):
        content = (_ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
        assert "streamlit" in content

    def test_requirements_contains_scikit(self):
        content = (_ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
        assert "scikit-learn" in content or "sklearn" in content
