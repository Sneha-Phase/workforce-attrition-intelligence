# Workforce Attrition Intelligence & Retention Analytics Platform

> **Portfolio project** · End-to-end responsible HR analytics in Python  
> **Data:** Synthetic dataset — no real employee data used  
> **Purpose:** Education, skill demonstration, responsible-AI advocacy

---

## ⚠ Important Disclaimers

| | |
|---|---|
| **Synthetic data only** | The dataset (`data/employee_attrition_dataset_10000.csv`) is computer-generated. It does not represent any real organisation or real employees. |
| **Near-baseline model** | CV PR-AUC ≈ 0.199 (class-prevalence baseline ≈ 0.200). The model has **very limited discriminative ability**. |
| **No adverse action** | This tool must **never** be used to fire, demote, penalise, or take any adverse employment action against any individual. |
| **Association ≠ Causation** | All feature associations and SHAP values are statistical patterns in synthetic data. They do not establish causation. |

---

## Project Overview

A six-phase, portfolio-grade data science project demonstrating:

- End-to-end ML pipeline with documented anomaly handling
- Responsible model evaluation (PR-AUC, honest baseline comparison)
- SHAP-based explainability (LinearExplainer, global + local)
- Multi-page Streamlit dashboard with comprehensive responsible-use messaging
- Full pytest test suite (378 tests, 100% pass rate)

### Phase Summary

| Phase | Description | Key Output |
|-------|-------------|------------|
| **1 — Data Profiling** | Load, validate, profile 10 000-row synthetic CSV | Discovered 22.84% role-tenure anomaly |
| **2 — Feature Engineering** | 5 engineered features, preprocessing pipeline | `best_pipeline.pkl` |
| **3 — Model Selection** | 5-fold CV across 22 model+treatment combinations | `cv_results.json`, `final_test_results.json` |
| **4 — Explainability** | SHAP LinearExplainer, global + local importance | `shap_explainer.pkl`, importance CSVs |
| **5 — Dashboard** | 7-page Streamlit app with Risk Simulator | `app.py` + `pages/` |
| **6 — Portfolio Polish** | End-to-end validation, bug fixes, documentation | This README |

---

## Dataset

**File:** `data/employee_attrition_dataset_10000.csv`

| Property | Value |
|----------|-------|
| Rows | 10 000 synthetic employees |
| Columns | 26 (25 features + 1 target) |
| Target | `Attrition` — `Yes` (~20%) / `No` (~80%) |
| Class imbalance | ~80.03% No / ~19.97% Yes |
| Missing values | None |
| Data source | Synthetic generation (algorithmic) |

### Known Data Quality Issue

`Years_in_Current_Role > Years_at_Company` for **22.84% of rows** (2 284 employees).
This is logically impossible (role tenure cannot exceed company tenure).

**Treatment:** `clamp_and_flag` — creates `Years_in_Current_Role_Clamped = min(role, company)`
alongside a binary `Role_Tenure_Anomaly` flag. The original columns are preserved unchanged.

---

## Feature Engineering

Five features engineered from raw inputs (`src/features/build_features.py`):

| Feature | Formula | Purpose |
|---------|---------|---------|
| `Role_Tenure_Anomaly` | `Years_in_Current_Role > Years_at_Company` | Binary anomaly flag |
| `Satisfaction_Composite` | Mean of 5 Likert scales | Well-being index |
| `Tenure_Promotion_Ratio` | `Years_at_Company / (Years_Since_Last_Promotion + 1)` | Career stagnation proxy |
| `Hours_Overload` | `Average_Hours_Worked_Per_Week − 40` | Deviation from standard week |
| `Income_to_Role_Ratio` | `Monthly_Income / Job_Level` | Compensation relative to level |

---

## Preprocessing Pipeline

Built with scikit-learn `ColumnTransformer` (`src/preprocessing/pipeline.py`):

| Transformer | Features | Notes |
|-------------|----------|-------|
| `OneHotEncoder` (drop=first) | Department, Job_Role, Marital_Status | Nominal categories |
| `OrdinalEncoder` | Gender, Overtime | Binary categories |
| Pass-through (no scaling) | Ordinal Likert scales, Role_Tenure_Anomaly | Integer scales |
| `StandardScaler` | Age, income, tenure, hours, engineered ratios | Continuous numeric |

---

## Model Selection

**22 combinations** evaluated: 8 model types × 3 anomaly treatments
(RF_SMOTE evaluated with flag_only only).

**Model types:** LR, LR_balanced, RF, RF_balanced, GB, XGB, XGB_balanced, RF_SMOTE  
**Anomaly treatments:** flag_only, clamp_and_flag, exclude  
**Selection criterion:** Highest mean 5-fold CV PR-AUC (simplicity as tiebreaker)

### Winner: LR_balanced + clamp_and_flag

| Metric | CV Mean | CV Std |
|--------|---------|--------|
| PR-AUC | 0.1994 | ±0.0111 |
| ROC-AUC | 0.4934 | ±0.0209 |
| F1 | 0.2789 | ±0.0142 |
| Precision | 0.1966 | ±0.0094 |
| Recall | 0.4799 | ±0.0299 |

> **Class-prevalence baseline (random PR-AUC):** ~0.200  
> **Random classifier ROC-AUC:** 0.500  
> The model performs **marginally above the random baseline**. This is expected for synthetic data without a realistic attrition signal.

### Test-Set Metrics (default threshold 0.5)

| PR-AUC | ROC-AUC | F1 | Precision | Recall |
|--------|---------|-----|-----------|--------|
| 0.2034 | 0.5040 | 0.2743 | 0.1934 | 0.4712 |

### Threshold Analysis

F1-optimal threshold: **0.3977** — flags ~99.1% of employees as high risk.
This illustrates that the model cannot meaningfully discriminate at the individual level.

---

## Explainability

Implemented in `src/explainability/` using SHAP `LinearExplainer` (interventional).

- **Global importance:** Mean |SHAP| across 800-sample background dataset (`models/global_importance_shap.csv`)
- **Coefficient importance:** Absolute LR coefficients after StandardScaler (`models/global_importance_coef.csv`)
- **Local explanations:** 5 pre-computed sample employees at 10th/30th/50th/70th/90th predicted-probability percentiles (`models/local_explanation_sample.csv`)

All explanations carry association ≠ causation disclaimers.

---

## Dashboard Pages

Run with: `streamlit run app.py`

| Page | Description |
|------|-------------|
| **📊 Executive HR Dashboard** | Workforce KPIs, attrition overview, department/role summaries |
| **👥 Workforce Analytics** | Distributions (demographics, income, satisfaction, categorical) |
| **📉 Attrition Analysis** | Attrition rates by group, numeric distributions, cross-tab explorer |
| **🎯 Model Performance** | CV and test metrics, confusion matrices, PR-AUC comparison |
| **🔍 Explainable AI** | Global coefficient + SHAP importance, local SHAP waterfall |
| **⚙️ Attrition Risk Simulator** | Hypothetical employee risk signal with acknowledgement gate |
| **📋 Responsible Use / About** | Limitations, responsible-use guide, methodology, project info |

---

## Project Structure

```
Workforce-Attrition-Intelligence/
├── app.py                          # Streamlit entry point
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── data/
│   └── employee_attrition_dataset_10000.csv
├── models/
│   ├── best_pipeline.pkl           # Trained sklearn Pipeline
│   ├── shap_explainer.pkl          # SHAP LinearExplainer
│   ├── cv_results.json             # Cross-validation results
│   ├── final_test_results.json     # Held-out test metrics
│   ├── global_importance_coef.csv  # Coefficient importance
│   ├── global_importance_shap.csv  # SHAP global importance
│   ├── local_explanation_sample.csv# Local SHAP for 5 samples
│   ├── model_comparison.csv        # All CV results table
│   ├── sensitivity_results.csv     # Sensitivity analysis
│   └── threshold_analysis.csv      # Threshold sweep results
├── pages/
│   ├── page_executive.py           # Page 1: Executive Dashboard
│   ├── page_workforce.py           # Page 2: Workforce Analytics
│   ├── page_attrition.py           # Page 3: Attrition Analysis
│   ├── page_performance.py         # Page 4: Model Performance
│   ├── page_xai.py                 # Page 5: Explainable AI
│   ├── page_simulator.py           # Page 6: Risk Simulator
│   └── page_about.py               # Page 7: Responsible Use / About
├── src/
│   ├── config.py                   # Central configuration
│   ├── data/loader.py              # CSV loading + validation
│   ├── features/build_features.py  # Feature engineering
│   ├── preprocessing/pipeline.py   # sklearn pipeline builder
│   ├── models/
│   │   ├── definitions.py          # Model definitions
│   │   ├── train.py                # Training loop
│   │   ├── evaluation.py           # Metric computation
│   │   ├── anomaly_treatments.py   # Flag/clamp/exclude treatments
│   │   ├── sensitivity.py          # Sensitivity analysis
│   │   └── threshold_analysis.py   # Threshold sweep
│   └── explainability/
│       ├── shap_explainer.py       # SHAP explainer builder
│       ├── global_importance.py    # Global importance computation
│       ├── local_explanation.py    # Local explanation computation
│       ├── prediction_explainer.py # Real-time simulator explainer
│       └── generate_artifacts.py   # Artifact generation script
├── utils/
│   └── dashboard_utils.py          # Shared dashboard helpers
└── tests/
    ├── test_loader.py              # Phase 1: data loading
    ├── test_features.py            # Phase 2: feature engineering
    ├── test_preprocessing.py       # Phase 2: pipeline
    ├── test_phase3_models.py       # Phase 3: model training/eval
    ├── test_phase4_explainability.py # Phase 4: SHAP/explainability
    └── test_phase5_dashboard.py    # Phase 5+6: dashboard helpers
```

---

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run the Streamlit Application

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` by default.

All model artifacts are pre-computed and stored in `models/`. No retraining is needed to run the dashboard.

### Regenerate Model Artifacts (optional)

If you want to retrain from scratch (requires ~2–5 minutes):

```bash
# Phase 3: Train and evaluate all models
python -m src.models.train

# Phase 4: Generate explainability artifacts
python -m src.explainability.generate_artifacts
```

---

## Running Tests

```bash
# Run the full test suite
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --tb=short

# Run a specific phase
python -m pytest tests/test_phase5_dashboard.py -v
```

**Expected result:** 378+ tests, 0 failures.

### Test Coverage by Module

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_loader.py` | ~20 | Data loading, schema validation |
| `test_features.py` | ~40 | Feature engineering, edge cases |
| `test_preprocessing.py` | ~60 | Pipeline, OHE, scaling |
| `test_phase3_models.py` | ~100 | Model training, evaluation, thresholds |
| `test_phase4_explainability.py` | ~80 | SHAP, coefficient importance |
| `test_phase5_dashboard.py` | ~78 | Dashboard helpers, Risk Simulator regression |

---

## Technology Stack

| Category | Libraries |
|----------|-----------|
| Data manipulation | pandas, numpy |
| Machine learning | scikit-learn, xgboost, imbalanced-learn |
| Explainability | shap (LinearExplainer) |
| Visualisation | plotly |
| Dashboard | streamlit |
| Testing | pytest |
| Model persistence | joblib |

---

## Known Limitations

1. **Synthetic data** — Patterns are artefacts of the data generator, not real HR dynamics.
2. **Near-baseline model** — PR-AUC ≈ 0.20 means essentially no discriminative ability at the individual level.
3. **No fairness validation** — The model was not tested for demographic bias. Do not use for real HR decisions without proper fairness audits.
4. **Role-tenure anomaly** — 22.84% of records have logically impossible tenure values. The clamp_and_flag treatment mitigates this but does not resolve the underlying data quality issue.
5. **Cross-sectional only** — A single snapshot dataset; no longitudinal dynamics are captured.
6. **No real-world validation** — Results have not been validated against any real organisation's data.

---

## Responsible Use Summary

✅ **Appropriate uses:** Educational demonstration · Portfolio showcase · Learning resource · Methodology illustration  
🚫 **Prohibited uses:** Employment decisions · Adverse actions · Surveillance · Any use with real employee data without proper validation

---

*Developed as part of an AICTE internship in Data Analytics. All data is synthetic. No real employee data was used.*
