# Phase 4 Explainability — Workforce Attrition Intelligence

## Overview

Phase 4 implements a modular explainability layer for the Phase 3 selected model:
**Logistic Regression (LR_balanced, clamp_and_flag treatment)**.

## Selected Model

| Property | Value |
|---|---|
| Model type | Logistic Regression (sklearn, lbfgs) |
| Imbalance strategy | class_weight='balanced' |
| Anomaly treatment | clamp_and_flag |
| CV PR-AUC | 0.1994 |
| Test PR-AUC | 0.2034 |
| Test ROC-AUC | 0.5040 |
| Optimal threshold (F1) | 0.3977 |
| Pipeline features | 36 |

## Explainability Approach

### Global Importance — Logistic Regression Coefficients

After StandardScaler brings continuous features to zero-mean/unit-variance and
categorical features to comparable integer/dummy scales, the magnitude of each
LR coefficient is a valid proxy for the feature's contribution to the log-odds
of attrition.

**Top 5 features by absolute coefficient:**

| Rank | Feature | Coefficient | Direction |
|---|---|---|---|
| 1 | `Years_in_Current_Role_Clamped` | +0.1615 | (+) higher attrition risk |
| 2 | `Years_in_Current_Role` | -0.1521 | (-) lower attrition risk |
| 3 | `Role_Tenure_Anomaly` | +0.1505 | (+) higher attrition risk |
| 4 | `Department_Marketing` | -0.1167 | (-) lower attrition risk |
| 5 | `Job_Role_Manager` | -0.1041 | (-) lower attrition risk |

### SHAP — LinearExplainer (Interventional)

`shap.LinearExplainer` computes exact Shapley values in closed form for linear
models.  SHAP values are additive contributions to the model's log-odds output:

    φᵢ(x) = cᵢ × (xᵢ − E[xᵢ])

The background dataset is X_train (after clamp_and_flag treatment).
SHAP values were computed using `shap.maskers.Independent` (interventional).

**SHAP base value** (population avg log-odds): 0.0070

**Top 5 features by mean |SHAP value| over X_train:**

| Rank | Feature | Mean |SHAP| | Direction |
|---|---|---|---|
| 1 | `Years_in_Current_Role_Clamped` | 0.1420 | (-) lower attrition risk |
| 2 | `Years_in_Current_Role` | 0.1362 | (-) lower attrition risk |
| 3 | `Role_Tenure_Anomaly` | 0.0529 | (-) lower attrition risk |
| 4 | `Job_Level` | 0.0507 | (+) higher attrition risk |
| 5 | `Number_of_Companies_Worked` | 0.0413 | (-) lower attrition risk |

### Local (Per-Employee) Explanation

`explain_prediction()` takes a single-row DataFrame and returns:
- `attrition_probability` — predicted probability (0–1)
- `risk_label` — 'High Risk' / 'Moderate Risk' / 'Low Risk'
- `local_shap_df` — per-feature SHAP contributions for this employee
- `global_importance_df` — global coefficient ranking
- `text_report` — formatted human-readable report with disclaimers

## ⚠ Critical Disclaimers

1. **Association ≠ Causation**: Positive or negative coefficients/SHAP values
   reflect statistical associations in a synthetic dataset. They do NOT
   establish that any feature **causes** attrition.

2. **No employee is described as certain to leave.** Risk predictions are
   analytical signals only. The output `attrition_probability` is a model
   estimate, not a fact.

3. **Synthetic-data limitation**: This dataset is computer-generated. Patterns
   and magnitudes may not reflect real workforce populations. All results should
   be treated as illustrative rather than operationally prescriptive.

4. **The held-out test set was never used for explainability fitting**. The SHAP
   explainer background dataset is X_train only.

## Saved Artifacts

| File | Description |
|---|---|
| `models/shap_explainer.pkl` | Serialised SHAP LinearExplainer |
| `models/global_importance_coef.csv` | Coefficient-based global importance |
| `models/global_importance_shap.csv` | SHAP mean-abs global importance |
| `models/local_explanation_sample.csv` | SHAP local explanations for 5 sample rows |
| `models/explainability_README.md` | This document |

## Module Structure

```
src/explainability/
├── __init__.py
├── feature_names.py          # Feature name extraction utilities
├── global_importance.py      # Coefficient-based global importance
├── shap_explainer.py         # SHAP LinearExplainer (build/save/load/compute)
├── local_explanation.py      # Per-employee SHAP and coef×deviation local explanations
├── prediction_explainer.py   # High-level PredictionExplanation API
└── generate_artifacts.py     # This script — generates all saved artifacts

tests/
└── test_phase4_explainability.py  # Full test suite for Phase 4
```
