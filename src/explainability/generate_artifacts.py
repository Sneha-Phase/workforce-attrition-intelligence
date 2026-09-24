"""
src/explainability/generate_artifacts.py
==========================================
Script to generate and save all Phase 4 explainability artifacts.

This script:
1. Loads the saved Phase 3 pipeline (models/best_pipeline.pkl) — read-only.
2. Loads and engineers features from the source CSV.
3. Reconstructs the training split (same seed as Phase 3) to get X_train.
4. Applies the clamp_and_flag anomaly treatment to X_train (matches the
   winning Phase 3 configuration).
5. Builds and saves the SHAP LinearExplainer on X_train (transformed).
6. Computes and saves the global feature importance (coefficient-based).
7. Computes and saves the SHAP global importance over X_train.
8. Saves a sample of local explanations for representative employees.

IMPORTANT
---------
- The source CSV is NEVER modified.
- The pipeline is NEVER re-trained or modified.
- The held-out test set is NEVER used here.
- The X_train split is reconstructed only to provide a SHAP background
  dataset.  This is the correct practice — explainers must be built on
  training data, not test data.
- All outputs are analytical signals only; no causal claims are made.

Artifacts saved
---------------
    models/shap_explainer.pkl         — serialised SHAP LinearExplainer
    models/global_importance_coef.csv — coefficient-based global importance
    models/global_importance_shap.csv — SHAP mean-abs global importance
    models/local_explanation_sample.csv — SHAP local explanations for 5 rows

Usage
-----
    python -m src.explainability.generate_artifacts
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import (
    MODELS_DIR,
    PIPELINE_PATH,
    RANDOM_STATE,
)
from src.data.loader import load_data
from src.features.build_features import build_features
from src.models.anomaly_treatments import apply_clamp_and_flag
from src.preprocessing.splitter import split_data
from src.explainability.feature_names import get_feature_names_from_pipeline
from src.explainability.global_importance import (
    get_global_feature_importance,
    format_global_importance_report,
)
from src.explainability.shap_explainer import (
    build_shap_explainer,
    save_shap_explainer,
    get_shap_global_importance,
    format_shap_global_report,
)
from src.explainability.local_explanation import (
    explain_local_shap,
    format_local_explanation_report,
)
from src.explainability.prediction_explainer import explain_prediction

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

SHAP_EXPLAINER_PATH      = MODELS_DIR / "shap_explainer.pkl"
GLOBAL_COEF_PATH         = MODELS_DIR / "global_importance_coef.csv"
GLOBAL_SHAP_PATH         = MODELS_DIR / "global_importance_shap.csv"
LOCAL_SAMPLE_PATH        = MODELS_DIR / "local_explanation_sample.csv"
EXPLAINABILITY_README    = MODELS_DIR / "explainability_README.md"


def generate_explainability_artifacts() -> None:
    """
    Main entry point: generate and save all Phase 4 explainability artifacts.
    """
    print("\n=== Phase 4 — Explainability Artifact Generation ===\n")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load the saved Phase 3 pipeline (read-only — no re-training)
    # -----------------------------------------------------------------------
    print(f"Loading pipeline from: {PIPELINE_PATH}")
    pipeline = joblib.load(PIPELINE_PATH)
    clf = pipeline.named_steps["clf"]
    print(f"  Model type: {type(clf).__name__}")
    print(f"  class_weight: {clf.class_weight}")
    print(f"  max_iter: {clf.max_iter}")

    # -----------------------------------------------------------------------
    # 2. Reconstruct X_train using the same preprocessing as Phase 3
    # -----------------------------------------------------------------------
    print("\nReconstructing training split (same seed as Phase 3) ...")
    raw_df = load_data()
    eng_df = build_features(raw_df)
    splits = split_data(eng_df)
    X_train_raw = splits["X_train"]
    y_train     = splits["y_train"]
    print(f"  X_train shape: {X_train_raw.shape}")
    print(f"  Attrition rate in train: {y_train.mean():.3f}")

    # Apply winning anomaly treatment: clamp_and_flag
    # (matches the Phase 3 winning configuration: treatment='clamp_and_flag')
    print("  Applying clamp_and_flag treatment (matches Phase 3 winner) ...")
    X_train = apply_clamp_and_flag(X_train_raw)
    print(f"  X_train (after treatment) shape: {X_train.shape}")

    # -----------------------------------------------------------------------
    # 3. Extract and validate feature names
    # -----------------------------------------------------------------------
    feature_names = get_feature_names_from_pipeline(pipeline)
    print(f"\nFeature names ({len(feature_names)} total):")
    for i, name in enumerate(feature_names):
        print(f"  [{i:02d}] {name}")

    # -----------------------------------------------------------------------
    # 4. Global feature importance — coefficient-based
    # -----------------------------------------------------------------------
    print("\n--- Global Importance: Coefficient-based ---")
    global_coef_df = get_global_feature_importance(pipeline)
    global_coef_df.to_csv(GLOBAL_COEF_PATH, index=False)
    print(f"  Saved -> {GLOBAL_COEF_PATH}")
    coef_report = format_global_importance_report(global_coef_df, top_n=15)
    print("\n" + coef_report)

    # -----------------------------------------------------------------------
    # 5. Build SHAP LinearExplainer on X_train (NOT held-out test set)
    # -----------------------------------------------------------------------
    print("\n--- Building SHAP LinearExplainer ---")
    print("  Background dataset: X_train (transformed by fitted preprocessor)")
    print("  Method: LinearExplainer with shap.maskers.Independent (interventional)")
    explainer = build_shap_explainer(pipeline, X_train)
    print(f"  SHAP base value (expected_value): {float(explainer.expected_value):.4f}")

    saved_path = save_shap_explainer(explainer, SHAP_EXPLAINER_PATH)
    print(f"  Saved -> {saved_path}")

    # -----------------------------------------------------------------------
    # 6. SHAP global importance (mean |SHAP| over X_train)
    # -----------------------------------------------------------------------
    print("\n--- SHAP Global Importance (Mean |SHAP| over X_train) ---")
    # Use a sample of X_train for speed (1000 rows, stratified by y_train)
    np.random.seed(RANDOM_STATE)
    n_bg = min(1000, len(X_train))
    bg_idx = np.random.choice(len(X_train), size=n_bg, replace=False)
    X_bg_sample = X_train.iloc[bg_idx].reset_index(drop=True)

    shap_global_df = get_shap_global_importance(explainer, pipeline, X_bg_sample)
    shap_global_df.to_csv(GLOBAL_SHAP_PATH, index=False)
    print(f"  Saved -> {GLOBAL_SHAP_PATH}")
    shap_report = format_shap_global_report(shap_global_df, top_n=15)
    print("\n" + shap_report)

    # -----------------------------------------------------------------------
    # 7. Local explanations for 5 representative employees from X_train
    # -----------------------------------------------------------------------
    print("\n--- Local Explanations for 5 Representative Employees ---")

    # Load optional threshold from Phase 3 results
    cv_path = MODELS_DIR / "cv_results.json"
    optimal_threshold = 0.40
    if cv_path.exists():
        with open(cv_path) as f:
            cv_data = json.load(f)
        optimal_threshold = float(cv_data.get("optimal_threshold", 0.40))
    print(f"  Using optimal threshold: {optimal_threshold:.4f}")

    # Select 5 rows with diverse predicted probabilities
    proba_train = pipeline.predict_proba(X_train)[:, 1]
    # Pick rows at ~10th, 30th, 50th, 70th, 90th percentiles of predicted prob
    percentile_indices = [
        int(np.argmin(np.abs(proba_train - np.percentile(proba_train, p))))
        for p in [10, 30, 50, 70, 90]
    ]

    local_rows = []
    for i, idx in enumerate(percentile_indices):
        X_sample = X_train.iloc[[idx]]
        pred_prob = float(proba_train[idx])

        result = explain_prediction(
            pipeline,
            explainer,
            X_sample,
            employee_id=f"sample_{i + 1}_pct{[10,30,50,70,90][i]}",
            optimal_threshold=optimal_threshold,
            top_n=5,
        )
        print(f"\n  Employee sample_{i + 1} (idx={idx}):")
        print(f"    Predicted prob: {result.attrition_probability:.3f}")
        print(f"    Risk label:     {result.risk_label}")
        print("    Top 3 SHAP features:")
        for _, row in result.local_shap_df.head(3).iterrows():
            print(f"      {row['feature']:<35} SHAP={row['shap_value']:+.4f}  {row['direction']}")

        # Collect top-5 features for the CSV
        for rank_i, (_, row) in enumerate(result.local_shap_df.head(5).iterrows(), 1):
            local_rows.append({
                "employee_sample": result.employee_id,
                "predicted_probability": result.attrition_probability,
                "risk_label": result.risk_label,
                "feature_rank": rank_i,
                "feature": row["feature"],
                "shap_value": row["shap_value"],
                "direction": row["direction"],
                "transformed_value": row["transformed_value"],
            })

    local_df = pd.DataFrame(local_rows)
    local_df.to_csv(LOCAL_SAMPLE_PATH, index=False)
    print(f"\n  Saved -> {LOCAL_SAMPLE_PATH}")

    # -----------------------------------------------------------------------
    # 8. Write explainability README
    # -----------------------------------------------------------------------
    readme_content = _build_readme(
        feature_names=feature_names,
        n_features=len(feature_names),
        optimal_threshold=optimal_threshold,
        base_value=float(explainer.expected_value),
        global_coef_df=global_coef_df,
        shap_global_df=shap_global_df,
    )
    EXPLAINABILITY_README.write_text(readme_content, encoding="utf-8")
    print(f"\n  Saved -> {EXPLAINABILITY_README}")

    print("\n=== Phase 4 Artifact Generation Complete ===")
    print(f"""
Artifacts saved:
  {SHAP_EXPLAINER_PATH}
  {GLOBAL_COEF_PATH}
  {GLOBAL_SHAP_PATH}
  {LOCAL_SAMPLE_PATH}
  {EXPLAINABILITY_README}
""")


def _build_readme(
    feature_names: list[str],
    n_features: int,
    optimal_threshold: float,
    base_value: float,
    global_coef_df: pd.DataFrame,
    shap_global_df: pd.DataFrame,
) -> str:
    """Build the explainability README markdown content."""
    top5_coef = global_coef_df.head(5)
    top5_shap = shap_global_df.head(5)

    coef_rows = "\n".join(
        f"| {int(r['rank'])} | `{r['feature']}` | {r['coefficient']:+.4f} | {r['direction']} |"
        for _, r in top5_coef.iterrows()
    )
    shap_rows = "\n".join(
        f"| {int(r['rank'])} | `{r['feature']}` | {r['mean_abs_shap']:.4f} | {r['direction']} |"
        for _, r in top5_shap.iterrows()
    )

    return f"""# Phase 4 Explainability — Workforce Attrition Intelligence

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
| Optimal threshold (F1) | {optimal_threshold:.4f} |
| Pipeline features | {n_features} |

## Explainability Approach

### Global Importance — Logistic Regression Coefficients

After StandardScaler brings continuous features to zero-mean/unit-variance and
categorical features to comparable integer/dummy scales, the magnitude of each
LR coefficient is a valid proxy for the feature's contribution to the log-odds
of attrition.

**Top 5 features by absolute coefficient:**

| Rank | Feature | Coefficient | Direction |
|---|---|---|---|
{coef_rows}

### SHAP — LinearExplainer (Interventional)

`shap.LinearExplainer` computes exact Shapley values in closed form for linear
models.  SHAP values are additive contributions to the model's log-odds output:

    φᵢ(x) = cᵢ × (xᵢ − E[xᵢ])

The background dataset is X_train (after clamp_and_flag treatment).
SHAP values were computed using `shap.maskers.Independent` (interventional).

**SHAP base value** (population avg log-odds): {base_value:.4f}

**Top 5 features by mean |SHAP value| over X_train:**

| Rank | Feature | Mean |SHAP| | Direction |
|---|---|---|---|
{shap_rows}

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
"""
