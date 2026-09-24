"""
src/models/definitions.py
==========================
Model definitions for the Workforce Attrition Intelligence project.

Design principles
-----------------
- Each model factory returns an *unfitted* sklearn estimator.
- ALL stochastic parameters receive RANDOM_STATE from src.config.
- No fitting happens here; fitting is the job of the training orchestration.
- The preprocessor (ColumnTransformer) is assembled into a full Pipeline
  alongside the classifier so that preprocessing is always fitted inside
  CV folds, never on the full dataset.

Models included
---------------
1. Logistic Regression  — linear baseline; calibrated probabilities
2. Random Forest        — ensemble, low-bias tree method
3. Gradient Boosting    — sklearn GradientBoostingClassifier
4. XGBoost              — xgboost.XGBClassifier (additional candidate)

Imbalance strategies
--------------------
For each model, class_weight='balanced' variants are also registered.
SMOTE is handled externally (via imblearn Pipeline) in evaluation.py
only when CV evidence justifies it.
"""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import RANDOM_STATE
from src.preprocessing.pipeline import build_preprocessor


# ---------------------------------------------------------------------------
# Individual estimator factories
# ---------------------------------------------------------------------------

def make_logistic_regression(class_weight: str | None = None) -> LogisticRegression:
    """
    Logistic Regression baseline.

    Parameters
    ----------
    class_weight : str | None
        Pass 'balanced' to weight classes inversely proportional to frequency.

    Returns
    -------
    LogisticRegression (unfitted)
    """
    return LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
    )


def make_random_forest(class_weight: str | None = None) -> RandomForestClassifier:
    """
    Random Forest classifier.

    Parameters
    ----------
    class_weight : str | None
        Pass 'balanced' to weight classes inversely proportional to frequency.

    Returns
    -------
    RandomForestClassifier (unfitted)
    """
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def make_gradient_boosting(scale_pos_weight: float | None = None) -> GradientBoostingClassifier:
    """
    sklearn GradientBoostingClassifier.

    Gradient Boosting does not accept class_weight directly.
    Use scale_pos_weight-equivalent approach: subsample weighting is handled
    via sample_weight in cross_validate calls, or accepted as-is (imbalance
    is addressed by threshold tuning and the balanced variant of other models).

    Parameters
    ----------
    scale_pos_weight : float | None
        Not directly supported by sklearn GBC; parameter is accepted for
        interface consistency but ignored here. Use XGBoost for native support.

    Returns
    -------
    GradientBoostingClassifier (unfitted)
    """
    return GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=RANDOM_STATE,
    )


def make_xgboost(scale_pos_weight: float | None = None) -> XGBClassifier:
    """
    XGBoost classifier.

    Parameters
    ----------
    scale_pos_weight : float | None
        Ratio of negative/positive samples.  Pass ~4.0 (≈80/20) to weight
        the minority class.  If None, no class weighting is applied.

    Returns
    -------
    XGBClassifier (unfitted)
    """
    kwargs: dict = dict(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        verbosity=0,
        n_jobs=-1,
    )
    if scale_pos_weight is not None:
        kwargs["scale_pos_weight"] = scale_pos_weight
    return XGBClassifier(**kwargs)


# ---------------------------------------------------------------------------
# Full sklearn Pipeline factories  (preprocessor + classifier)
# ---------------------------------------------------------------------------

def build_pipeline(classifier) -> Pipeline:
    """
    Wrap a classifier in a leakage-safe sklearn Pipeline with the standard
    preprocessor.

    The preprocessor is constructed fresh for each call so that multiple
    pipelines do not share transformer state.

    Parameters
    ----------
    classifier : sklearn estimator
        Any unfitted classifier (e.g. from the factories above).

    Returns
    -------
    sklearn.pipeline.Pipeline (unfitted)
        Steps: [("preprocessor", ColumnTransformer), ("clf", classifier)]
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("clf", classifier),
        ]
    )


# ---------------------------------------------------------------------------
# Catalogue of named model configurations
# ---------------------------------------------------------------------------
# Each entry: (name, pipeline)
# name conventions:
#   LR            → Logistic Regression, no class weighting
#   LR_balanced   → Logistic Regression, class_weight='balanced'
#   RF            → Random Forest, no class weighting
#   RF_balanced   → Random Forest, class_weight='balanced'
#   GB            → sklearn Gradient Boosting (no native class weighting)
#   XGB           → XGBoost, no class weighting
#   XGB_balanced  → XGBoost, scale_pos_weight≈4.0 (≈ 80/20 ratio)

# Approximate ratio of majority to minority class (80/20 → ~4.0)
_SPW = 4.0  # scale_pos_weight for XGBoost balanced variant


def get_model_catalogue() -> dict[str, Pipeline]:
    """
    Return all named model pipelines as an ordered dict.

    The catalogue contains exactly the combinations needed for Phase 3:
    - 3 model families × 2 imbalance strategies (where applicable) + GB
    - Total: 7 pipeline configurations

    Returns
    -------
    dict[str, Pipeline]
        Mapping of model_name → unfitted Pipeline.
    """
    return {
        "LR":           build_pipeline(make_logistic_regression()),
        "LR_balanced":  build_pipeline(make_logistic_regression(class_weight="balanced")),
        "RF":           build_pipeline(make_random_forest()),
        "RF_balanced":  build_pipeline(make_random_forest(class_weight="balanced")),
        "GB":           build_pipeline(make_gradient_boosting()),
        "XGB":          build_pipeline(make_xgboost()),
        "XGB_balanced": build_pipeline(make_xgboost(scale_pos_weight=_SPW)),
    }
