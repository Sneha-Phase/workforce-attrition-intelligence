"""
src/config.py
=============
Central configuration for the Workforce Attrition Intelligence project.

ALL stochastic operations must receive RANDOM_STATE.
ALL file paths must be resolved via the path constants below.
No configuration should be hard-coded in individual modules.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Paths  (resolved relative to the project root, not the calling file)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR        = PROJECT_ROOT / "data"
MODELS_DIR      = PROJECT_ROOT / "models"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"

RAW_CSV = DATA_DIR / "employee_attrition_dataset_10000.csv"

# Serialised model artifacts (written by train.py, read by app)
PIPELINE_PATH      = MODELS_DIR / "best_pipeline.pkl"
SHAP_VALUES_PATH   = MODELS_DIR / "shap_values.pkl"
CV_RESULTS_PATH    = MODELS_DIR / "cv_results.json"
MODEL_COMPARE_PATH = MODELS_DIR / "model_comparison.csv"

# ---------------------------------------------------------------------------
# Dataset schema contract
# ---------------------------------------------------------------------------
EXPECTED_SHAPE = (10_000, 26)

REQUIRED_COLUMNS = [
    "Employee_ID",
    "Age",
    "Gender",
    "Marital_Status",
    "Department",
    "Job_Role",
    "Job_Level",
    "Monthly_Income",
    "Hourly_Rate",
    "Years_at_Company",
    "Years_in_Current_Role",
    "Years_Since_Last_Promotion",
    "Work_Life_Balance",
    "Job_Satisfaction",
    "Performance_Rating",
    "Training_Hours_Last_Year",
    "Overtime",
    "Project_Count",
    "Average_Hours_Worked_Per_Week",
    "Absenteeism",
    "Work_Environment_Satisfaction",
    "Relationship_with_Manager",
    "Job_Involvement",
    "Distance_From_Home",
    "Number_of_Companies_Worked",
    "Attrition",
]

TARGET_COLUMN   = "Attrition"
ID_COLUMN       = "Employee_ID"
TARGET_POSITIVE = "Yes"   # the minority / attrition class
TARGET_NEGATIVE = "No"

# ---------------------------------------------------------------------------
# Feature groups (used by pipeline and feature engineering)
# ---------------------------------------------------------------------------
CATEGORICAL_NOMINAL = ["Department", "Job_Role", "Marital_Status"]
CATEGORICAL_BINARY  = ["Gender", "Overtime"]
ORDINAL_FEATURES    = [
    "Job_Level",
    "Work_Life_Balance",
    "Job_Satisfaction",
    "Performance_Rating",
    "Work_Environment_Satisfaction",
    "Relationship_with_Manager",
    "Job_Involvement",
    "Number_of_Companies_Worked",
]

# ---------------------------------------------------------------------------
# Anomaly definition (used in feature engineering and tests)
# ---------------------------------------------------------------------------
ANOMALY_COLUMN   = "Years_in_Current_Role"
ANOMALY_BASELINE = "Years_at_Company"
# Number of anomalous rows confirmed during profiling
EXPECTED_ANOMALY_COUNT = 2_284

# ---------------------------------------------------------------------------
# Class imbalance
# ---------------------------------------------------------------------------
# Approximate split from profiling: 80.03% No / 19.97% Yes
MINORITY_CLASS_APPROX_PCT = 0.1997
