# Dataset Profile — Workforce Attrition Intelligence
**File:** `data/employee_attrition_dataset_10000.csv`  
**Profiled on:** Initial inspection pass (read-only; no modifications made)

---

## 1. Shape

| Metric   | Value |
|----------|-------|
| Rows     | 10,000 |
| Columns  | 26 |

---

## 2. Column Inventory

| # | Column | dtype | Unique Values | Missing | Missing % |
|---|--------|-------|---------------|---------|-----------|
| 1 | Employee_ID | int64 | 10,000 | 0 | 0.00% |
| 2 | Age | int64 | 40 | 0 | 0.00% |
| 3 | Gender | object | 2 | 0 | 0.00% |
| 4 | Marital_Status | object | 3 | 0 | 0.00% |
| 5 | Department | object | 5 | 0 | 0.00% |
| 6 | Job_Role | object | 4 | 0 | 0.00% |
| 7 | Job_Level | int64 | 5 | 0 | 0.00% |
| 8 | Monthly_Income | int64 | 7,586 | 0 | 0.00% |
| 9 | Hourly_Rate | int64 | 85 | 0 | 0.00% |
| 10 | Years_at_Company | int64 | 29 | 0 | 0.00% |
| 11 | Years_in_Current_Role | int64 | 14 | 0 | 0.00% |
| 12 | Years_Since_Last_Promotion | int64 | 10 | 0 | 0.00% |
| 13 | Work_Life_Balance | int64 | 4 | 0 | 0.00% |
| 14 | Job_Satisfaction | int64 | 5 | 0 | 0.00% |
| 15 | Performance_Rating | int64 | 4 | 0 | 0.00% |
| 16 | Training_Hours_Last_Year | int64 | 100 | 0 | 0.00% |
| 17 | Overtime | object | 2 | 0 | 0.00% |
| 18 | Project_Count | int64 | 9 | 0 | 0.00% |
| 19 | Average_Hours_Worked_Per_Week | int64 | 30 | 0 | 0.00% |
| 20 | Absenteeism | int64 | 20 | 0 | 0.00% |
| 21 | Work_Environment_Satisfaction | int64 | 4 | 0 | 0.00% |
| 22 | Relationship_with_Manager | int64 | 4 | 0 | 0.00% |
| 23 | Job_Involvement | int64 | 4 | 0 | 0.00% |
| 24 | Distance_From_Home | int64 | 49 | 0 | 0.00% |
| 25 | Number_of_Companies_Worked | int64 | 4 | 0 | 0.00% |
| 26 | Attrition | object | 2 | 0 | 0.00% |

> **No missing values detected in any column.** The dataset is complete.

---

## 3. Target Variable

**Column:** `Attrition`  
**Type:** Binary categorical (string) — values: `"Yes"` / `"No"`

### Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| No (retained) | 8,003 | 80.03% |
| Yes (left) | 1,997 | 19.97% |

**Imbalance ratio ≈ 4:1.** This is a moderately imbalanced binary classification problem. Techniques such as SMOTE, class-weight balancing, or threshold tuning will be needed during modelling. Evaluation metrics should favour F1-score, ROC-AUC, and PR-AUC over raw accuracy.

---

## 4. Feature Classification

### 4a. Numerical Features (19)

| Column | Min | Max | Mean | Std | Notes |
|--------|-----|-----|------|-----|-------|
| Age | 20 | 59 | 39.6 | 11.5 | Continuous integer |
| Job_Level | 1 | 5 | 3.0 | 1.4 | Ordinal (1–5) |
| Monthly_Income | 3,000 | 19,999 | 11,437 | 4,927 | Wide range; right-skew likely |
| Hourly_Rate | 15 | 99 | 57.0 | 24.7 | Mostly uniform |
| Years_at_Company | 1 | 29 | 14.9 | 8.4 | Tenure |
| Years_in_Current_Role | 1 | 14 | 7.5 | 4.0 | Role tenure |
| Years_Since_Last_Promotion | 0 | 9 | 4.5 | 2.9 | Stagnation indicator |
| Work_Life_Balance | 1 | 4 | 2.5 | 1.1 | Ordinal Likert |
| Job_Satisfaction | 1 | 5 | 3.0 | 1.4 | Ordinal Likert |
| Performance_Rating | 1 | 4 | 2.5 | 1.1 | Ordinal Likert |
| Training_Hours_Last_Year | 0 | 99 | 49.6 | 28.8 | Continuous; near-uniform |
| Project_Count | 1 | 9 | 5.0 | 2.6 | Discrete count |
| Average_Hours_Worked_Per_Week | 30 | 59 | 44.5 | 8.6 | Workload indicator |
| Absenteeism | 0 | 19 | 9.4 | 5.8 | Discrete count |
| Work_Environment_Satisfaction | 1 | 4 | 2.5 | 1.1 | Ordinal Likert |
| Relationship_with_Manager | 1 | 4 | 2.5 | 1.1 | Ordinal Likert |
| Job_Involvement | 1 | 4 | 2.5 | 1.1 | Ordinal Likert |
| Distance_From_Home | 1 | 49 | 25.3 | 14.2 | Continuous integer |
| Number_of_Companies_Worked | 1 | 4 | 2.5 | 1.1 | Discrete; only 4 unique values |

### 4b. Categorical Features (5)

| Column | Unique Values | Distribution |
|--------|---------------|--------------|
| Gender | Female (5,042), Male (4,958) | Balanced ~50/50 |
| Marital_Status | Married (3,375), Divorced (3,330), Single (3,295) | Evenly distributed |
| Department | Marketing (2,133), Sales (2,008), Finance (1,990), HR (1,953), IT (1,916) | Evenly distributed |
| Job_Role | Analyst (2,572), Assistant (2,538), Executive (2,476), Manager (2,414) | Evenly distributed |
| Overtime | No (5,103), Yes (4,897) | Balanced ~50/50 |

---

## 5. Potential ID Columns (Exclude from Modelling)

| Column | Reason |
|--------|--------|
| `Employee_ID` | Sequential integer surrogate key (1–10,000). Unique per row. Carries no predictive signal. **Must be dropped before training.** |

---

## 6. Duplicate Rows

**Zero duplicate rows detected.** All 10,000 rows are unique.

---

## 7. Suspicious / Invalid Values

### ⚠️ Critical Finding — `Years_in_Current_Role` > `Years_at_Company`

**2,284 rows (22.84%)** have `Years_in_Current_Role` strictly greater than `Years_at_Company`. This is logically impossible — an employee cannot have been in their current role longer than their total tenure at the company.

Sample:

| Employee_ID | Years_at_Company | Years_in_Current_Role |
|-------------|-----------------|----------------------|
| 5 | 3 | 9 |
| 17 | 2 | 14 |
| 29 | 5 | 8 |
| 30 | 2 | 14 |

**Interpretation:** This appears to be a data quality / synthetic data artefact. Suggested handling options:
- Clip `Years_in_Current_Role` to `min(Years_in_Current_Role, Years_at_Company)`
- Create a derived flag `Role_Tenure_Anomaly` (1/0) and investigate further
- Treat `Years_in_Current_Role` as unreliable; consider dropping or correcting

### ⚠️ Notable Finding — `Job_Level` vs `Monthly_Income` No Correlation

Job levels 1–5 have nearly identical income distributions (median ≈ $11,400–$11,726 across all levels). In real-world HR data, income increases sharply with job level. This strongly suggests either:
- The income and job-level columns were generated independently (synthetic data artefact), or
- `Monthly_Income` is not meaningfully associated with `Job_Level` in this dataset.

**Implication for modelling:** `Job_Level` and `Monthly_Income` cannot be treated as proxies for each other; they carry independent (though possibly weak) signals.

### Other Range Checks — All Pass

All other numerical columns fall within plausible human-resource ranges. No negative values, no extreme outliers beyond expected domain bounds.

---

## 8. Correlation with Target (Pearson r, binary Attrition)

> All correlations are weak (|r| < 0.02), consistent with a synthetic dataset. **No single feature is a strong linear predictor.** Non-linear ensemble models (XGBoost, Random Forest) should capture interactions that linear correlation misses.

| Feature | r with Attrition |
|---------|-----------------|
| Work_Life_Balance | +0.0152 |
| Job_Involvement | +0.0150 |
| Performance_Rating | +0.0119 |
| Job_Level | +0.0114 |
| Distance_From_Home | +0.0114 |
| Work_Environment_Satisfaction | −0.0111 |
| Number_of_Companies_Worked | −0.0082 |
| Age | +0.0078 |
| … | … |

---

## 9. Attrition Rate by Segment

### By Department
| Department | Attrition Rate |
|------------|---------------|
| Finance | 20.85% |
| IT | 20.35% |
| Sales | 19.82% |
| HR | 19.51% |
| Marketing | 19.36% |

All departments have near-identical attrition rates (~20%). This suggests `Department` may have low predictive value in isolation, consistent with the low linear correlations observed.

---

## 10. Features Requiring Encoding or Transformation

### Label / Binary Encoding (direct)
| Column | Encoding | Reason |
|--------|----------|--------|
| `Attrition` (target) | `Yes`→1, `No`→0 | Binary label |
| `Overtime` | `Yes`→1, `No`→0 | Binary categorical |
| `Gender` | Label encode | 2 categories |

### One-Hot Encoding (nominal)
| Column | Categories | Reason |
|--------|------------|--------|
| `Department` | 5 | No ordinal relationship |
| `Job_Role` | 4 | No ordinal relationship |
| `Marital_Status` | 3 | No ordinal relationship |

### Ordinal / Keep as-is (natural order confirmed)
| Column | Scale | Action |
|--------|-------|--------|
| `Job_Level` | 1–5 | Keep numeric or ordinal-encode |
| `Work_Life_Balance` | 1–4 | Keep numeric |
| `Job_Satisfaction` | 1–5 | Keep numeric |
| `Performance_Rating` | 1–4 | Keep numeric |
| `Work_Environment_Satisfaction` | 1–4 | Keep numeric |
| `Relationship_with_Manager` | 1–4 | Keep numeric |
| `Job_Involvement` | 1–4 | Keep numeric |
| `Number_of_Companies_Worked` | 1–4 | Keep numeric |

### Scaling (recommended for linear/distance-based models)
All continuous features (`Monthly_Income`, `Age`, `Distance_From_Home`, `Training_Hours_Last_Year`, `Years_*`, etc.) should be **StandardScaler** or **MinMaxScaler** scaled when used with logistic regression, SVM, or k-NN. Tree-based models do not require scaling.

---

## 11. Potential Data Leakage Risks

| Column | Risk | Notes |
|--------|------|-------|
| `Absenteeism` | ⚠️ Medium | High absenteeism may be a *consequence* of impending resignation rather than a *cause*. Depending on how this was measured, it could be post-decision behaviour leaking into features. |
| `Training_Hours_Last_Year` | ⚠️ Low-Medium | Companies often reduce training investment for employees identified as flight risks. Could be a lagged proxy. |
| `Performance_Rating` | ⚠️ Low | Managers may rate departing employees lower, making this partially a consequence. |
| `Years_Since_Last_Promotion` | ⚠️ Low | Long promotion gaps can be both a cause and a trailing symptom. |

None of these are direct label leakage, but they warrant careful interpretation of model coefficients and SHAP values.

---

## 12. Feature Engineering Opportunities

| Derived Feature | Formula / Logic | Rationale |
|----------------|-----------------|-----------|
| `Income_to_Role_Ratio` | `Monthly_Income / Job_Level` | Captures under-compensation relative to seniority |
| `Tenure_Promotion_Ratio` | `Years_at_Company / (Years_Since_Last_Promotion + 1)` | Career progression pace |
| `Role_Tenure_Ratio` | `Years_in_Current_Role / Years_at_Company` | Role stagnation index |
| `Role_Tenure_Anomaly` | `1 if Years_in_Current_Role > Years_at_Company else 0` | Flag the 22.84% data quality anomaly |
| `Hours_Overload` | `Average_Hours_Worked_Per_Week - 40` | Deviation from standard work week |
| `Satisfaction_Composite` | Mean of `Job_Satisfaction`, `Work_Life_Balance`, `Work_Environment_Satisfaction`, `Relationship_with_Manager`, `Job_Involvement` | Single well-being index |
| `Absenteeism_per_Year` | `Absenteeism / Years_at_Company` | Normalized absenteeism by tenure |
| `Training_per_Project` | `Training_Hours_Last_Year / Project_Count` | Investment per project responsibility |
| `Is_Single` | `1 if Marital_Status == 'Single' else 0` | Mobility proxy (literature-backed) |
| `Is_Overtime` | Binary encode of `Overtime` | Already binary; rename for clarity |

---

## 13. Summary of Key Findings

| Finding | Status |
|---------|--------|
| Missing values | ✅ None |
| Duplicate rows | ✅ None |
| Target variable | ✅ `Attrition` (Yes/No) |
| Class imbalance | ⚠️ ~4:1 (80% No, 20% Yes) — requires balancing strategy |
| ID column to drop | ⚠️ `Employee_ID` |
| Data quality anomaly | ❌ `Years_in_Current_Role > Years_at_Company` in 22.84% of rows |
| `Job_Level` vs `Monthly_Income` | ⚠️ No expected income gradient — synthetic data artefact |
| Linear correlations with target | ⚠️ All very weak (|r| < 0.02) — non-linear models preferred |
| Features needing encoding | ⚠️ Gender, Overtime, Department, Job_Role, Marital_Status, Attrition |
| Features needing scaling | ⚠️ All continuous numerics (for linear/distance models) |
| Potential leakage | ⚠️ Absenteeism, Training_Hours, Performance_Rating |
| Feature engineering | ✅ 10 candidate derived features identified |

---

## 14. Recommended Next Steps (Pending Approval)

1. **Handle `Years_in_Current_Role` anomaly** — clip or flag before feature engineering
2. **Encode categorical features** — binary for Overtime/Gender, OHE for Department/Job_Role/Marital_Status
3. **Engineer composite features** — satisfaction index, tenure ratios
4. **Drop `Employee_ID`** from feature matrix
5. **Scale numerical features** (pipeline step, not destructive)
6. **Address class imbalance** — evaluate SMOTE, class_weight='balanced', or threshold tuning
7. **Select modelling approach** — recommend starting with XGBoost + SHAP for interpretability

---

*Report generated from read-only dataset inspection. No data files were modified.*
