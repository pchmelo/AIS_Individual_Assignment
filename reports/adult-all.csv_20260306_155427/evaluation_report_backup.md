# Dataset Fairness Evaluation Report

## Metadata

- **Dataset:** adult-all.csv
- **Timestamp:** 2026-03-06 15:55:23
- **Dataset Hash:** 5dba2d39
- **Target Column:** Income
- **Objective:** Evaluate the dataset 'adult-all.csv' for data quality and fairness issues. Target: Income. Provide a detailed report highlighting any problems found and suggestions for improvement.

---

## Stage 0: Dataset Loading

**Tool Used:** `load_dataset`

### Analysis

## Dataset Summary

The **adult-all.csv** dataset contains **48,842 rows** and **15 columns**, representing demographic and employment information used to predict income levels.

### Key Characteristics:
- **Target variable**: `Income` (binary classification: `<=50K` or `>50K`)
- **Features include**: Age, Workclass, Education, Marital-status, Occupation, Relationship, Race, Sex, Capital-gain, Capital-loss, Hours-per-week, Native-country, and weighting variables
- **Data type**: Mixed categorical and numerical features
- **Source**: Classic UCI Adult Income dataset (Census data)

### Sample Observations:
The preview shows typical entries with middle-aged adults, predominantly White males, working in various sectors (State-gov, Self-emp-not-inc, Private), with education levels ranging from HS-grad to Bachelors. Most individuals work 40 hours per week, and the majority are from the United States.

This dataset is commonly used for binary classification tasks and fairness analysis, particularly regarding income prediction and potential biases across sensitive attributes like Race, Sex, and Age.

---

## Stage 1: Objective Inspection

**Objective:** Evaluate the dataset 'adult-all.csv' for data quality and fairness issues. Target: Income. Provide a detailed report highlighting any problems found and suggestions for improvement.

**Validation:** Dataset format compatible (CSV)

---

## Stage 2: Data Quality Analysis

**Tool Used:** `check_missing_data`

### Analysis

## Missing Data Analysis Report: adult-all.csv

### Dataset Overview
- Total rows: 48,842
- Total missing values: 6,465
- Overall missing percentage: 0.88%
- Columns with missing data issues: 4

### Detailed Column Analysis

1. Workclass
   - Missing count: 2,799 (5.73%)
   - Missing values encoded as: '?'
   - Data type: object (categorical)

2. Occupation
   - Missing count: 2,809 (5.75%)
   - Missing values encoded as: '?'
   - Data type: object (categorical)

3. Capital-gain
   - Missing count: 0 (0.0%)
   - Suspicious pattern: value 99999 appears 244 times (0.50%)
   - Data type: int64 (numeric)
   - Note: 99999 is likely a placeholder for missing data

4. Native-country
   - Missing count: 857 (1.75%)
   - Missing values encoded as: '?'
   - Data type: object (categorical)

### Key Findings

1. Consistent encoding pattern: Three categorical columns use '?' to represent missing values, indicating systematic data collection issues.

2. Hidden missing data: Capital-gain column contains 244 instances of 99999, a common placeholder for missing values in the Adult dataset. This should be treated as missing data despite having 0 official missing values.

3. Missing data distribution: Workclass and Occupation have nearly identical missing rates (~5.7%), suggesting these may be related - possibly respondents refused to answer both questions or data was lost together.

4. Impact severity: With 6,465 total missing values, approximately 13.2% of rows have at least one missing value (assuming no overlap), which could affect model training if not handled properly.

### Recommendations

1. Standardize missing value representation: Convert all '?' values to proper NaN/None values before analysis.

2. Treat 99999 in Capital-gain as missing: Replace 99999 with NaN to maintain data integrity.

3. Investigate correlation: Check if missing values in Workclass and Occupation occur in the same rows, which might indicate a systematic pattern.

4. Imputation strategy: For categorical variables with ~5% missing data, consider mode imputation or creating a "Missing" category. For Capital-gain, consider median imputation or using a model-based approach given it's numeric.

5. Document decisions: Record how missing values are handled for reproducibility and to avoid bias introduction.

---

## Stage 3: Sensitive Attribute Detection

**Tool Used:** `detect_sensitive_attributes`

### Analysis

1. Column: Age | Reason: Demographic attribute representing age | Values: [39, 50, 38, 53, 28]
2. Column: Education | Reason: Socioeconomic attribute indicating education level | Values: ['HS-grad', 'Some-college', 'Bachelors']
3. Column: Education Number of Years | Reason: Socioeconomic attribute representing years of education | Values: [9, 10, 13]
4. Column: Marital-status | Reason: Personal attribute indicating marital status | Values: ['Married-civ-spouse', 'Never-married', 'Divorced']
5. Column: Occupation | Reason: Socioeconomic attribute representing occupation | Values: ['Prof-specialty', 'Craft-repair', 'Exec-managerial']
6. Column: Relationship | Reason: Personal attribute indicating relationship status | Values: ['Husband', 'Not-in-family', 'Own-child']
7. Column: Race | Reason: Demographic attribute indicating race | Values: ['White', 'Black', 'Asian-Pac-Islander']
8. Column: Sex | Reason: Demographic attribute indicating sex/gender | Values: ['Male', 'Female']
9. Column: Native-country | Reason: Geographic attribute indicating native country | Values: ['United-States', 'Mexico', '?']

---

*Report generated by Dataset Fairness Evaluation System*