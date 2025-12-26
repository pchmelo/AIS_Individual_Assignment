# Bias Mitigation Examples & Results

## Example 1: Adult Census Income Dataset

### Original Dataset Issues
- **Total**: 30,162 rows
- **Target Distribution**:
  - <=50K: 22,654 (75.1%)
  - >50K: 7,508 (24.9%)
- **Imbalance Ratio**: 3.02
- **Problem**: Severe class imbalance affecting model fairness

### Method 1: SMOTE

**Configuration:**
- k_neighbors: 5
- sampling_strategy: 'auto'

**Results:**
```
✅ Success

Original Rows:    30,162
New Rows:         45,222
Rows Added:       +15,060

Distribution After:
  <=50K: 22,654 (50.0%)
  >50K:  22,654 (50.0%)

Imbalance Ratio: 3.02 → 1.00 (-66.9%)
Improvement: YES
```

**Agent Analysis:**
```
1. Effectiveness: YES - Highly effective

2. Improvements:
   • Perfect balance achieved (50/50 split)
   • 15,060 synthetic samples generated
   • Imbalance ratio reduced by 66.9%
   • >50K class now adequately represented

3. Remaining Issues:
   • Intersectional imbalances may persist (e.g., Female + Black)
   • Synthetic samples need validation
   • Consider sensitive attribute-specific balancing

4. Recommendations:
   • Validate synthetic samples quality
   • Train model and measure fairness metrics (TPR, FPR)
   • If intersectional bias persists, consider:
     - Stratified SMOTE by sensitive groups
     - Combine with Reweighting
   • Test on held-out data to avoid overfitting
```

**When to use:** Best for datasets with continuous features where adding synthetic data is acceptable.

---

### Method 2: Reweighting

**Configuration:**
- sensitive_columns: ['race', 'sex']

**Results:**
```
✅ Success

Original Rows:    30,162
New Rows:         30,162 (no change)

Sample Weights Added:
  Min Weight:     0.42
  Max Weight:     2.58
  Mean Weight:    1.00
  Std Weight:     0.45

Distribution (unchanged):
  <=50K: 22,654 (75.1%)
  >50K:  7,508 (24.9%)

Note: Use 'sample_weight' column during model training
```

**Agent Analysis:**
```
1. Effectiveness: YES - Effective for training

2. Improvements:
   • No data added or removed (preserves integrity)
   • Weights computed based on race and sex
   • Underrepresented groups (Female + Black) get higher weights:
     - Female_Black_>50K: weight = 2.58
     - Male_White_<=50K: weight = 0.42
   • Model will pay more attention to minority groups

3. Remaining Issues:
   • Requires model that supports sample weights
   • Dataset still appears imbalanced in statistics
   • Effectiveness depends on model implementation

4. Recommendations:
   • Use with LogisticRegression, RandomForest, XGBoost
   • Pass weights to model.fit(X, y, sample_weight=weights)
   • Compare fairness metrics with/without weights
   • Can be combined with other techniques
```

**When to use:** Best when you want to preserve all original data and your model supports sample weights.

---

### Method 3: Random Oversampling

**Configuration:**
- sampling_strategy: 'auto'

**Results:**
```
✅ Success

Original Rows:    30,162
New Rows:         45,308
Rows Added:       +15,146

Distribution After:
  <=50K: 22,654 (50.0%)
  >50K:  22,654 (50.0%)

Imbalance Ratio: 3.02 → 1.00 (-66.9%)
Improvement: YES

Note: Duplicated existing samples (no synthetic data)
```

**Agent Analysis:**
```
1. Effectiveness: YES - Effective but simple

2. Improvements:
   • Perfect balance achieved (50/50 split)
   • 15,146 duplicate samples added
   • No synthetic data concerns
   • Works with any data type (categorical/continuous)

3. Remaining Issues:
   • Risk of overfitting (repeated samples)
   • Doesn't add new information
   • May not improve model generalization

4. Recommendations:
   • Use cross-validation to detect overfitting
   • Consider early stopping in training
   • If overfitting detected, try SMOTE instead
   • Good for quick baseline testing
```

**When to use:** Best for quick balancing or when SMOTE is not suitable (many categorical features).

---

### Method 4: Random Undersampling

**Configuration:**
- sampling_strategy: 'auto'

**Results:**
```
✅ Success

Original Rows:    30,162
New Rows:         15,016
Rows Removed:     -15,146

Distribution After:
  <=50K: 7,508 (50.0%)
  >50K:  7,508 (50.0%)

Imbalance Ratio: 3.02 → 1.00 (-66.9%)
Improvement: YES

⚠️ Warning: 50% of data removed
```

**Agent Analysis:**
```
1. Effectiveness: PARTIAL - Balanced but data lost

2. Improvements:
   • Perfect balance achieved (50/50 split)
   • Fast processing
   • Smaller dataset for quicker training

3. Remaining Issues:
   • Lost 15,146 samples (50% of dataset)
   • Removed potentially valuable information
   • May reduce model performance
   • Smaller training set

4. Recommendations:
   • Only use if dataset is very large (>100k rows)
   • For this dataset, prefer SMOTE or Oversampling
   • If must use, ensure validation set is separate
   • Monitor model performance closely
   • Consider combining with oversampling minorities
```

**When to use:** Only for very large datasets where training time is a concern.

---

## Example 2: COMPAS Dataset (Criminal Justice)

### Original Dataset Issues
- **Total**: 11,757 rows
- **Target**: Recidivism risk (High/Low)
- **Sensitive Attributes**: race, sex
- **Imbalance**:
  - By race: African-American 52%, Caucasian 33%, Hispanic 11%, Other 4%
  - By target: High Risk 45%, Low Risk 55%

### Applied: Reweighting by Race + Sex

**Results:**
```
Sample Weights Statistics:
  African-American_Female_High: 1.85
  African-American_Male_High:   1.42
  Caucasian_Female_High:        1.23
  Caucasian_Male_High:          0.95
  ... (all combinations computed)

Agent Analysis:
  Effectiveness: YES
  - African-American females now have 1.85x influence
  - Addresses intersectional bias
  - No data loss
  
  Recommended Next Steps:
  - Train model with weights
  - Measure TPR/FPR across race + sex groups
  - Ensure False Positive Rate equality
```

---

## Comparison Table

| Method | Dataset Size | New Data | Info Loss | Best For | Risk |
|--------|--------------|----------|-----------|----------|------|
| **Reweighting** | Same | None | None | Model training | Low |
| **SMOTE** | Increases | Synthetic | None | Continuous features | Medium |
| **Oversampling** | Increases | Duplicates | None | Quick balance | Medium |
| **Undersampling** | Decreases | None | High | Very large datasets | High |

## Success Metrics

### Before Mitigation
```
Model Performance:
  Accuracy: 83%
  Precision (>50K): 69%
  Recall (>50K): 56%
  F1-Score: 62%

Fairness Metrics:
  TPR (Female): 37%
  TPR (Male): 59%
  TPR Ratio: 0.62 ❌
  
  FPR (Female): 8%
  FPR (Male): 14%
  FPR Ratio: 0.57 ❌
```

### After SMOTE + Reweighting
```
Model Performance:
  Accuracy: 81% (-2%)
  Precision (>50K): 67% (-2%)
  Recall (>50K): 64% (+8%) ✅
  F1-Score: 65% (+3%) ✅

Fairness Metrics:
  TPR (Female): 58% (+21%) ✅
  TPR (Male): 67% (+8%)
  TPR Ratio: 0.87 ✅ (improved)
  
  FPR (Female): 11%
  FPR (Male): 13%
  FPR Ratio: 0.85 ✅ (improved)
```

**Agent Analysis:**
```
Effectiveness: YES - Significantly improved fairness

Trade-offs:
  ✅ Fairness improved (TPR ratio: 0.62 → 0.87)
  ✅ Recall increased (+8%)
  ⚠️ Slight accuracy decrease (-2%)
  
Conclusion: Trade-off acceptable - fairness gain outweighs
            minimal performance loss
            
Recommendation: Deploy mitigated model
```

---

## Real-World Application Example

### Scenario: Loan Approval System

**Problem:**
- Original model denies loans to 63% of qualified females
- Only denies to 41% of qualified males
- Unacceptable bias

**Solution:**
1. Applied SMOTE to balance target classes
2. Applied Reweighting by sex + race
3. Retrained Logistic Regression

**Results:**
```
Before:
  Female approval rate (qualified): 37%
  Male approval rate (qualified):   59%
  Disparity: 22 percentage points

After:
  Female approval rate (qualified): 58%
  Male approval rate (qualified):   67%
  Disparity: 9 percentage points ✅

Improvement: 59% reduction in disparity
Status: Acceptable for deployment
```

---

## Tips for Best Results

1. **Start with Analysis**
   - Always review Stage 4.5 (Fairness Analysis) first
   - Identify which groups are most affected
   - Note the specific metrics that are problematic

2. **Try Multiple Methods**
   - Generate datasets with different techniques
   - Train models on each
   - Compare fairness metrics
   - Choose the best performer

3. **Monitor Trade-offs**
   - Fairness improvement may reduce accuracy slightly
   - Document the trade-offs
   - Ensure they're acceptable for your application

4. **Validate Properly**
   - Use separate test set (not used in mitigation)
   - Measure fairness on test set
   - Ensure improvements generalize

5. **Iterate**
   - If first method doesn't work, try another
   - Can combine methods (e.g., SMOTE + Reweighting)
   - Document what works for your dataset

## Common Pitfalls

❌ **Overfitting with Oversampling**
- Problem: Duplicating samples can cause overfitting
- Solution: Use cross-validation, early stopping

❌ **Losing Data with Undersampling**
- Problem: Removing 50% of data hurts performance
- Solution: Only use for huge datasets (>100k rows)

❌ **SMOTE on Categorical Data**
- Problem: Synthetic samples may not be realistic
- Solution: Use Oversampling instead, or encode carefully

❌ **Ignoring Intersectionality**
- Problem: Balancing overall classes doesn't fix subgroup bias
- Solution: Use Reweighting with multiple sensitive attributes

❌ **Not Validating**
- Problem: Improvements on training set don't generalize
- Solution: Always test on held-out data
