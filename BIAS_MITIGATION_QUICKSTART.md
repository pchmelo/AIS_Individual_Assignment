# Quick Start: Bias Mitigation Feature

## Installation

1. **Install new requirements:**
```bash
pip install -r requirements-gui.txt
```

This will install:
- `imbalanced-learn>=0.11.0` (for SMOTE, oversampling, undersampling)
- `scikit-learn>=1.3.0` (required by imbalanced-learn)

## Running the GUI

```bash
streamlit run src/gui_app.py
```

## Using Bias Mitigation

### Step-by-Step

1. **Complete the initial analysis** (Stages 0-5):
   - Load your dataset
   - Specify target column
   - Review fairness analysis results

2. **Stage 6 will automatically appear** after Recommendations

3. **Choose whether to apply mitigation:**
   - Click **"Yes, apply mitigation"** to proceed
   - Click **"No, skip this step"** to finish without mitigation

4. **If Yes, select your method:**
   - **Reweighting**: Assigns weights (no new rows)
   - **SMOTE**: Generates synthetic samples
   - **Random Oversampling**: Duplicates minority samples
   - **Random Undersampling**: Removes majority samples

5. **Configure parameters** (method-specific)

6. **Click "Apply Mitigation"**

7. **Review the results:**
   - Check the comparison table
   - Read the agent analysis
   - Download the generated CSV if satisfied

### Example: Fixing Imbalanced Adult Dataset

```
Original Distribution:
  <=50K: 22,654 (75%)
  >50K:   7,508 (25%)
  
After SMOTE:
  <=50K: 22,654 (50%)
  >50K:  22,654 (50%)
  
Improvement: Imbalance ratio from 3.0 to 1.0
```

## Output Location

Generated datasets are saved in:
```
reports/<your_dataset>_<timestamp>/generated_csv/
```

## Tips

- **Start Simple**: Try Reweighting first (safest, no data changes)
- **Read Agent Analysis**: It tells you if the mitigation helped
- **Try Multiple Methods**: Compare which works best for your case
- **Keep Original**: Never delete your original dataset

## Troubleshooting

### "Sensitive columns required for reweighting"
- **Solution**: Select at least one sensitive column from the dropdown

### "SMOTE failed with k_neighbors error"
- **Solution**: Your minority class has fewer samples than k_neighbors. Reduce k_neighbors value.

### "Cannot apply SMOTE to categorical data"
- **Solution**: The tool auto-encodes categorical columns, but results may vary. Consider Oversampling instead.

## Method Comparison

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Reweighting** | No data change, preserves all info | Requires model support | Model training |
| **SMOTE** | Creates realistic synthetic samples | Complex, may not work well with categorical | Continuous features |
| **Oversampling** | Simple, increases minority | Risk of overfitting | Quick balance |
| **Undersampling** | Fast, reduces majority | Loses information | Large datasets |

## Next Steps

After generating mitigated datasets:

1. **Train models** with the new data
2. **Compare fairness metrics** between models trained on original vs mitigated data
3. **Validate results** on a separate test set
4. **Document** which mitigation technique worked best for your use case
