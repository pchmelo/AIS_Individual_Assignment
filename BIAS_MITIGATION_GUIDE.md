# Bias Mitigation Feature Guide

## Overview

A new **Bias Mitigation** stage has been added to the GUI workflow (Stage 6), which appears after the Recommendations stage (Stage 5). This feature allows users to apply various bias mitigation techniques to generate balanced datasets.

## What Was Added

### 1. New Tools (`bias_mitigation_tools.py`)

Four bias mitigation techniques are now available:

#### **Reweighting**
- **What it does**: Assigns sample weights to each training example to balance underrepresented groups
- **When to use**: When you want to keep all original data but give more importance to minority groups during model training
- **Parameters**:
  - `sensitive_columns`: Which sensitive attributes to consider (e.g., race, sex)
- **Output**: Original dataset + a new `sample_weight` column
- **Note**: No new rows are added; weights should be used during model training

#### **SMOTE (Synthetic Minority Over-sampling Technique)**
- **What it does**: Generates synthetic samples for minority classes using k-nearest neighbors
- **When to use**: When you need more samples for minority classes and synthetic data is acceptable
- **Parameters**:
  - `k_neighbors`: Number of nearest neighbors to use (default: 5)
  - `sampling_strategy`: Which classes to oversample ('auto', 'minority', 'not majority', 'all')
- **Output**: New dataset with synthetic samples added
- **Note**: Dataset size increases

#### **Random Oversampling**
- **What it does**: Randomly duplicates samples from minority classes
- **When to use**: When you want a simple approach to increase minority class representation
- **Parameters**:
  - `sampling_strategy`: Which classes to oversample ('auto', 'minority', 'not majority', 'all')
- **Output**: New dataset with duplicated samples
- **Note**: Dataset size increases; may lead to overfitting

#### **Random Undersampling**
- **What it does**: Randomly removes samples from majority classes
- **When to use**: When you want to reduce dataset size and balance classes
- **Parameters**:
  - `sampling_strategy`: Which classes to undersample ('auto', 'not minority', 'majority', 'all')
- **Output**: New dataset with fewer samples
- **⚠️ Warning**: Dataset size decreases; you lose information

### 2. Workflow Integration

After completing Stage 5 (Recommendations), the GUI will:

1. **Ask the user** if they want to apply bias mitigation
   - **Yes**: Proceed to method selection
   - **No**: Skip this stage and complete the evaluation

2. **Method Selection Interface** (if user chose "Yes"):
   - Dropdown to select mitigation technique
   - Method-specific parameters
   - Real-time help text explaining each method
   - Visual warnings for methods that reduce dataset size

3. **Apply Mitigation**:
   - Generate the new dataset
   - Save it to `reports/<dataset_timestamp>/generated_csv/`
   - Compare original vs mitigated dataset

4. **Results Display**:
   - Summary metrics (rows before/after, changes)
   - Target distribution comparison table
   - Imbalance improvement metrics
   - **Agent Analysis**: AI-generated review answering:
     - Was the mitigation effective?
     - What improved? (specific numbers)
     - What remained problematic?
     - Recommendations for further improvements
   - Download button for the generated CSV

### 3. File Structure

Generated datasets are saved in:
```
reports/
  └── <dataset_name>_<timestamp>/
      └── generated_csv/
          ├── <dataset>_reweighted.csv
          ├── <dataset>_smote.csv
          ├── <dataset>_oversampled.csv
          └── <dataset>_undersampled.csv
```

## Usage Example

### Scenario: Adult Census Income Dataset

1. Complete stages 0-5 normally
2. At Stage 6, click **"Yes, apply mitigation"**
3. Select **"SMOTE"**
4. Set parameters:
   - k_neighbors: 5
   - sampling_strategy: 'auto'
5. Click **"Apply Mitigation"**
6. Review results:
   - Original: 30,162 rows
   - After SMOTE: 45,222 rows (+15,060)
   - Target distribution changed from 75%/25% to 50%/50%
   - Agent Analysis confirms improvement in balance
7. Download `adult-all_smote.csv` for model training

## Agent Review Format

The agent provides a comprehensive review including:

```
1. Effectiveness: Yes/No with explanation
2. Improvements:
   - Original imbalance ratio: 3.0
   - Mitigated imbalance ratio: 1.0
   - Improvement: 66.7%
3. Remaining Issues:
   - Sensitive attribute X still shows 2:1 ratio
   - Consider applying reweighting on top
4. Recommendations:
   - Try combining SMOTE + Reweighting
   - Apply to specific sensitive groups
```

## Technical Details

### Requirements
Added to `requirements-gui.txt`:
```
imbalanced-learn>=0.11.0
scikit-learn>=1.3.0
```

### Pipeline Integration
- New `BiasMitigationTools` class in `bias_mitigation_tools.py`
- New `bias_mitigation_agent` in pipeline
- Methods: `apply_bias_mitigation()` and `compare_mitigation_results()`

### GUI Updates
- Stage 6 added to stages list
- Special handling in `display_evaluation_workflow()`
- Custom display in `display_stage_results()`
- Session state management for user choices

## Best Practices

1. **Start with Analysis**: Always review Stage 4.5 (Target Fairness Analysis) before applying mitigation
2. **Try Multiple Methods**: Different techniques work better for different scenarios
3. **Compare Results**: The agent provides objective metrics - use them to choose the best approach
4. **Combine Techniques**: You can generate multiple versions and compare them
5. **Consider Your Goal**:
   - For model training: Reweighting (no data loss)
   - For data augmentation: SMOTE or Oversampling
   - For quick balance: Undersampling (loses data)

## Limitations

- SMOTE works best with continuous features; categorical features are label-encoded
- Undersampling may lose important information
- Synthetic samples (SMOTE) may not represent real-world distributions
- Reweighting requires sensitive attribute information
- Generated datasets should be validated before production use

## Future Enhancements

Possible additions for future versions:
- ADASYN (Adaptive Synthetic Sampling)
- Combination of techniques (e.g., SMOTE + Tomek Links)
- More sophisticated fairness-aware algorithms (e.g., Fair-SMOTE)
- Automated method selection based on dataset characteristics
- Fairness metrics before/after for multiple sensitive attributes
