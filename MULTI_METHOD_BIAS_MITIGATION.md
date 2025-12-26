# Multi-Method Bias Mitigation Feature

## Overview
Enhanced Stage 6 to allow users to select and apply multiple bias mitigation methods simultaneously, with individual analysis for each method and a comprehensive comparison dashboard.

## What Changed

### 1. Method Selection (Stage 6 UI)
**Before:** Single method selection using `st.selectbox()`
```python
method = st.selectbox(
    "Mitigation Technique:",
    options=["Reweighting", "SMOTE", "Random Oversampling", "Random Undersampling"]
)
```

**After:** Multiple method selection using `st.multiselect()`
```python
selected_methods = st.multiselect(
    "Select one or more mitigation techniques:",
    options=["Reweighting", "SMOTE", "Random Oversampling", "Random Undersampling"],
    default=[]
)
```

### 2. Parameter Configuration
**Before:** Single configuration UI for one selected method

**After:** Individual configuration for each selected method using expanders
- Each method gets its own expandable configuration section
- Session state keys include method name to prevent conflicts
- Example: `f"smote_k_{method}"` instead of just `"smote_k"`

### 3. Application Logic
**Before:** Applied one method and stored single result dictionary

**After:** Iterates through all selected methods with progress tracking
- Progress bar shows current method being applied
- Each method applied independently
- Results stored in dictionary with method names as keys

### 4. Results Storage Structure
**Before:**
```python
results["stages"]["6_bias_mitigation"] = {
    "status": "success",
    "method": method,
    "method_params": method_params,
    "mitigation_result": mitigation_result,
    "comparison_result": comparison_result
}
```

**After:**
```python
results["stages"]["6_bias_mitigation"] = {
    "status": "success",
    "methods": {
        "SMOTE": {
            "status": "success",
            "method_params": {...},
            "mitigation_result": {...},
            "comparison_result": {...}
        },
        "Reweighting": {
            "status": "success",
            "method_params": {...},
            "mitigation_result": {...},
            "comparison_result": {...}
        }
    },
    "applied_methods": ["SMOTE", "Reweighting"]
}
```

### 5. Display Results (Stage 6)

#### A. Comparison Dashboard
**New Feature:** Displays all methods side-by-side in a comparison table

Shows for each method:
- Original Rows
- New Rows
- Rows Change (±)
- Original Imbalance Ratio
- New Imbalance Ratio
- Improvement (✓/✗)

Additionally:
- Automatic identification of best method (lowest imbalance ratio)
- Recommendation displayed above the table

#### B. Individual Method Results
**New Feature:** Expandable sections for each method's detailed results

Each method dropdown contains:
- Summary metrics (rows, changes, output file)
- Target distribution comparison table
- Imbalance improvement metrics
- Agent analysis
- Download button for generated CSV

**Backward Compatibility:** Single-method display still supported if old format is detected

## User Workflow

### Step 1: Select Methods
```
Stage 6: Bias Mitigation
Would you like to apply bias mitigation techniques?
[Yes] [No]

Select one or more mitigation techniques:
☑ SMOTE
☑ Reweighting
☐ Random Oversampling
☐ Random Undersampling
```

### Step 2: Configure Each Method
```
⚙️ SMOTE Parameters
   K Neighbors: [5]
   Sampling Strategy: [auto]

⚙️ Reweighting Parameters
   Select sensitive columns: [Age, Race, Sex]
```

### Step 3: Apply All Methods
```
[Cancel] [Apply All Methods]

Applying SMOTE... (1/2)
█████████████░░░░░ 50%
```

### Step 4: View Results

#### Comparison Dashboard
```
📊 Methods Comparison Dashboard
┌─────────────┬──────────┬──────────┬──────────┬──────────┬──────────┬────────────┐
│ Method      │ Original │ New Rows │ Rows     │ Original │ New      │ Improvement│
│             │ Rows     │          │ Change   │ Imbalance│ Imbalance│            │
├─────────────┼──────────┼──────────┼──────────┼──────────┼──────────┼────────────┤
│ SMOTE       │ 10,000   │ 15,000   │ +5,000   │ 3.00     │ 1.00     │ ✓          │
│ Reweighting │ 10,000   │ 10,000   │ 0        │ 3.00     │ 3.00     │ ✗          │
└─────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴────────────┘

💡 Best Method: SMOTE achieved the lowest imbalance ratio (1.00)
```

#### Individual Results
```
📂 Individual Method Results

▼ 📋 SMOTE - Detailed Results
  Summary
  [Original Rows: 10,000] [New Rows: 15,000] [Rows Added: +5,000] [Output File: ✓]
  
  Target Distribution Comparison
  [Table showing before/after distribution]
  
  Imbalance Improvement
  [Metrics with deltas]
  
  Agent Analysis
  [AI analysis of effectiveness]
  
  [Download SMOTE Dataset]

▶ 📋 Reweighting - Detailed Results
```

## Benefits

### 1. Time Efficiency
- Apply multiple methods in one batch instead of running Stage 6 repeatedly
- Compare results without switching between different runs

### 2. Better Decision Making
- Side-by-side comparison makes it easy to choose the best method
- Automatic recommendation highlights optimal choice
- See trade-offs clearly (e.g., data size vs. balance improvement)

### 3. Comprehensive Analysis
- Each method gets full individual analysis
- No information loss compared to single-method approach
- Agent provides insights for each technique

### 4. Flexibility
- Can still select just one method if desired
- Can select all methods to see full spectrum
- Can exclude methods known to be unsuitable

## Technical Details

### Session State Management
- Each method's parameters stored with unique keys
- Session state cleared after successful application
- Progress tracking during multi-method application

### Error Handling
- Individual method failures don't stop other methods
- Failed methods marked with "error" status
- Error messages displayed in individual dropdowns

### Download Functionality
- Each method generates separate CSV file
- Download buttons use unique keys: `f"download_{method.replace(' ', '_')}"`
- Files saved to `reports/.../generated_csv/` with method name

### Performance
- Methods applied sequentially (not parallel) to avoid resource conflicts
- Progress bar provides real-time feedback
- Streamlit rerun after completion to display results

## Examples

### Example 1: Comparing Oversampling Methods
Select: SMOTE, Random Oversampling

**Expected Result:**
- SMOTE creates synthetic samples (higher quality)
- Random Oversampling duplicates existing samples (simpler)
- Dashboard shows both achieve similar balance
- SMOTE may have slightly better diversity metrics

### Example 2: Reweighting vs Undersampling
Select: Reweighting, Random Undersampling

**Expected Result:**
- Reweighting preserves all data (10,000 rows)
- Undersampling reduces data (maybe 3,000 rows)
- Both may achieve similar imbalance ratios
- Trade-off: data quantity vs. true balance

### Example 3: All Methods
Select: All four methods

**Expected Result:**
- Complete overview of all mitigation approaches
- Clear ranking from best to worst
- Understanding of which methods work for this specific dataset
- Can choose based on constraints (e.g., can't afford to lose data → avoid undersampling)

## Future Enhancements

### Possible Additions:
1. **Ensemble Methods:** Combine results from multiple methods
2. **Export Comparison:** Download comparison table as CSV/PDF
3. **Visualization:** Chart showing imbalance ratios before/after for each method
4. **Method Chaining:** Apply methods sequentially (e.g., SMOTE then Reweighting)
5. **Custom Metrics:** Allow users to define what "best" means (speed, balance, data retention, etc.)

## Testing

### Test Cases:
1. ✓ Select single method → Should work (backward compatible)
2. ✓ Select multiple methods → Should apply all and show comparison
3. ✓ One method fails → Should show error in that dropdown, others still succeed
4. ✓ Cancel during configuration → Should return to initial prompt
5. ✓ Download each method's CSV → Should download correct file with unique name
6. ✓ Reweighting without sensitive columns → Button should be disabled
7. ✓ Skip mitigation → Should skip entire stage

### Manual Testing Checklist:
- [ ] Select 2+ methods, configure each, apply successfully
- [ ] Verify comparison dashboard shows correct metrics
- [ ] Expand each method dropdown, verify individual results
- [ ] Download each CSV, verify contents match method
- [ ] Check "Best Method" recommendation is accurate
- [ ] Test with dataset that has clear imbalance
- [ ] Test with already-balanced dataset
- [ ] Verify progress bar shows during application
- [ ] Check error handling if method fails

## Files Modified

### `src/gui_app.py`
- **Lines ~690-900:** Stage 6 method selection and configuration
  - Changed from `st.selectbox()` to `st.multiselect()`
  - Added per-method expanders for configuration
  - Updated session state keys to include method names
  - Added progress tracking during application
  - Updated results storage structure

- **Lines ~1000-1350:** Stage 6 results display
  - Added comparison dashboard with metrics table
  - Added best method recommendation
  - Added individual method dropdowns
  - Maintained backward compatibility for single-method format
  - Added unique download button keys

## Conclusion

The multi-method bias mitigation feature transforms Stage 6 from a trial-and-error process into a comprehensive analysis tool. Users can now efficiently compare different mitigation strategies and make informed decisions based on their specific needs and constraints.

**Key Achievement:** What previously required 4 separate runs through Stage 6 now happens in a single, streamlined workflow with comparative analysis.
