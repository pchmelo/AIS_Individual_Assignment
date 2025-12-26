# Bias Mitigation Feature - Setup & Testing Checklist

## ✅ Pre-Implementation Checklist

All of these have been completed:

- [x] Created `bias_mitigation_tools.py` with 4 mitigation methods
- [x] Updated `pipeline.py` with bias mitigation methods
- [x] Updated `gui_app.py` with Stage 6 UI
- [x] Updated `requirements-gui.txt` with new dependencies
- [x] Updated `readme.md` with feature documentation
- [x] Created comprehensive documentation (5 files)
- [x] Added comparison and analysis functionality
- [x] Integrated AI agent for result review

## 📋 Setup Checklist (User Actions Required)

Follow these steps to set up the feature:

### 1. Install Dependencies

```bash
# Navigate to project root
cd "d:\Vasco\UN\mestrado\1 ano\1 semestre\Inteligencia Artificial e Sociedade\projeto\individual_assignment"

# Install/upgrade requirements
pip install -r requirements-gui.txt
```

**Verify installation:**
```bash
python -c "import imblearn; print('imbalanced-learn:', imblearn.__version__)"
python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"
```

Expected output:
```
imbalanced-learn: 0.11.0 (or higher)
scikit-learn: 1.3.0 (or higher)
```

- [ ] Dependencies installed successfully
- [ ] Versions verified

### 2. Test the GUI

```bash
# Run the GUI
streamlit run src/gui_app.py
```

**Verify:**
- [ ] GUI opens in browser
- [ ] No import errors in terminal
- [ ] Home page loads correctly

### 3. Test Basic Functionality

**Create a new evaluation:**
1. Click "New Evaluation"
2. Upload dataset: `src/data/adult.csv` or `adult-all.csv`
3. Configure:
   - Dataset name: `adult-all.csv`
   - Target column: `income`
   - Model: Choose any (recommend Google Gemini for testing)

**Complete stages 0-5:**
- [ ] Stage 0: Dataset Loading ✓
- [ ] Stage 1: Objective Inspection ✓
- [ ] Stage 2: Data Quality Analysis ✓
- [ ] Stage 3: Sensitive Attribute Detection ✓
- [ ] Stage 4: Imbalance Analysis ✓
- [ ] Stage 4.5: Target Fairness Analysis ✓
- [ ] Stage 5: Recommendations ✓

### 4. Test Stage 6 (Bias Mitigation)

**Prompt appears:**
- [ ] "Apply bias mitigation?" prompt shows
- [ ] "Yes" and "No" buttons visible

**Click "Yes, apply mitigation":**
- [ ] Method selection dropdown appears
- [ ] Shows 4 methods: Reweighting, SMOTE, Random Oversampling, Random Undersampling
- [ ] Help text displays for each method

**Select "SMOTE":**
- [ ] Parameter fields appear (K Neighbors, Sampling Strategy)
- [ ] Default values shown (5, 'auto')
- [ ] "Apply Mitigation" button visible

**Click "Apply Mitigation":**
- [ ] Spinner shows "Applying SMOTE..."
- [ ] No errors occur
- [ ] Results display appears

**Verify results display:**
- [ ] Success message: "✅ SMOTE applied successfully!"
- [ ] Summary metrics show (Original Rows, New Rows, Rows Added)
- [ ] Target Distribution Comparison table visible
- [ ] Imbalance Improvement metrics show
- [ ] Agent Analysis section visible
- [ ] Download button appears

**Download and verify file:**
- [ ] Click "📥 Download Mitigated Dataset"
- [ ] CSV file downloads
- [ ] Open CSV and verify:
  - File opens correctly
  - Row count matches "New Rows" metric
  - Columns are present and data looks valid

**Verify file location:**
```bash
# Check that file was saved correctly
ls -la "reports/adult-all.csv_*/generated_csv/"
```
- [ ] File exists in `generated_csv/` folder
- [ ] Filename format: `adult-all_smote.csv`

### 5. Test Other Methods

**Reweighting:**
1. Start new evaluation or go back to Stage 6
2. Click "Yes, apply mitigation"
3. Select "Reweighting"
4. Select sensitive columns (should show dropdown)
5. Apply
6. Verify:
   - [ ] `sample_weight` column added to CSV
   - [ ] Row count unchanged
   - [ ] Weight statistics shown

**Random Oversampling:**
1. Start new evaluation or go back to Stage 6
2. Click "Yes, apply mitigation"
3. Select "Random Oversampling"
4. Select sampling strategy
5. Apply
6. Verify:
   - [ ] Row count increased
   - [ ] Distribution balanced
   - [ ] No synthetic data warning

**Random Undersampling:**
1. Start new evaluation or go back to Stage 6
2. Click "Yes, apply mitigation"
3. Select "Random Undersampling"
4. Note the warning: "⚠️ This method reduces dataset size"
5. Apply
6. Verify:
   - [ ] Row count decreased
   - [ ] Distribution balanced
   - [ ] Warning was visible

### 6. Test "No" Path

**Start new evaluation:**
1. Complete stages 0-5
2. At Stage 6 prompt, click "No, skip this step"
3. Verify:
   - [ ] Stage 6 marked as skipped
   - [ ] Info message shows: "Bias mitigation was skipped by user."
   - [ ] Evaluation completes successfully
   - [ ] Reports generate without mitigation section

### 7. Test Agent Analysis

**Review agent analysis for SMOTE:**
- [ ] Analysis includes "Effectiveness: YES/NO"
- [ ] Lists specific improvements with numbers
- [ ] Identifies remaining issues (if any)
- [ ] Provides actionable recommendations

**Quality checks:**
- [ ] Analysis is relevant to the dataset
- [ ] Numbers match the comparison table
- [ ] Recommendations are actionable
- [ ] No generic or placeholder text

### 8. Test Error Handling

**SMOTE with small k_neighbors:**
1. Select SMOTE
2. Set k_neighbors to 10 (might be too high for minority class)
3. Apply
4. If error occurs:
   - [ ] Error message is clear
   - [ ] Suggests reducing k_neighbors
   - [ ] GUI doesn't crash

**Reweighting without sensitive columns:**
1. Select Reweighting
2. Don't select any sensitive columns (if possible)
3. Try to apply
4. Verify:
   - [ ] Button is disabled OR
   - [ ] Error message shows: "Sensitive columns required"

### 9. Test Multiple Generations

**Generate multiple datasets:**
1. Complete evaluation with SMOTE
2. Go back to Stage 6 (if possible, or start new evaluation)
3. Apply Reweighting
4. Verify:
   - [ ] Both files exist in `generated_csv/`
   - [ ] Files have different names (`_smote.csv`, `_reweighted.csv`)
   - [ ] Both are downloadable

### 10. Verify Documentation

**Check documentation files exist:**
- [ ] `BIAS_MITIGATION_GUIDE.md` exists and is readable
- [ ] `BIAS_MITIGATION_QUICKSTART.md` exists and is readable
- [ ] `BIAS_MITIGATION_WORKFLOW.md` exists and is readable
- [ ] `BIAS_MITIGATION_EXAMPLES.md` exists and is readable
- [ ] `IMPLEMENTATION_SUMMARY.md` exists and is readable

**Verify links in readme.md:**
- [ ] Open `readme.md`
- [ ] Check documentation links work
- [ ] Feature is mentioned in features list
- [ ] Pipeline stages updated

## 🐛 Known Issues & Solutions

### Issue: Import error for `imblearn`
**Solution:**
```bash
pip install imbalanced-learn
```

### Issue: SMOTE fails with error about neighbors
**Solution:**
- Minority class has fewer samples than k_neighbors
- Reduce k_neighbors value (try 3 or 2)
- Or use Random Oversampling instead

### Issue: GUI freezes during mitigation
**Solution:**
- Large datasets take time
- Wait for spinner to complete
- Check terminal for progress messages

### Issue: Download button doesn't work
**Solution:**
- File might not have been generated
- Check terminal for errors
- Verify file exists in `reports/.../generated_csv/`

## ✅ Final Verification

After completing all tests above:

- [ ] All stages complete without errors
- [ ] All 4 mitigation methods work
- [ ] Generated files are valid CSVs
- [ ] Agent analysis is meaningful
- [ ] Download functionality works
- [ ] Error handling is appropriate
- [ ] Documentation is accessible

## 📊 Test Results Template

Use this to document your test results:

```
Test Date: _______________
Tester: _______________

Installation:
✓ Dependencies installed
✓ No errors during import

Basic Functionality:
✓ Stages 0-5 complete
✓ Stage 6 prompt appears
✓ Method selection works

SMOTE Test:
✓ Parameters configurable
✓ Mitigation applies successfully
✓ Results display correctly
✓ File generated: reports/.../generated_csv/adult-all_smote.csv
✓ Agent analysis present
✓ Download works

Other Methods:
✓ Reweighting: [Pass/Fail]
✓ Oversampling: [Pass/Fail]
✓ Undersampling: [Pass/Fail]

Skip Test:
✓ "No" button works
✓ Skipped message shows

Error Handling:
✓ Errors handled gracefully
✓ No crashes

Overall Status: [PASS/FAIL]
Notes: _______________________________
```

## 🎉 Success Criteria

The feature is ready for production when:

✅ All checklist items are marked complete  
✅ No critical errors occur during testing  
✅ Agent analysis is meaningful and accurate  
✅ Generated files are valid and downloadable  
✅ Documentation is clear and accessible  
✅ Users can complete full workflow without confusion  

---

**Good luck with testing! The feature is ready to use. 🚀**
