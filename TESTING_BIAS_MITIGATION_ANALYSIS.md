# Testing Guide for Bias Mitigation Analysis Integration

## Quick Test Steps

### Option 1: Test with New Bias Mitigation Run

1. **Start the GUI**:
   ```bash
   cd "d:\Vasco\UN\mestrado\1 ano\1 semestre\Inteligencia Artificial e Sociedade\projeto\individual_assignment"
   streamlit run src/gui_app.py
   ```

2. **Run a new evaluation with bias mitigation**:
   - Start a new evaluation on any dataset (e.g., adult-all.csv)
   - Complete stages 0-5
   - When you reach Stage 6, select "Yes, apply mitigation"
   - Choose multiple methods (e.g., SMOTE, Reweighting, Random Oversampling)
   - Click "Apply All Methods"

3. **Check the report**:
   - Open `reports/<dataset>_<timestamp>/evaluation_report.txt`
   - Look for the `6_BIAS_MITIGATION` section
   - Verify it contains:
     - Status and Applied Methods list
     - For each method: [METHOD_NAME], [MITIGATION RESULTS], [COMPARISON RESULTS], [AGENT ANALYSIS]

4. **Check the display**:
   - In the GUI, expand each method in "Individual Method Details"
   - Verify you see the "Agent Analysis" section with the full text

### Option 2: Test with Previous Results

1. **Start the GUI**:
   ```bash
   streamlit run src/gui_app.py
   ```

2. **View previous results**:
   - Click "View Previous Results" in the sidebar
   - Select a report from the dropdown (e.g., `adult-all.csv_20251227_151854`)
   - Navigate to "Tab 4: Bias Mitigation"

3. **Verify the display**:
   - Check if the Methods Comparison table shows
   - Expand each method's details
   - Look for the "Agent Analysis" section
   - If the report is old (no analysis), it should gracefully show nothing
   - If the report is new (has analysis), it should show the full agent analysis text

## Expected Behavior

### When Bias Mitigation is Applied

**Console Output:**
```
Applying smote bias mitigation...
Comparing original and mitigated datasets...
[Agent analysis generation]
Report saved: .../evaluation_report.txt
```

**Report File (`evaluation_report.txt`):**
```
6_BIAS_MITIGATION
--------------------------------------------------------------------------------

Status: success
Applied Methods: SMOTE, Reweighting

[SMOTE]
----------------------------------------

[MITIGATION RESULTS]
{
  "status": "success",
  "original_rows": 45222,
  ...
}

[COMPARISON RESULTS]
{
  "imbalance_metrics": {
    "original_imbalance_ratio": 3.15,
    ...
  },
  ...
}

[AGENT ANALYSIS]
Based on the comparison between the original and mitigated datasets:

1. **Was the bias mitigation effective?** Yes
   ...
```

**GUI Display (Current Session):**
- Shows comparison dashboard with all methods
- Expandable sections for each method
- Each section shows: Dataset info → Sample data → Agent Analysis → Download

**GUI Display (Previous Results):**
- Parses the report file
- Shows the same comparison dashboard
- Expandable sections for each method
- Each section shows: Dataset info → Sample data → **Agent Analysis (NEW!)** → Download

### When No Bias Mitigation

**Report File:**
```
6_BIAS_MITIGATION
--------------------------------------------------------------------------------


================================================================================
END OF REPORT
```

**GUI Display:**
- Shows "No bias mitigation was applied in this evaluation."

## Verification Checklist

- [ ] Report file contains `6_BIAS_MITIGATION` section header
- [ ] Each applied method has its own subsection with [METHOD_NAME] header
- [ ] Each method subsection contains [AGENT ANALYSIS] with text
- [ ] GUI Tab 4 shows Methods Comparison table
- [ ] GUI Tab 4 expandable sections show "Agent Analysis" subsection
- [ ] Analysis text matches what's in the report file
- [ ] Download buttons work for each method
- [ ] Old reports (without analysis) don't break the display
- [ ] New reports (with analysis) display correctly

## Common Issues and Solutions

### Issue: Agent Analysis not showing in GUI
**Check:**
1. Is the report file formatted correctly?
2. Does the method name in the report match the CSV filename?
3. Is the regex pattern matching? (Should match `[METHOD_NAME]\n----...`)

**Solution:**
- The code handles variations like "SMOTE" vs "Smote" vs "RANDOM OVERSAMPLING"
- Check console for any parsing errors

### Issue: Empty bias mitigation section
**Check:**
1. Was bias mitigation actually applied?
2. Check if `6_bias_mitigation` is in session state
3. Check if methods were successfully run

**Solution:**
- Verify stage 6 was completed successfully
- Check for error messages during method application

## Debug Mode

To see detailed parsing information, you can temporarily add debug prints to the GUI code:

```python
# In gui_app.py, tab4 section after parsing
st.write("DEBUG: methods_analysis keys:", list(methods_analysis.keys()))
for method, analysis in methods_analysis.items():
    st.write(f"DEBUG: {method} analysis length:", len(analysis))
```

## Performance Notes

- Parsing the report file is fast (< 100ms for typical reports)
- Regex matching is efficient even with large reports
- No impact on report generation time
- GUI responsiveness not affected

## Next Steps After Testing

If everything works:
1. Remove any debug prints
2. Test with multiple datasets
3. Test with all 4 mitigation methods
4. Verify with different parameter configurations
5. Test error handling (e.g., corrupted report files)

If issues found:
1. Check the regex pattern in gui_app.py
2. Verify the report format in pipeline.py
3. Add more error handling if needed
4. Check method name mapping logic
