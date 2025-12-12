# Quick Start Guide - GUI Mode

## Installation

### Step 1: Install GUI Dependencies
```bash
pip install -r requirements-gui.txt
```

This installs:
- streamlit (web framework)
- pandas (data processing)
- numpy (numerical operations)
- matplotlib (plotting)
- seaborn (statistical visualizations)

### Step 2: Verify Installation
```bash
streamlit --version
```

Should output something like: `Streamlit, version 1.28.0`

## Running the GUI

### Option 1: Using main.py (Recommended)

1. Open `src/main.py`
2. Change the RUN_MODE variable:
```python
RUN_MODE = "gui"  # Changed from "terminal"
```
3. Run:
```bash
python src/main.py
```

### Option 2: Using launch.bat (Windows)

Double-click `launch.bat` and select option 2.

### Option 3: Direct Streamlit Command

```bash
streamlit run src/gui_app.py
```

## First-Time Setup

### Configure API Keys (Optional)

If using Grok or Gemini models, set environment variables:

**Windows PowerShell:**
```powershell
$env:OPENROUTER_API_KEY = "your-key-here"
$env:GOOGLE_API_KEY = "your-key-here"
```

**Windows Command Prompt:**
```cmd
set OPENROUTER_API_KEY=your-key-here
set GOOGLE_API_KEY=your-key-here
```

**Linux/Mac:**
```bash
export OPENROUTER_API_KEY="your-key-here"
export GOOGLE_API_KEY="your-key-here"
```

### Verify Data Directory

Ensure you have datasets in `src/data/`:
```
src/data/
├── adult-all.csv
├── adult.csv
└── (your other datasets)
```

## Using the GUI

### Creating Your First Evaluation

1. **Launch the GUI** (browser opens automatically at http://localhost:8501)

2. **Main Page** appears with two options:
   - Click "🆕 New Evaluation"

3. **Configure Evaluation** (in sidebar):
   - Select a dataset from dropdown (or upload new one)
   - Choose a model:
     - IBM Granite (runs locally, no API key needed)
     - Grok (requires API key)
     - Google Gemini (requires API key)
   - (Optional) Check "Specify target column" and select column

4. **Start Evaluation**:
   - Click "▶️ Start Evaluation" button
   - Wait for pipeline to complete (10-60 seconds typically)

5. **Review Results**:
   - Each stage appears with a dropdown
   - Click dropdown to see details
   - Click "▶️ Continue" to proceed to next stage
   - Stop at any stage if issues are found

6. **View Visualizations** (Stage 4.5):
   - See scale-grouped charts automatically
   - Select specific combinations from dropdown to view
   - All images also saved to disk

### Viewing Previous Results

1. From main page, click "📂 View Previous Results"

2. Select a report from dropdown

3. Navigate three tabs:
   - **Full Report**: Complete text report
   - **Agent Summary**: Condensed findings
   - **Visualizations**: All images organized by folder

4. Expand folders to see images

5. Click "⬅️ Back to Main" when done

## Testing the GUI

### Test Checklist

1. ✅ **GUI Launches Successfully**
   - Run `streamlit run src/gui_app.py`
   - Browser opens to main page
   - No error messages in terminal

2. ✅ **Dataset Selection Works**
   - See list of datasets in sidebar
   - Can select different datasets
   - Upload new dataset (test with small CSV)

3. ✅ **Model Selection Works**
   - Can toggle between three model options
   - Selection persists during session

4. ✅ **Target Selection Works**
   - Checkbox enables/disables target selection
   - Dropdown populated with dataset columns
   - Can select different targets

5. ✅ **Pipeline Execution Works**
   - Click "Start Evaluation"
   - See initialization message
   - Results appear after completion
   - No crashes or errors

6. ✅ **Stage Display Works**
   - All stages appear in order
   - Can expand/collapse dropdowns
   - Tool results display correctly
   - Agent analysis shows

7. ✅ **Continue Buttons Work**
   - Button appears after each stage (except last)
   - Clicking reveals next stage
   - Next stage hidden until clicked

8. ✅ **Stage 4.5 Visualizations Work**
   - Scale charts display (high/medium/low)
   - Individual combinations dropdown appears
   - Can select combinations to view
   - Selected images display

9. ✅ **Previous Results Work**
   - Can select past reports
   - Full report tab shows text
   - Agent summary tab shows text
   - Visualizations tab shows images
   - Images organized by folder

10. ✅ **Navigation Works**
    - Back buttons return to main page
    - Can switch between new/view modes
    - Session state preserved

### Sample Test Session

**Test Case: Complete Evaluation Flow**

1. Launch GUI: `streamlit run src/gui_app.py`
2. Click "🆕 New Evaluation"
3. Select dataset: "adult-all.csv"
4. Select model: "Google Gemini"
5. Check "Specify target column"
6. Select target: "Income"
7. Click "▶️ Start Evaluation"
8. Wait for completion (~30 seconds)
9. Verify Stage 0 appears
10. Click "▶️ Continue"
11. Verify Stage 1 appears
12. Click "▶️ Continue"
13. Verify Stage 2 appears with metrics table
14. Continue through all stages
15. Verify Stage 4.5 shows visualizations
16. Select 2-3 combinations to view
17. Verify individual charts display
18. Complete evaluation
19. Click "⬅️ Back to Main"
20. Click "📂 View Previous Results"
21. Select the just-created report
22. Verify all three tabs work
23. Check images are organized properly
24. Return to main page

Expected: No errors, all features work, report saved correctly.

## Troubleshooting

### GUI Won't Start

**Problem**: "No module named 'streamlit'"
**Solution**: 
```bash
pip install streamlit
```

**Problem**: Port 8501 already in use
**Solution**: Kill existing process or use different port:
```bash
streamlit run src/gui_app.py --server.port 8502
```

### Dataset Issues

**Problem**: "No datasets found"
**Solution**: 
- Check `src/data/` folder exists
- Ensure CSV files are present
- Try uploading a dataset through GUI

**Problem**: "Could not read dataset columns"
**Solution**:
- Verify CSV is properly formatted
- Check for encoding issues (should be UTF-8)
- Ensure file is not corrupted

### Model Issues

**Problem**: API key errors
**Solution**:
- Verify environment variables are set
- Try IBM Granite (no API key needed)
- Check API key validity

**Problem**: "Model not responding"
**Solution**:
- Check internet connection
- Verify API quotas
- Try different model
- Check terminal for detailed errors

### Visualization Issues

**Problem**: Images not displaying
**Solution**:
- Check report folder exists
- Verify images were generated
- Refresh browser page
- Check terminal for image generation errors

**Problem**: Too many images, browser slow
**Solution**:
- Only select few combinations at a time
- Use "View Previous Results" for offline viewing
- Check images directly in report folder

### Performance Issues

**Problem**: GUI is slow
**Solution**:
- Use smaller datasets for testing
- Close other browser tabs
- Don't expand all sections at once
- Limit selected combinations

**Problem**: Pipeline takes too long
**Solution**:
- Normal for large datasets
- Check terminal for progress
- IBM Granite is slower (local processing)
- Consider using faster API models

## Tips and Best Practices

### 1. Start Small
- Test with small datasets first
- Verify everything works before large evaluations

### 2. Use Continue Buttons
- Review each stage before proceeding
- Stop if early stages show critical issues
- Saves time by not running full pipeline

### 3. Organize Reports
- Reports auto-timestamped
- Clean old reports periodically
- Keep important reports, delete test runs

### 4. Manage Visualizations
- Don't view all combinations at once
- Select specific ones of interest
- Access full set from report folder

### 5. Dataset Management
- Upload datasets with descriptive names
- Keep datasets small when possible
- Remove unused datasets from data folder

### 6. Model Selection
- IBM Granite: Best for privacy, slower
- Grok: Fast, requires API key
- Gemini: Advanced reasoning, requires API key

### 7. Browser Performance
- Use Chrome/Edge for best performance
- Close unused tabs
- Refresh page if it becomes sluggish

### 8. Error Recovery
- Check terminal output for details
- Most errors show in GUI too
- Can restart evaluation if needed

## Advanced Usage

### Running Multiple Evaluations

You can't run multiple evaluations in same session (Streamlit limitation).
To run multiple:
1. Complete first evaluation
2. Click "Back to Main"
3. Start new evaluation
4. Or open new browser tab to same URL

### Custom Port

If default port (8501) conflicts:
```bash
streamlit run src/gui_app.py --server.port 8502
```

### Headless Mode

For remote servers without browser:
```bash
streamlit run src/gui_app.py --server.headless true --server.port 8501
```
Then access via: `http://your-server-ip:8501`

### Custom Theme

Edit `~/.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

## Next Steps

Once comfortable with GUI:
1. Try different datasets
2. Experiment with target columns
3. Compare different models
4. Explore fairness visualizations
5. Share reports with team
6. Use insights to improve datasets

## Getting Help

If issues persist:
1. Check this guide
2. Review GUI_GUIDE.md for detailed docs
3. Check terminal output for errors
4. Verify all dependencies installed
5. Try terminal mode as alternative

## Switching Back to Terminal Mode

To return to terminal/CLI mode:
1. Open `src/main.py`
2. Change:
```python
RUN_MODE = "terminal"  # Changed from "gui"
```
3. Run: `python src/main.py`

Both modes generate identical reports, just different interfaces!
