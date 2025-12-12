# GUI Mode User Guide

## Overview
The Dataset Fairness Evaluation System now supports both **Terminal Mode** and **GUI Mode**.

## Quick Start

### Switch Between Modes
Edit `src/main.py` and change the `RUN_MODE` variable:

```python
RUN_MODE = "terminal"  # For command-line interface
RUN_MODE = "gui"       # For web-based graphical interface
```

### Install GUI Dependencies
```bash
pip install -r requirements-gui.txt
```

### Run the Application
```bash
python src/main.py
```

## GUI Mode Features

### 1. **Main Page**
   - Choose between starting a new evaluation or viewing previous results
   - Clean, intuitive interface with two main options

### 2. **New Evaluation**
   **Sidebar Configuration:**
   - **Dataset Selection**: Choose from existing datasets or upload a new CSV file
   - **Model Selection**: Choose between 3 AI models:
     - 🏠 IBM Granite (Local)
     - 🌐 Grok (API)
     - 🔮 Google Gemini (API)
   - **Target Column**: Optionally specify a target column for Stage 4.5 fairness analysis

   **Evaluation Process:**
   - Click "Start Evaluation" to run the pipeline
   - Results appear step-by-step
   - Each stage has a collapsible dropdown to view details
   - **Continue button** after each step allows you to:
     - Review results before proceeding
     - Stop the pipeline if issues are detected
     - Control the evaluation flow

### 3. **Stage-by-Stage Display**
   Each stage shows:
   - **Stage Header**: Clearly labeled with emoji icons
   - **Tool Used**: Which analysis tool was employed
   - **Tool Results**: Structured data display
     - Stage 2 (Quality): Interactive metrics and issue tables
     - Stage 3 (Sensitive): List of detected sensitive attributes
     - Stage 4 (Imbalance): Expandable details per column
     - Stage 4.5 (Fairness): **Interactive visualizations**
   - **Agent Analysis**: AI-generated insights
   - **Recommendations**: Actionable suggestions (when available)

### 4. **Stage 4.5: Target Fairness Analysis** (Special Features)
   When a target column is specified:
   - **Main Visualizations**: Overall target distribution charts
   - **Scale-Based Visualizations**: High, Medium, Low scale groupings
   - **Individual Combinations**: 
     - Select which specific combinations to view from dropdown
     - Only display combinations you're interested in (prevents overwhelming UI)
     - Shows first 20 combinations by default for performance

### 5. **View Previous Results**
   - Browse all previous evaluation reports
   - Three tabs:
     - **Full Report**: Complete evaluation text report
     - **Agent Summary**: Condensed findings
     - **Visualizations**: All generated images organized by folder
   - Images grouped by combination folders for easy navigation

## GUI vs Terminal Mode

| Feature | Terminal Mode | GUI Mode |
|---------|--------------|----------|
| Interface | Command-line | Web browser |
| Configuration | Edit code | Interactive forms |
| Dataset Upload | Manual copy | Upload button |
| Results Display | Text files | Interactive tabs/dropdowns |
| Visualizations | Save to disk | Embedded in page |
| Step Control | Runs all stages | Stop after each stage |
| Previous Results | Manual file browsing | Built-in viewer |
| Progress Tracking | Console output | Visual indicators |

## Technical Details

### File Structure
```
src/
  ├── main.py           # Entry point with mode selection
  ├── gui_app.py        # Streamlit GUI application
  ├── pipeline.py       # Core evaluation pipeline (shared)
  └── ...

reports/
  └── {dataset}_{timestamp}/
      ├── evaluation_report.txt
      ├── agent_summary.txt
      └── images/
          └── {combination_folders}/
```

### How GUI Mode Works
1. **main.py** detects `RUN_MODE="gui"` and launches Streamlit
2. **gui_app.py** provides the web interface
3. Same **pipeline.py** is used by both modes
4. GUI reads generated reports and displays them interactively
5. All reports are saved to disk (same as terminal mode)

### Session State Management
- Streamlit maintains session state for:
  - Selected mode (new/view)
  - Current pipeline instance
  - Evaluation results
  - Step approval status
- State persists during a single browser session

## Best Practices

### 1. **Step-by-Step Review**
   - Use the "Continue" button to review each stage
   - Stop if you notice issues in early stages
   - No need to wait for full pipeline completion

### 2. **Visualization Selection**
   - In Stage 4.5, only select the combinations you need to review
   - Viewing too many images at once can slow down the browser
   - All images are saved to disk for detailed offline analysis

### 3. **Dataset Management**
   - Upload datasets through GUI to automatically add them to `src/data/`
   - Uploaded datasets become available for future evaluations
   - Keep datasets organized by naming them descriptively

### 4. **Report Management**
   - Reports are timestamped and never overwritten
   - View previous results anytime without re-running pipeline
   - Clean old reports manually from `reports/` folder when needed

## Troubleshooting

### GUI Won't Start
```bash
# Install Streamlit
pip install streamlit

# Verify installation
streamlit --version

# Run directly
cd src
streamlit run gui_app.py
```

### Port Already in Use
If you see "Port 8501 is already in use":
```bash
# Kill existing Streamlit process or specify different port
streamlit run gui_app.py --server.port 8502
```

### Images Not Displaying
- Ensure the pipeline completed successfully
- Check that `images/` folder exists in the report directory
- Verify image paths are correct

### Performance Issues
- Limit number of images displayed simultaneously
- Use smaller datasets for testing
- Close unused browser tabs

## Advanced Usage

### Custom Styling
The GUI uses custom CSS in `gui_app.py`. Modify the `st.markdown()` CSS section to change:
- Colors
- Fonts
- Layout spacing
- Button styles

### Adding New Features
To add new GUI features:
1. Modify `gui_app.py`
2. Add new display functions for custom visualizations
3. Update session state management if needed
4. Keep terminal mode functionality unchanged

## Support
For issues or questions:
- Check that all dependencies are installed
- Verify dataset format (CSV with proper encoding)
- Review terminal output for error messages
- Ensure API keys are configured (if using Grok/Gemini)
