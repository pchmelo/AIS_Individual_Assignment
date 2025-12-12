# Dataset Quality & Fairness Evaluation System

An AI-powered system for evaluating datasets for data quality issues and fairness concerns. Supports both **Terminal Mode** and **GUI Mode**.

## Features

- 🔍 **Comprehensive Data Quality Analysis**: Detects missing values, type inconsistencies, suspicious patterns
- ⚖️ **Fairness Evaluation**: Identifies sensitive attributes and analyzes target distribution bias
- 📊 **Rich Visualizations**: Interactive charts showing fairness metrics across attribute combinations
- 🤖 **Multi-Model Support**: IBM Granite (local), Grok, or Google Gemini
- 💻 **Dual Interface**: Choose between terminal CLI or web-based GUI
- 📁 **Report Management**: Automatic timestamped reports with detailed findings

## Quick Start

### Installation

```bash
# Install base dependencies
pip install pandas numpy matplotlib seaborn

# Install GUI dependencies (if using GUI mode)
pip install -r requirements-gui.txt

# Configure API keys (if using Grok or Gemini)
# Set environment variables:
# - OPENROUTER_API_KEY for Grok
# - GOOGLE_API_KEY for Gemini
```

### Running the Application

**1. Choose Your Mode**

Edit `src/main.py`:
```python
RUN_MODE = "terminal"  # or "gui"
```

**2. Run**

```bash
python src/main.py
```

## Terminal Mode

Simple command-line interface:
- Edit configuration in `src/main.py` (dataset name, target column, model)
- Run the script
- Results saved to `reports/{dataset}_{timestamp}/`

## GUI Mode

Interactive web interface:

### Features:
- 🆕 **New Evaluation**: Upload datasets, select models, configure targets
- 📂 **View Previous Results**: Browse and visualize past reports
- ⏸️ **Step-by-Step Control**: Continue or stop after each stage
- 📊 **Interactive Visualizations**: Select which fairness combinations to view
- 📋 **Structured Display**: Clean presentation of tool results and agent analysis

### Usage:
1. Set `RUN_MODE = "gui"` in `src/main.py`
2. Run `python src/main.py`
3. Browser opens automatically to the web interface
4. Choose "New Evaluation" or "View Previous Results"

For detailed GUI documentation, see [GUI_GUIDE.md](GUI_GUIDE.md)

## Pipeline Stages

1. **Stage 0**: Dataset Loading
2. **Stage 1**: Objective Inspection
3. **Stage 2**: Data Quality Analysis (missing values, type issues)
4. **Stage 3**: Sensitive Attribute Detection
5. **Stage 4**: Imbalance Analysis
6. **Stage 4.5**: Target Fairness Analysis (optional, requires target column)
7. **Stage 5**: Findings Integration
8. **Stage 6**: Recommendations
9. **Stage 7**: Report Generation

## Stage 4.5: Target Fairness Analysis

When a target column is specified, the system generates:
- **Scale-grouped visualizations**: High/medium/low count ranges
- **Individual combination graphs**: All pairs of sensitive attributes
- **Detailed metrics**: Target distribution across intersectional groups

Example combinations analyzed:
- Age × Education
- Sex × Race
- Age × Sex
- Race × Education
- ... (all C(n,2) pairs)

## Project Structure

```
individual_assignment/
├── src/
│   ├── main.py              # Entry point with mode selection
│   ├── gui_app.py           # Streamlit GUI application
│   ├── pipeline.py          # Core evaluation pipeline
│   ├── agents/              # AI agents (function caller, data analyst, conversational)
│   ├── tools/               # Analysis tools (fairness, dataset analysis)
│   └── data/                # Dataset storage
├── reports/                 # Generated evaluation reports
│   └── {dataset}_{timestamp}/
│       ├── evaluation_report.txt
│       ├── agent_summary.txt
│       └── images/          # Visualizations
├── docs/                    # Documentation
├── notebooks/               # Jupyter notebooks for exploration
├── requirements-gui.txt     # GUI dependencies
└── GUI_GUIDE.md            # Detailed GUI documentation
```

## Models

### 0: IBM Granite (Local)
- Runs locally using IBM Granite 3B model
- No API key required
- Good for privacy-sensitive data

### 1: Grok (API)
- Requires OpenRouter API key
- Set `OPENROUTER_API_KEY` environment variable
- Fast and powerful

### 2: Google Gemini (API)
- Requires Google API key
- Set `GOOGLE_API_KEY` environment variable
- Advanced reasoning capabilities

## Output

### Terminal Mode
- Console output with progress indicators
- Text reports in `reports/` folder
- Images saved to `reports/{dataset}_{timestamp}/images/`

### GUI Mode
- Interactive web interface
- Same reports generated as terminal mode
- Embedded visualizations
- Previous results browser

## Key Features

### Missing Data Detection
- Detects 15+ missing value indicators ('?', 'NA', 'NULL', whitespace, etc.)
- Identifies type inconsistencies (mixed numeric/string)
- Shows which values were converted to NA

### Fairness Analysis
- Automatic sensitive attribute detection
- Target distribution analysis across all attribute pairs
- Scale-grouped visualizations for clarity
- Individual combination graphs for detailed inspection

### Quality Assessment
- Mathematical facts only (no subjective opinions)
- Transparent tool outputs
- Problem descriptions for each issue
- Actionable recommendations

## Troubleshooting

### GUI won't start
```bash
pip install streamlit
streamlit --version
```

### Missing dependencies
```bash
pip install -r requirements-gui.txt
```

### API errors
- Verify API keys are set correctly
- Check network connection
- Try switching to local model (IBM Granite)

## Development

Built with:
- Python 3.x
- Pandas, NumPy for data processing
- Matplotlib, Seaborn for visualizations
- Streamlit for GUI
- Google Gemini / Grok APIs for AI analysis

## License

Academic project for AI & Society master's course.
