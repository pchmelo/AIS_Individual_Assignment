# Usage Guide

This document covers installation, configuration, and all execution modes of EquiAudit.

## Prerequisites

```bash
# From PyPI (recommended)
pip install equiaudit              # core pipeline
pip install "equiaudit[gui]"       # optional: Streamlit GUI

# From source
pip install -r requirements.txt
pip install -r requirements-gui.txt    # optional: Streamlit GUI
```

Pass an absolute path to your dataset at runtime, or use one of the bundled example datasets (`adult-all.csv`, `GermanCredit.csv`, `bank-additional.csv`). The bundled example is the UCI Adult Census dataset, target column `Income`.


## Configuration File

All runtime behaviour is controlled by a single YAML file. Copy `examples/config.yml` as your starting point.

```yaml
# ── Model selection ───────────────────────────────────────────────────────────
default_model: "gemini-flash"          # must match a key under `models:`

models:
  gemini-flash:
    provider: gemini                   # "openrouter" | "gemini" | "ollama"
    model: "gemini-2.0-flash"          # provider-specific model identifier

  openrouter-free:
    provider: openrouter
    model: "openrouter/owl-alpha"
    base_url: "https://openrouter.ai/api/v1"
    model_info:
      function_calling: true
      vision: false
      structured_output: true

  deepseek-local:
    provider: ollama
    model: "deepseek-r1:8b"

# ── Tool managers ─────────────────────────────────────────────────────────────
tools:
  fairness:
    class: "FairnessTools"
    description: "Fairness analysis and dataset inspection"
  bias_mitigation:
    class: "BiasMitigationTools"
    description: "Bias mitigation techniques"

# ── Agents ────────────────────────────────────────────────────────────────────
agents:
  file_parser:
    type: FunctionCallerAgent          # FunctionCallerAgent | DataAnalystAgent | ConversationalAgent
    tools: [fairness]
    stages: [parsing]
    reflect_on_tool_use: true
    # model: gemini-flash              # per-agent model override (optional)

  quality_analyst:
    type: DataAnalystAgent
    tools: [fairness]
    stages: [quality_analysis]

  recommendation:
    type: ConversationalAgent
    tools: []
    stages: [recommendation]

# ── Pipeline stage ordering ────────────────────────────────────────────────────
pipeline:
  stages:
    - name: parsing
      description: "Parse and load the dataset"
      agents: [file_parser]
    - name: quality_analysis
      description: "Analyse data quality"
      agents: [quality_analyst]
    - name: recommendation
      description: "Generate recommendations"
      agents: [recommendation]

# ── Launch mode ───────────────────────────────────────────────────────────────
mode: gui                              # "gui" | "quick"

# ── Evaluation settings ───────────────────────────────────────────────────────
target_column: "Income"
use_humanizer: false                   # post-process output for readability
generate_detailed_report: true         # detailed_metrics_report.md/.pdf
generate_executive_summary: true       # top-level summary agent pass

sensitive_attribute_analysis:
  type: auto                           # "auto" | "restricted"
  attributes: ["Sex", "Race", "Age"]   # used only when type: restricted
  discretization: enable               # "enable" | "disable"
  method: auto                         # "auto" | "equal_width" | "equal_frequency"
  number_of_bins: 5                    # bins when method != auto
  continuous_threshold: 10             # unique-value count below which column is already categorical

pair_evaluation:
  type: auto                           # "auto" | "restricted"
  max_pairs: 2                         # cap when type: auto; null = unlimited
  # pairs: [["Sex", "Race"]]           # explicit list when type: restricted

mitigation_techniques:
  techniques: ["reweighting", "resampling"]
  # available: "reweighting", "resampling", "smote", "oversampling", "undersampling"

ml_evaluation:
  model_type: "Random Forest"          # "Random Forest" | "Gradient Boosting" | "Logistic Regression" | "SVC"
  model_params:
    "Random Forest":
      n_estimators: 100
      max_depth: null
```

### API Keys

Set credentials as environment variables or in a `.env` file in the project root:

```
OPENROUTER_API_KEY=sk-or-v1-...
GOOGLE_API_KEY=...
```

Ollama requires a running local server (`ollama serve`) and no API key.


## Execution Modes

### 1. GUI Mode (Streamlit)

Launches an interactive browser-based interface for step-by-step evaluation with per-stage confirmation dialogs.

```bash
# Via example script (set mode: gui in examples/config.yml)
python examples/example_usage.py

# From source: via main.py flag
python src/equiaudit/main.py --gui

# Directly via the GUI module (from source)
python -m streamlit run src/equiaudit/gui/app.py
```

Requires `pip install "equiaudit[gui]"` (or `requirements-gui.txt` from source). Opens on `http://localhost:8501`.


### 2. Quick Mode (Headless, config-driven)

Runs a non-interactive evaluation using settings from the active config. Intended for local smoke-testing without writing Python.

```bash
# Via example script (set mode: quick in examples/config.yml)
python examples/example_usage.py

# From source: via main.py flag
python src/equiaudit/main.py --quick
```

The dataset, target column, sensitive attributes, mitigation techniques, and all evaluation settings are read from `config.yml` (searched first in `$CWD`, then the bundled default).


### 3. CLI Mode (Headless, argument-driven)

Full headless evaluation with per-run argument overrides.

```bash
# Minimal: detect target and sensitive attributes automatically
equiaudit --data adult-all.csv

# Explicit target
equiaudit --data adult-all.csv --target Income

# Override model and specify sensitive columns
equiaudit --data adult-all.csv \
          --target Income \
          --model gemini-flash \
          --sensitive "Sex,Race,Age"

# Point to a custom config and output directory
equiaudit --config examples/config.yml \
          --data adult-all.csv \
          --output ./reports \
          --target Income

# Load API key from .env file
equiaudit --data adult-all.csv --env-file .env

# Verify config without running evaluation
equiaudit --verify

# Skip PDF generation
equiaudit --data adult-all.csv --no-pdf

# Suppress progress output
equiaudit --data adult-all.csv --quiet
```

> When running from source without installing, replace `equiaudit` with `python -m equiaudit.cli`.

#### Full CLI Reference

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--data` | `-d` | str | — | Path to CSV dataset file |
| `--target` | `-t` | str | config / auto | Target column for classification |
| `--objective` | `-o` | str | auto-generated | Custom evaluation prompt |
| `--sensitive` | `-s` | str | config / auto | Comma-separated sensitive column names |
| `--config` | `-c` | str | bundled default | YAML config file path |
| `--model` | `-m` | str | config default | Model name key (must exist in config `models:`) |
| `--env-file` | `-e` | str | — | Path to `.env` for API key loading |
| `--output` | `-O` | str | `reports/` | Report output directory |
| `--no-pdf` | — | flag | false | Disable PDF generation |
| `--verify` | `-V` | flag | false | Config check only, no evaluation |
| `--quiet` | `-q` | flag | false | Suppress progress messages |

`--sensitive` only sets `confirmed_sensitive`; pair selection and mitigation still read from the config unless overridden via the Python API (see below).


### 4. Python API

`FairnessEvaluator` (in `src/equiaudit/cli/evaluator.py`) is the programmatic entry point.

```python
from equiaudit.cli import FairnessEvaluator

evaluator = FairnessEvaluator(
    config_path="examples/config.yml",  # path to your config
    output_dir="./reports",             # report destination
    model="gemini-flash",               # optional model override
    api_key=None,                       # optional; reads from env if None
    verbose=True,
)

# ── Verification (recommended before evaluate) ────────────────────────────────
check = evaluator.verify()
if not check.success:
    print(check.errors)
    raise SystemExit(1)

# ── Basic evaluation ──────────────────────────────────────────────────────────
result = evaluator.evaluate(
    data="adult-all.csv",              # absolute path, or filename of a bundled dataset
    target="Income",
)
print(result.report_dir)              # e.g. reports/adult-all.csv_20260601_113129/

# ── Full parameter control ────────────────────────────────────────────────────
result = evaluator.evaluate(
    data="adult-all.csv",
    target="Income",
    objective="Evaluate census data for gender and racial bias in income prediction.",
    sensitive_columns=["Sex", "Race", "Age"],
    sensitive_pairs=[["Sex", "Race"], ["Age", "Sex"]],
    mitigation_techniques=["reweighting", "smote"],
    ml_config={
        "enabled": True,
        "model_type": "Logistic Regression",
        "model_params": {"C": 0.5, "penalty": "l2"},
        "test_size": 0.2,
    },
    output_dir="./my_reports",
    generate_pdf=True,
    max_pairs=3,                       # cap auto pair selection; ignored when sensitive_pairs is set
)

# ── Inspect result ────────────────────────────────────────────────────────────
assert result.success, result.error
print(result.dataset)                 # "adult-all.csv"
print(result.target_column)           # "Income"
print(result.report_dir)              # absolute path to output folder
print(result.pdf_path)                # path to evaluation_report.pdf (or None)
print(result.markdown_path)           # path to evaluation_report.md
print(result.json_path)               # path to stage_data.json
print(result.stages_completed)        # list of stage keys that finished
print(result.warnings)                # non-fatal warnings (e.g. missing PDF)
```

#### Passing a DataFrame directly

```python
import pandas as pd
df = pd.read_csv("my_data.csv")

result = evaluator.evaluate(
    data=df,                           # DataFrame written to a temp CSV before the pipeline runs
    target="label",
    sensitive_columns=["gender"],
)
```

The DataFrame is saved as a temporary CSV before the pipeline runs.


## Output Artifacts

Every run creates a timestamped subdirectory under the reports folder:

```
reports/
└── adult-all.csv_20260601_113129/
    ├── evaluation_report.md          # Markdown narrative report
    ├── evaluation_report.pdf         # PDF rendered from the Markdown
    ├── detailed_metrics_report.md    # Per-group fairness metrics table
    ├── detailed_metrics_report.pdf   # PDF version of the above
    ├── stage_data.json               # Raw structured output from every stage
    └── images/
        ├── target_distribution.png
        ├── sensitive_distributions.png
        ├── target_by_sensitive.png
        └── <Sex_Race_combinations>/
            ├── high_scale.png
            ├── medium_scale.png
            └── individual_combinations/
                └── *.png
```

`stage_data.json` is the machine-readable record of every stage result, keyed by stage name. It contains `tool_used`, `tool_result` (full dict returned by the tool function), and `agent_analysis` (LLM narrative).


## Worked Example: German Credit Dataset

```python
from equiaudit.cli import FairnessEvaluator

evaluator = FairnessEvaluator(config_path="examples/config.yml")

result = evaluator.evaluate(
    data="GermanCredit.csv",
    target="Risk",
    sensitive_columns=["Sex", "Age"],
    sensitive_pairs=[["Sex", "Age"]],
    mitigation_techniques=["reweighting"],
    ml_config={
        "enabled": True,
        "model_type": "Gradient Boosting",
        "model_params": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4},
        "test_size": 0.25,
    },
)

if result.success:
    print(f"Report: {result.report_dir}")
```

Equivalent CLI invocation:

```bash
equiaudit \
  --data GermanCredit.csv \
  --target Risk \
  --sensitive "Sex,Age" \
  --config examples/config.yml
```

Mitigation techniques and ML model config come from `examples/config.yml` in the CLI path; use the Python API if you need per-run overrides beyond what CLI flags expose.
