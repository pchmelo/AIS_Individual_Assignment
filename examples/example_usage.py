import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

# Load API keys from .env before anything else
load_dotenv()

# Paths (relative to this file)
CONFIG_PATH  = os.path.join(os.path.dirname(__file__), "config.yml")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "adult-all.csv")

# ---------------------------------------------------------------------------
# Mode dispatch  —  controlled by  mode:  in config.yml
# ---------------------------------------------------------------------------
import yaml
with open(CONFIG_PATH, encoding="utf-8") as f:
    _mode = (yaml.safe_load(f) or {}).get("mode", "quick")

if _mode == "gui":
    # Requires: pip install -r requirements-gui.txt
    from gui import launch
    launch(config_path=CONFIG_PATH, dataset_path=DATASET_PATH)
    sys.exit(0)

# ---------------------------------------------------------------------------
# CLI / headless mode
# ---------------------------------------------------------------------------
from cli import FairnessEvaluator

evaluator = FairnessEvaluator(config_path=CONFIG_PATH)

# All settings (target column, sensitive attributes, pairs, mitigation) are
# read from config.yml automatically. Override any of them here if needed:
result = evaluator.evaluate(
    data=DATASET_PATH,
    # target="Income",                              # overrides target_column in config
    # sensitive_columns=["Sex", "Race", "Age"],     # overrides sensitive_attribute_analysis
    # sensitive_pairs=[["Sex", "Race"], ["Age", "Education"]],  # overrides pair_evaluation
    # max_pairs=3,                                  # cap auto-selected pairs
    # mitigation_techniques=["reweighting", "smote"],           # overrides mitigation_techniques
)

if result.success:
    print(f"\nReport: {result.report_dir}")
    if result.pdf_path:
        print(f"PDF:    {result.pdf_path}")
else:
    print(f"\nError: {result.error}")
