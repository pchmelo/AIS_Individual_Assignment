import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cli import FairnessEvaluator

# =============================================================================
# Configuration
# =============================================================================
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yml")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "adult-all.csv")

# Load API key (Option 1: Direct assignment - not recommended for production) or (Option 2: Load from .env file)
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")  # Get free key at https://openrouter.ai/keys

# =============================================================================
# Initialize Evaluator
# =============================================================================
evaluator = FairnessEvaluator(
    config_path=CONFIG_PATH,
    api_key=API_KEY,
)

# =============================================================================
# Run Diagnostic Checks (recommended before first evaluation)
# =============================================================================
# Uncomment to run diagnostics:
# check = evaluator.doctor(dataset=DATASET_PATH)
# if not check.all_passed:
#     print("Fix the issues above before running evaluation.")
#     exit(1)

# =============================================================================
# Run Evaluation
# =============================================================================
result = evaluator.evaluate(
    data=DATASET_PATH,
    target="Income",
)

print(result.pdf_path if result.success else result.error)
