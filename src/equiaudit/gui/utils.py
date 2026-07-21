import os
import re

import streamlit as st
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_available_datasets() -> list:
    """Return dataset list available in the GUI.

    When FAIRNESS_DATASET_PATH is set (via launch(dataset_path=...)) only that
    single dataset is returned — the bundled development datasets are hidden.
    """
    env_dataset = os.environ.get("FAIRNESS_DATASET_PATH", "")
    if env_dataset and os.path.exists(env_dataset):
        return [os.path.basename(env_dataset)]
    # Development fallback: datasets bundled in src/data/
    data_dir = os.path.join(SRC_DIR, "data")
    if os.path.exists(data_dir):
        return [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    return []


def get_available_reports() -> list:
    """Return list of report directory names."""
    reports_dir = os.path.join(BASE_DIR, "reports")
    if os.path.exists(reports_dir):
        return [d for d in os.listdir(reports_dir) if os.path.isdir(os.path.join(reports_dir, d))]
    return []


def upload_dataset(uploaded_file) -> str:
    """Save an uploaded file to the data/ directory and return its name."""
    data_dir = os.path.join(SRC_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return uploaded_file.name


def get_dataset_columns(dataset_name: str) -> list:
    """Read only the header row and return column names."""
    try:
        env_dataset = os.environ.get("FAIRNESS_DATASET_PATH", "")
        if env_dataset and os.path.exists(env_dataset) and os.path.basename(env_dataset) == dataset_name:
            file_path = env_dataset
        else:
            data_dir = os.path.join(SRC_DIR, "data")
            file_path = os.path.join(data_dir, dataset_name)
        df = pd.read_csv(file_path, nrows=1)
        return list(df.columns)
    except Exception as e:
        st.error(f"Error reading dataset: {str(e)}")
        return []


# ---------------------------------------------------------------------------
# Report file parsing (used by the view-results page)
# ---------------------------------------------------------------------------

def parse_report_file(filepath: str):
    """Parse a markdown report file into (header_info, stages) or None.
    
    Supports new markdown format with YAML-like frontmatter and ## headers.
    """
    if not os.path.exists(filepath):
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    stages = {}
    lines = content.split("\n")

    # Extract header metadata from YAML-like frontmatter
    header_info = {}
    in_frontmatter = False
    frontmatter_end = 0
    
    for i, line in enumerate(lines[:50]):
        if line.strip() == "---":
            if not in_frontmatter:
                in_frontmatter = True
                continue
            else:
                frontmatter_end = i
                break
        
        if in_frontmatter and ":" in line:
            key, _, value = line.partition(":")
            key = key.strip().lower()
            value = value.strip()
            if key == "dataset":
                header_info["dataset"] = value
            elif key == "timestamp":
                header_info["timestamp"] = value
            elif key in ("target_column", "target"):
                header_info["target"] = value
            elif key in ("objective", "user_objective"):
                header_info["objective"] = value

    current_stage = None
    current_content: list = []

    for line in lines[frontmatter_end:]:
        # Match markdown ## Stage headers
        stage_match = re.match(r"^##\s+Stage\s+(\d+(?:\.\d+)?)[:\s]+(.*)", line)

        if stage_match:
            if current_stage:
                stages[current_stage] = "\n".join(current_content).strip()
            stage_num = stage_match.group(1)
            stage_name = stage_match.group(2).strip()
            current_stage = f"Stage {stage_num}: {stage_name}"
            current_content = []
        elif line.startswith("# ") and not line.startswith("## "):
            # Skip main title
            continue
        elif current_stage:
            current_content.append(line)

    if current_stage:
        stages[current_stage] = "\n".join(current_content).strip()

    return header_info, stages
