import os
import re

import streamlit as st
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_available_datasets() -> list:
    """Return list of CSV filenames in the data/ directory."""
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
    """Parse a report text file into (header_info, stages) or None."""
    if not os.path.exists(filepath):
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    stages = {}
    lines = content.split("\n")

    # Extract header metadata
    header_info = {}
    for line in lines[:30]:
        if "Dataset:" in line:
            header_info["dataset"] = line.split("Dataset:")[1].strip()
        elif "Timestamp:" in line:
            header_info["timestamp"] = line.split("Timestamp:")[1].strip()
        elif "Target Column:" in line:
            header_info["target"] = line.split("Target Column:")[1].strip()
        elif "User Objective:" in line:
            header_info["objective"] = line.split("User Objective:")[1].strip()

    current_stage = None
    current_content: list = []

    for line in lines:
        stage_match = re.match(r"STAGE\s+(\d+(?:\.\d+)?)[:\s]+(.*)", line)

        if stage_match:
            if current_stage:
                stages[current_stage] = "\n".join(current_content).strip()
            stage_num = stage_match.group(1)
            stage_name = stage_match.group(2).strip()
            current_stage = f"Stage {stage_num}: {stage_name}"
            current_content = []
        elif line.startswith("=" * 10) or line.startswith("-" * 10):
            continue
        elif current_stage:
            current_content.append(line)

    if current_stage:
        stages[current_stage] = "\n".join(current_content).strip()

    return header_info, stages
