"""
helper.py
=========
Small generic utilities not tied to any specific pipeline stage.
"""

import os

import pandas as pd

from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, PROJECT_ROOT


def load_data(filename: str) -> pd.DataFrame:
    """Load a CSV from data/raw/ by filename."""
    return pd.read_csv(DATA_RAW_DIR / filename)


def save_processed(df: pd.DataFrame, filename: str) -> None:
    """Save a DataFrame as CSV into data/processed/."""
    path = DATA_PROCESSED_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def get_project_root() -> str:
    return str(PROJECT_ROOT)
