"""Dataset loading and validation."""
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "Student_Performance.csv"
TARGET = "Performance Index"
FEATURES = [
    "Hours Studied",
    "Previous Scores",
    "Extracurricular Activities",
    "Sleep Hours",
    "Sample Question Papers Practiced",
]

def load_data(path=DEFAULT_DATA_PATH, deduplicate=True):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path)
    required = FEATURES + [TARGET]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    frame = frame[required].copy()
    if deduplicate:
        frame = frame.drop_duplicates().reset_index(drop=True)
    if frame.empty:
        raise ValueError("Dataset is empty after validation.")
    if frame.isna().any().any():
        raise ValueError("Dataset contains missing values; inspect and resolve them before training.")
    if not set(frame["Extracurricular Activities"].unique()).issubset({"Yes", "No"}):
        raise ValueError("Unexpected values in 'Extracurricular Activities'; expected Yes/No.")
    if not pd.api.types.is_numeric_dtype(frame[TARGET]):
        raise ValueError(f"Target '{TARGET}' must be numeric.")
    return frame
