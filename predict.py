"""Prediction helpers with schema, type, finite-value and training-domain validation."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .data import FEATURES, DEFAULT_DATA_PATH, load_data

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "best_pipeline.joblib"
NUMERIC_FEATURES = [
    "Hours Studied", "Previous Scores", "Sleep Hours",
    "Sample Question Papers Practiced",
]


def get_training_ranges(data_path=DEFAULT_DATA_PATH):
    """Return observed inclusive min/max for numeric input features."""
    frame = load_data(data_path, deduplicate=True)
    return {
        feature: (float(frame[feature].min()), float(frame[feature].max()))
        for feature in NUMERIC_FEATURES
    }


def load_model(model_path=DEFAULT_MODEL_PATH):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}. Run `python -m src.train` first.")
    return joblib.load(path)


def _validate_frame(frame):
    missing = sorted(set(FEATURES) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(FEATURES))
    if missing or extra:
        raise ValueError(f"Feature schema mismatch. Missing: {missing}; unexpected: {extra}")
    frame = frame[FEATURES].copy()

    category = frame["Extracurricular Activities"]
    if category.isna().any() or not category.isin(["Yes", "No"]).all():
        raise ValueError("Extracurricular Activities must contain only 'Yes' or 'No'.")

    ranges = get_training_ranges()
    for feature in NUMERIC_FEATURES:
        values = pd.to_numeric(frame[feature], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{feature} must contain finite numeric values.")
        low, high = ranges[feature]
        if not values.between(low, high).all():
            raise ValueError(
                f"{feature} must be within the observed training range [{low:g}, {high:g}]."
            )
        frame[feature] = values
    return frame


def predict(input_data, model=None):
    if isinstance(input_data, dict):
        frame = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        frame = input_data.copy()
    else:
        raise TypeError("input_data must be a dictionary or pandas DataFrame.")
    frame = _validate_frame(frame)
    if model is None:
        model = load_model()
    return model.predict(frame)


if __name__ == "__main__":
    sample = {
        "Hours Studied": 6, "Previous Scores": 75,
        "Extracurricular Activities": "Yes", "Sleep Hours": 7,
        "Sample Question Papers Practiced": 4,
    }
    print(f"Predicted Performance Index: {predict(sample)[0]:.2f}")
