import joblib
import pandas as pd
import pytest
from src.data import FEATURES, TARGET, load_data
from src.pipeline import build_pipeline
from src.predict import predict
from sklearn.linear_model import LinearRegression

def sample_frame():
    return pd.DataFrame([
        [5, 70, "Yes", 7, 3],
        [2, 55, "No", 6, 1],
        [8, 90, "Yes", 8, 6],
        [4, 65, "No", 7, 2],
    ], columns=FEATURES)

def test_load_data_schema_and_dedup(data_path):
    frame = load_data(data_path)
    assert list(frame.columns) == FEATURES + [TARGET]
    assert not frame.duplicated().any()
    assert frame.isna().sum().sum() == 0

def test_preprocessing_and_training():
    frame = sample_frame()
    target = pd.Series([70.0, 52.0, 90.0, 63.0])
    model = build_pipeline(LinearRegression()).fit(frame, target)
    prediction = model.predict(frame.iloc[:1])
    assert len(prediction) == 1
    assert pd.notna(prediction[0])

def test_model_save_load_and_prediction(tmp_path):
    frame = sample_frame()
    target = pd.Series([70.0, 52.0, 90.0, 63.0])
    model = build_pipeline(LinearRegression()).fit(frame, target)
    path = tmp_path / "pipeline.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    assert restored.predict(frame).shape == (4,)
    result = predict(frame.iloc[0].to_dict(), restored)
    assert len(result) == 1

def test_feature_schema_validation():
    with pytest.raises(ValueError, match="Feature schema mismatch"):
        predict({"Hours Studied": 5})

def test_training_pipeline_on_real_dataset(data_path):
    frame = load_data(data_path)
    X, y = frame[FEATURES], frame[TARGET]
    model = build_pipeline(LinearRegression()).fit(X, y)
    assert model.predict(X.iloc[:3]).shape == (3,)


def test_prediction_rejects_unknown_category():
    row = sample_frame().iloc[0].to_dict()
    row["Extracurricular Activities"] = "Sometimes"
    with pytest.raises(ValueError, match="only 'Yes' or 'No'"):
        predict(row)


def test_prediction_rejects_non_finite_numeric_values():
    row = sample_frame().iloc[0].to_dict()
    row["Hours Studied"] = float("inf")
    with pytest.raises(ValueError, match="finite numeric"):
        predict(row)


def test_prediction_rejects_values_outside_training_domain():
    row = sample_frame().iloc[0].to_dict()
    row["Previous Scores"] = 1000
    with pytest.raises(ValueError, match="observed training range"):
        predict(row)


def test_prediction_rejects_non_numeric_values():
    row = sample_frame().iloc[0].to_dict()
    row["Sleep Hours"] = "many"
    with pytest.raises(ValueError, match="finite numeric"):
        predict(row)
