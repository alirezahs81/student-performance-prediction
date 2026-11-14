"""Reusable preprocessing and regression pipeline."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .data import FEATURES

NUMERIC_FEATURES = [
    "Hours Studied", "Previous Scores", "Sleep Hours",
    "Sample Question Papers Practiced",
]
CATEGORICAL_FEATURES = ["Extracurricular Activities"]

def build_pipeline(estimator=None):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocessor", preprocess),
        ("regressor", estimator if estimator is not None else LinearRegression()),
    ])
