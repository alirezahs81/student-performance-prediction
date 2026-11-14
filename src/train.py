"""Train, cross-validate, evaluate and persist the best regression pipeline."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from .data import DEFAULT_DATA_PATH, TARGET, load_data
from .pipeline import build_pipeline

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

def candidate_models():
    return {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001, max_iter=20000),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1),
    }

def train(data_path=DEFAULT_DATA_PATH):
    frame = load_data(data_path, deduplicate=True)
    X, y = frame.drop(columns=TARGET), frame[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rows, fitted = [], {}
    for name, estimator in candidate_models().items():
        pipeline = build_pipeline(estimator)
        scores = cross_validate(
            pipeline, X_train, y_train, cv=cv, n_jobs=-1,
            scoring={"mae": "neg_mean_absolute_error",
                     "mse": "neg_mean_squared_error", "r2": "r2"},
        )
        pipeline.fit(X_train, y_train)
        prediction = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, prediction)
        rows.append({
            "model": name,
            "cv_mae_mean": float(-scores["test_mae"].mean()),
            "cv_rmse_mean": float(np.sqrt(-scores["test_mse"].mean())),
            "cv_r2_mean": float(scores["test_r2"].mean()),
            "test_mae": float(mean_absolute_error(y_test, prediction)),
            "test_mse": float(mse),
            "test_rmse": float(np.sqrt(mse)),
            "test_r2": float(r2_score(y_test, prediction)),
        })
        fitted[name] = pipeline
    results = pd.DataFrame(rows).sort_values(
        ["cv_rmse_mean", "cv_mae_mean"], ascending=True
    ).reset_index(drop=True)
    best_name = str(results.iloc[0]["model"])
    best_pipeline = fitted[best_name]
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    joblib.dump(best_pipeline, MODELS_DIR / "best_pipeline.joblib")
    results.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    metadata = {
        "best_model": best_name,
        "selection_metric": "5-fold training CV RMSE (lower is better)",
        "random_state": 42,
        "test_size": 0.20,
        "n_rows_after_exact_deduplication": int(len(frame)),
        "n_features": int(X.shape[1]),
        "feature_names": list(X.columns),
        "target": TARGET,
        "python_version": __import__("platform").python_version(),
        "scikit_learn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "test_metrics": results.loc[results.model == best_name].iloc[0].to_dict(),
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(results.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nSelected: {best_name}")
    return results, best_pipeline, metadata

if __name__ == "__main__":
    train()
