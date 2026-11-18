import numpy as np
import pandas as pd
import pytest

from src.data import FEATURES
from src.robustness import run_robustness


def make_dataset(path):
    rng = np.random.default_rng(7)
    size = 100
    frame = pd.DataFrame({
        "Hours Studied": rng.integers(1, 10, size),
        "Previous Scores": rng.integers(40, 100, size),
        "Extracurricular Activities": rng.choice(["Yes", "No"], size),
        "Sleep Hours": rng.integers(4, 10, size),
        "Sample Question Papers Practiced": rng.integers(0, 10, size),
    })
    frame["Performance Index"] = (
        2.5 * frame["Hours Studied"]
        + 0.8 * frame["Previous Scores"]
        + 0.4 * frame["Sleep Hours"]
        + 0.2 * frame["Sample Question Papers Practiced"]
        + (frame["Extracurricular Activities"] == "Yes") * 2
    )
    frame.to_csv(path, index=False)


def test_nested_robustness_writes_reproducible_outputs(tmp_path):
    data_path = tmp_path / "sample.csv"
    output_dir = tmp_path / "reports"
    make_dataset(data_path)

    results, summary = run_robustness(
        data_path=data_path,
        output_dir=output_dir,
        n_splits=3,
        random_state_start=11,
        inner_splits=3,
        make_plots=True,
    )

    assert len(results) == 3
    assert results["seed"].tolist() == [11, 12, 13]
    assert set(["selected_model", "mae", "rmse", "r2"]).issubset(results.columns)
    assert set(results["selected_model"]).issubset({
        "Linear Regression", "Ridge", "Lasso", "Gradient Boosting",
        "HistGradientBoosting", "Random Forest", "Extra Trees",
    })
    assert summary["n_splits"] == 3
    assert summary["inner_splits"] == 3
    assert summary["selected_model_counts"]
    assert "Outer test rows are excluded from inner model selection." in summary["leakage_controls"]
    assert "not confidence intervals" in summary["uncertainty_note"]
    for metric in ("mae", "rmse", "r2"):
        assert "empirical_percentile_2_5" in summary["metrics"][metric]
        assert "empirical_percentile_97_5" in summary["metrics"][metric]

    assert (output_dir / "robustness_seed_results.csv").exists()
    assert (output_dir / "robustness_summary.json").exists()
    assert (output_dir / "robustness_metric_distributions.png").exists()
    assert (output_dir / "robustness_metrics_by_seed.png").exists()


def test_robustness_requires_at_least_two_outer_splits(tmp_path):
    data_path = tmp_path / "sample.csv"
    make_dataset(data_path)
    with pytest.raises(ValueError, match="at least 2"):
        run_robustness(data_path=data_path, output_dir=tmp_path, n_splits=1)


def test_robustness_requires_at_least_two_inner_splits(tmp_path):
    data_path = tmp_path / "sample.csv"
    make_dataset(data_path)
    with pytest.raises(ValueError, match="inner_splits"):
        run_robustness(
            data_path=data_path, output_dir=tmp_path,
            n_splits=2, inner_splits=1, make_plots=False,
        )
