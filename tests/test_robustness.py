import numpy as np
import pandas as pd
import pytest

from src.data import FEATURES
from src import robustness
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


def test_nested_robustness_is_deterministic(tmp_path):
    data_path = tmp_path / "sample.csv"
    make_dataset(data_path)

    first, first_summary = run_robustness(
        data_path=data_path,
        output_dir=tmp_path / "first",
        n_splits=2,
        random_state_start=23,
        inner_splits=3,
        make_plots=False,
    )
    second, second_summary = run_robustness(
        data_path=data_path,
        output_dir=tmp_path / "second",
        n_splits=2,
        random_state_start=23,
        inner_splits=3,
        make_plots=False,
    )

    pd.testing.assert_frame_equal(first, second)
    assert first_summary == second_summary


def test_outer_test_rows_are_excluded_from_model_selection(tmp_path, monkeypatch):
    data_path = tmp_path / "sample.csv"
    make_dataset(data_path)
    original_split = robustness.train_test_split
    original_select = robustness._select_model
    split_indices = []

    def tracking_split(X, y, **kwargs):
        X_train, X_test, y_train, y_test = original_split(X, y, **kwargs)
        split_indices.append((set(X_train.index), set(X_test.index)))
        return X_train, X_test, y_train, y_test

    def guarded_select(X_train, y_train, inner_splits, seed):
        expected_train, outer_test = split_indices[-1]
        supplied_train = set(X_train.index)
        assert supplied_train == expected_train
        assert supplied_train.isdisjoint(outer_test)
        return original_select(X_train, y_train, inner_splits, seed)

    monkeypatch.setattr(robustness, "train_test_split", tracking_split)
    monkeypatch.setattr(robustness, "_select_model", guarded_select)

    results, _ = run_robustness(
        data_path=data_path,
        output_dir=tmp_path / "reports",
        n_splits=2,
        random_state_start=31,
        inner_splits=3,
        make_plots=False,
    )

    assert len(results) == 2
    assert len(split_indices) == 2


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
