"""Evaluate holdout robustness across repeated random train/test splits.

The selected estimator is Linear Regression, matching the model chosen by
src.train. Every split gets a fresh preprocessing/model pipeline; no test
partition is used for fitting or model selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .data import DEFAULT_DATA_PATH, TARGET, load_data
from .pipeline import build_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports"


def run_robustness(
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    n_splits: int = 30,
    random_state_start: int = 0,
    make_plots: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Run repeated 80/20 holdouts and save per-split and aggregate results."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    frame = load_data(data_path, deduplicate=True)
    if len(frame) < 10:
        raise ValueError("At least 10 unique rows are required for robustness evaluation.")

    X = frame.drop(columns=TARGET)
    y = frame[TARGET]
    rows: list[dict[str, float | int]] = []

    for seed in range(random_state_start, random_state_start + n_splits):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=seed
        )
        pipeline = build_pipeline(LinearRegression())
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rows.append({
            "seed": seed,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(y_test, predictions)),
        })

    results = pd.DataFrame(rows)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "robustness_seed_results.csv"
    summary_path = output / "robustness_summary.json"
    results.to_csv(csv_path, index=False)

    metric_summary = {}
    for metric in ("mae", "rmse", "r2"):
        values = results[metric]
        metric_summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    summary = {
        "estimator": "LinearRegression",
        "protocol": "Repeated random 80/20 holdout; a fresh sklearn Pipeline is fitted per split.",
        "selection_note": (
            "The estimator was fixed in advance because Linear Regression was selected "
            "by the project's training cross-validation. This robustness run does not "
            "perform model selection on the holdout partitions."
        ),
        "n_splits": int(n_splits),
        "random_state_start": int(random_state_start),
        "random_state_end": int(random_state_start + n_splits - 1),
        "test_size": 0.20,
        "n_unique_rows": int(len(frame)),
        "n_features": int(X.shape[1]),
        "metrics": metric_summary,
        "artifacts": {
            "per_split_csv": csv_path.name,
            "summary_json": summary_path.name,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if make_plots:
        _save_metric_boxplot(results, output / "robustness_metric_distributions.png")
        _save_metric_trends(results, output / "robustness_metrics_by_seed.png")

    print(results.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print("\nAggregate metrics:")
    for metric, stats in metric_summary.items():
        print(
            f"{metric.upper():>4}: mean={stats['mean']:.5f}, "
            f"std={stats['std']:.5f}, min={stats['min']:.5f}, max={stats['max']:.5f}"
        )
    print(f"\nSaved reports to: {output.resolve()}")
    return results, summary


def _save_metric_boxplot(results: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, metric in zip(axes, ("mae", "rmse", "r2")):
        axis.boxplot(results[metric], tick_labels=[metric.upper()])
        axis.set_title(f"{metric.upper()} across splits")
        axis.set_ylabel("Score")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Robustness: metric distributions across random holdouts")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _save_metric_trends(results: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for axis, metric in zip(axes, ("mae", "rmse", "r2")):
        axis.plot(results["seed"], results[metric], marker="o", linewidth=1)
        axis.set_ylabel(metric.upper())
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Random seed")
    figure.suptitle("Robustness metrics by random seed")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-splits", type=int, default=30)
    parser.add_argument("--random-state-start", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    run_robustness(
        data_path=args.data,
        output_dir=args.output_dir,
        n_splits=args.n_splits,
        random_state_start=args.random_state_start,
        make_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
