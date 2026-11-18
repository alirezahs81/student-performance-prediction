"""Nested repeated-holdout evaluation with fold-local model selection.

Each outer split is treated as an unseen evaluation set. Candidate estimators
are compared using inner CV on the outer training partition only; the selected
pipeline is then refitted on that partition and scored once on the outer test
partition. Preprocessing remains inside each sklearn Pipeline.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split

from .data import DEFAULT_DATA_PATH, TARGET, load_data
from .pipeline import build_pipeline
from .train import candidate_models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports"


def _select_model(X_train, y_train, inner_splits: int, seed: int):
    """Select a candidate using only the supplied outer-training data."""
    cv = KFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    selection_rows = []
    for name, estimator in candidate_models().items():
        scores = cross_validate(
            build_pipeline(estimator),
            X_train,
            y_train,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=1,
        )
        mean_rmse = float(np.sqrt(-scores["test_score"].mean()))
        selection_rows.append((mean_rmse, name, estimator))

    # Stable tie-breaking by model name makes the protocol deterministic.
    _, selected_name, selected_estimator = min(
        selection_rows, key=lambda row: (row[0], row[1])
    )
    return selected_name, build_pipeline(selected_estimator)


def run_robustness(
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    n_splits: int = 10,
    random_state_start: int = 0,
    make_plots: bool = True,
    inner_splits: int = 3,
) -> tuple[pd.DataFrame, dict]:
    """Run nested repeated 80/20 holdouts and save metrics and diagnostics."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if inner_splits < 2:
        raise ValueError("inner_splits must be at least 2.")

    frame = load_data(data_path, deduplicate=True)
    if len(frame) < max(10, inner_splits * 2):
        raise ValueError("Not enough unique rows for nested robustness evaluation.")

    X = frame.drop(columns=TARGET)
    y = frame[TARGET]
    rows: list[dict[str, float | int | str]] = []

    for seed in range(random_state_start, random_state_start + n_splits):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=seed
        )

        selected_name, pipeline = _select_model(
            X_train, y_train, inner_splits=inner_splits, seed=seed
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rows.append({
            "seed": seed,
            "selected_model": selected_name,
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
            "empirical_percentile_2_5": float(values.quantile(0.025)),
            "empirical_percentile_97_5": float(values.quantile(0.975)),
        }

    model_counts = results["selected_model"].value_counts().sort_index()
    summary = {
        "estimator_selection": "Best of the seven project candidates by inner-CV RMSE, independently within each outer split.",
        "protocol": (
            "Repeated random 80/20 outer holdouts. Within each outer training "
            "partition, candidate models are ranked by shuffled inner K-fold CV "
            "RMSE; the selected pipeline is refitted on the full outer training "
            "partition and evaluated once on the outer test partition."
        ),
        "leakage_controls": [
            "Outer test rows are excluded from inner model selection.",
            "Imputation, scaling, and encoding are fitted inside each CV fold through sklearn Pipelines.",
            "The outer test partition is used only for final scoring within its split.",
        ],
        "uncertainty_note": (
            "The 2.5th and 97.5th percentiles describe the empirical distribution "
            "of repeated-split scores; they are not confidence intervals. Outer "
            "holdouts overlap across seeds, so split-level scores are dependent."
        ),
        "n_splits": int(n_splits),
        "inner_splits": int(inner_splits),
        "random_state_start": int(random_state_start),
        "random_state_end": int(random_state_start + n_splits - 1),
        "test_size": 0.20,
        "n_unique_rows": int(len(frame)),
        "n_features": int(X.shape[1]),
        "selected_model_counts": {str(name): int(count) for name, count in model_counts.items()},
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
            f"std={stats['std']:.5f}, empirical 2.5–97.5%="
            f"[{stats['empirical_percentile_2_5']:.5f}, "
            f"{stats['empirical_percentile_97_5']:.5f}]"
        )
    print("\nSelected models:")
    print(model_counts.to_string())
    print(f"\nSaved reports to: {output.resolve()}")
    return results, summary


def _save_metric_boxplot(results: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, metric in zip(axes, ("mae", "rmse", "r2")):
        axis.boxplot(results[metric], tick_labels=[metric.upper()])
        axis.set_title(f"{metric.upper()} across outer splits")
        axis.set_ylabel("Score")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Nested repeated-holdout evaluation: outer metric distributions")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _save_metric_trends(results: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for axis, metric in zip(axes, ("mae", "rmse", "r2")):
        axis.plot(results["seed"], results[metric], marker="o", linewidth=1)
        axis.set_ylabel(metric.upper())
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Outer split random seed")
    figure.suptitle("Nested repeated-holdout metrics by outer split")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--random-state-start", type=int, default=0)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    run_robustness(
        data_path=args.data,
        output_dir=args.output_dir,
        n_splits=args.n_splits,
        random_state_start=args.random_state_start,
        inner_splits=args.inner_splits,
        make_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
