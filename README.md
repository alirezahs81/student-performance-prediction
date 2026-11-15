# Student Performance Prediction

A reproducible regression project that estimates a student's **Performance Index** from five study, academic-history, and lifestyle features. It includes a leakage-conscious scikit-learn pipeline, model comparison with cross-validation, a saved inference pipeline, a Streamlit interface, tests, and GitHub Actions CI.

> **Scope and responsible use:** This is a prediction exercise on a single tabular dataset, not a validated educational assessment. The data's provenance and collection methodology were not included in the supplied archive. Do not use predictions for admissions, grading, discipline, or other high-stakes decisions.

## Problem and dataset

The supplied `data/Student_Performance.csv` contains 10,000 rows and six columns:

| Feature | Description | Type |
|---|---|---|
| `Hours Studied` | Study hours (observed range: 1–9) | Numeric |
| `Previous Scores` | Previous academic score (40–99) | Numeric |
| `Extracurricular Activities` | Participation (`Yes` / `No`) | Categorical |
| `Sleep Hours` | Sleep hours (4–9) | Numeric |
| `Sample Question Papers Practiced` | Number practiced (0–9) | Numeric |
| `Performance Index` | Continuous outcome to predict (10–100) | Target |

There are no missing values in the supplied file. There are **127 exact duplicate rows**; exact duplicates are removed before splitting, leaving **9,873 unique rows**. No records are removed based on an outlier rule: the 1.5×IQR check flags zero values in the numeric columns, and automatic deletion would not be justified.

### EDA highlights

- `Previous Scores` has the strongest Pearson correlation with the target (**r = 0.915**).
- `Hours Studied` has a positive moderate correlation (**r = 0.374**).
- `Sleep Hours` (**r = 0.048**) and `Sample Question Papers Practiced` (**r = 0.043**) have weak linear correlations in this dataset.
- The extracurricular feature is nearly balanced: 4,948 `Yes` and 5,052 `No` observations.
- Correlation is descriptive, not causal. The dataset documentation supplied with the archive does not explain how the Performance Index was constructed, so possible target-generation relationships cannot be independently ruled out.

See `reports/eda_summary.json` and `notebooks/eda_and_model_analysis.ipynb` for reproducible analysis.

## Methodology

1. Exact duplicate rows are removed before splitting. This avoids identical records appearing in both train and test sets.
2. A single reproducible 80/20 holdout split is made (`random_state=42`).
3. Preprocessing is encapsulated inside an sklearn `Pipeline` and fitted only on each training fold: median imputation + standard scaling for numeric columns, and most-frequent imputation + one-hot encoding for the categorical column.
4. Seven regressors are compared using 5-fold shuffled `KFold` cross-validation on the training partition.
5. The model with the lowest mean CV RMSE is selected. The untouched holdout set is used once for final reporting.
6. The complete preprocessing + estimator pipeline is saved as `models/best_pipeline.joblib`.

No feature selection or target-informed preprocessing is performed. No outlier removal is applied. The model is a statistical association model; feature importance or correlations should not be interpreted causally.

## Model comparison

Metrics below were computed by running `python -m src.train` on the supplied dataset in Python 3.13.5, pandas 2.2.3, NumPy 2.3.5, and scikit-learn 1.8.0. CV MAE and CV R² are arithmetic means across five folds. CV RMSE is calculated as the square root of the mean of the five fold MSE values (not the arithmetic mean of fold-level RMSE), all on the training partition. Test metrics are from the 20% holdout partition.

| Model | CV MAE | CV RMSE | CV R² | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|---:|---:|
| Linear Regression | 1.6185 | **2.0377** | 0.9887 | 1.6470 | **2.0751** | **0.9884** |
| Lasso | 1.6185 | 2.0377 | 0.9887 | 1.6471 | 2.0752 | 0.9884 |
| Ridge | 1.6185 | 2.0377 | 0.9887 | 1.6470 | 2.0751 | 0.9884 |
| Gradient Boosting | 1.6877 | 2.1249 | 0.9877 | 1.6958 | 2.1391 | 0.9877 |
| HistGradientBoosting | 1.6886 | 2.1266 | 0.9877 | 1.7084 | 2.1551 | 0.9875 |
| Random Forest | 1.8142 | 2.2744 | 0.9859 | 1.8399 | 2.3016 | 0.9858 |
| Extra Trees | 1.8738 | 2.3555 | 0.9849 | 1.9044 | 2.3717 | 0.9849 |

**Selected model: Linear Regression.** It has the lowest mean CV RMSE (2.037735) among the tested candidates; differences from Ridge and Lasso are negligible. Holdout metrics: **MAE 1.6470, MSE 4.3059, RMSE 2.0751, R² 0.9884**. Results are not evidence of external validity; the high R² should be interpreted in light of the undocumented dataset-generation process.

Full precision results are in `reports/model_comparison.csv`; model selection metadata is in `models/metadata.json`.

## Repository layout

```text
student-performance-prediction/
├── .github/workflows/ci.yml
├── data/
│   └── Student_Performance.csv
├── models/
│   ├── best_pipeline.joblib
│   └── metadata.json
├── notebooks/
│   └── eda_and_model_analysis.ipynb
├── reports/
│   ├── eda_summary.json
│   └── model_comparison.csv
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── pipeline.py
│   ├── predict.py
│   ├── robustness.py
│   └── train.py
├── tests/
│   ├── conftest.py
│   ├── test_project.py
│   └── test_robustness.py
├── app.py
├── requirements.txt
└── README.md
```

The Churn datasets, churn-specific models, stale NumPy arrays, and Python bytecode from the original archive were excluded because they belong to a separate customer-churn workflow and are not used by this project.

## Setup

Use Python 3.11–3.13. From the repository root:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

The serialized pipeline was created with scikit-learn 1.8.0. Keep the pinned scikit-learn version when loading the included artifact; sklearn does not guarantee pickle/joblib compatibility across arbitrary versions. For a different environment, retrain the model there.

## Train and evaluate

```bash
python -m src.train
```

This command re-creates `models/best_pipeline.joblib`, `models/metadata.json`, and `reports/model_comparison.csv` from the included CSV.

## Robustness evaluation

Run a repeated holdout evaluation of the selected Linear Regression pipeline:

```bash
python -m src.robustness
```

The command uses 30 repeated 80/20 random splits (seeds 0–29), removes exact duplicate rows before splitting, and fits a fresh preprocessing + regression pipeline on each training partition. It reports MAE, RMSE, and R² for every split, plus mean, sample standard deviation, minimum, and maximum. Outputs are written to `reports/robustness_seed_results.csv`, `reports/robustness_summary.json`, `reports/robustness_metric_distributions.png`, and `reports/robustness_metrics_by_seed.png`. Use `--n-splits` and `--random-state-start` to change the number and range of seeds; use `--no-plots` to skip figures. This is a stability check within the supplied dataset, not external validation. **Important evaluation caveat:** Linear Regression was selected earlier using cross-validation on the project's original training partition. The repeated holdout splits below are then drawn from the full deduplicated dataset, so some observations appearing in a repeated holdout test partition may have participated in that earlier model-selection process. The estimator is held fixed during this robustness run, and each split's preprocessing and fitting use only its training partition; however, these repeated-holdout metrics should be interpreted as conditional stability diagnostics, not as a fully independent, unbiased estimate of generalization after model selection.

## Predict from Python

```python
from src.predict import predict

student = {
    "Hours Studied": 6,
    "Previous Scores": 75,
    "Extracurricular Activities": "Yes",
    "Sleep Hours": 7,
    "Sample Question Papers Practiced": 4,
}
print(predict(student)[0])
```

Input must contain exactly the five feature names shown above. The inference function validates the schema, requires `Yes`/`No` for the categorical feature, rejects non-finite/non-numeric values, and restricts numeric inputs to the observed inclusive ranges in the supplied training dataset. The Streamlit sliders read these ranges from the dataset rather than duplicating hard-coded bounds.

## Run the Streamlit app

```bash
streamlit run app.py
```

The app uses the same saved pipeline and feature schema as the Python prediction function. Train first if the model artifact is absent.

## Tests and CI

```bash
pytest -q
```

GitHub Actions runs the test suite on Python 3.11, 3.12, and 3.13 for pushes, pull requests, and manual dispatch. CI is configured in `.github/workflows/ci.yml` and runs on GitHub Actions for pushes, pull requests, and manual dispatch. The workflow also runs the 30-split robustness evaluation on Python 3.12 and uploads its reports as a build artifact.

## Limitations

- Dataset provenance, sampling design, and target construction are not documented in the provided archive.
- No external validation dataset was supplied.
- A random holdout and cross-validation estimate performance only within this dataset's distribution.
- The app and model are demonstrative, not a substitute for educator judgment or validated psychometric assessment.
