# Predictive Analysis Coursework

A collection of applied predictive analysis notebooks covering exploratory data analysis (EDA), linear regression, logistic regression with hyperparameter tuning, and gradient-boosted time series forecasting. Each notebook works through a complete workflow, from loading and cleaning data to training, evaluating, and interpreting models.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)

## Table of Contents

- [Overview](#overview)
- [Notebooks](#notebooks)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Key Results](#key-results)
- [Getting Started](#getting-started)
- [Tech Stack](#tech-stack)
- [Lessons Learned](#lessons-learned)
- [License](#license)
- [Contact](#contact)

## Overview

This repository contains the notebooks I developed for a predictive analysis course. Together they cover the full modeling pipeline:

1. **Exploratory data analysis** on a customer churn dataset.
2. **Linear regression** to predict athlete body weight from Olympic data.
3. **Logistic regression** for customer churn classification, with tuning for different evaluation priorities.
4. **Time series forecasting** of store sales using tree-based boosting models, benchmarked against a naive baseline.

A recurring theme is avoiding **data leakage**: every notebook separates training and test data before preprocessing, and the preprocessing steps are packaged in `Pipeline` and `ColumnTransformer` objects so they are fit only on training data.

## Notebooks

### 1. Telco Customer Churn: Exploratory Data Analysis (`Taller_EDA_resuelto-2.ipynb`)

A structured EDA of the IBM Telco Customer Churn dataset (7,043 customers, 33 columns).

- **Data loading:** Found that `Total Charges` contained blank spaces instead of numbers, which pandas did not treat as nulls. Fixed it at load time with `na_values=[' ']`.
- **Variable pruning:** Dropped constant columns (`Count`, `Country`, `State`), the identifier `CustomerID` (set as the index instead), and `City` and `Lat Long` as required.
- **Missing values:** Imputed the 11 missing `Total Charges` values with `0`, since they belong exactly to customers with `Tenure Months = 0`. Left the 5,174 missing `Churn Reason` values untouched, because they correspond to customers who did not churn, and imputing them would invent information.
- **Quality checks:** Verified duplicates, categorical label consistency, and the relationship between `Churn Label` and `Churn Value`.

### 2. Athlete Weight Prediction: Linear Regression (`AP_Evaluación_1.ipynb`)

An evaluation that predicts athlete `Weight` from the Olympic `athlete_events.csv` dataset.

- **Cleaning:** Removed duplicates. Filled missing `Medal` values with `"No medal"`, and imputed `Age`, `Height`, and `Weight` using group medians by sport and sex.
- **Feature selection:** Dropped `ID`, `Games`, `Team`, `Name`, and `Event` with justifications for each (primary key, redundancy, high cardinality, or unstandardized values).
- **Variable analysis:** Classified categorical variables as nominal, ordinal, or high-cardinality, and numerical variables by distribution and outliers.
- **Modeling:** Built a `Pipeline` with `ColumnTransformer` that standardizes numerical features and one-hot encodes categorical ones (using `max_categories` and `min_frequency` to limit high-cardinality columns). Trained a `LinearRegression` model on an 80/20 split with `random_state=1`, and then added features in a second version.

### 3. Churn Classification: Logistic Regression (`Taller_regresión_logística.ipynb`)

A tuning workshop that builds a logistic regression model to predict `Churn Label` using eight features: Contract, Dependents, Online Security, Tenure Months, Senior Citizen, Payment Method, Internet Service, and Monthly Charges.

- **Model 1:** Tune `C` and `l1_ratio` with a metric that prioritizes correctly identifying customers who churn, then address class imbalance if the model is unbalanced.
- **Model 2:** Repeat the tuning with a metric that prioritizes correctly identifying customers who stay, then compare coefficients and the top five most important features.

> **Status:** This notebook currently contains the assignment instructions. The code cells are empty, so it works as a template for the exercise.

### 4. Store Sales Forecasting: Boosting Models (`examen_final.ipynb`)

A final exam that forecasts daily sales for **Store 3** using Gradient Boosting, XGBoost, and LightGBM, optimized for Mean Absolute Error (MAE).

- **Feature engineering:** Extracted calendar features (`month`, `day`, `dayofweek`, `year`) from the date, and built lag features (`lag_1`, `lag_7`) and a 7-day rolling mean.
- **Validation:** Used `TimeSeriesSplit` with 5 folds instead of `KFold`, which would leak future data into training. Held out the last 60 days as a chronological test set.
- **Tuning:** `GridSearchCV` over `learning_rate`, `n_estimators`, and `max_depth`.
- **Benchmark:** Compared all models against a naive forecast that predicts yesterday's sales (`t-1`).
- **Analysis:** Explained why tree-based models cannot extrapolate trends unless the trend is captured through engineered features, and reviewed feature importance.

## Key Results

### Store 3 Forecasting (Test Set: 60 days, Nov 2 to Dec 31, 2023)

| Model | MAE (Cross-Validation) | MAE (Test) | Improvement vs. Naive |
|-------|------------------------|-----------|----------------------|
| Naive baseline (t-1) | N/A | 96.00 units | 0.00% |
| Gradient Boosting (best) | 32.58 units | **27.07 units** | **71.80%** |
| XGBoost | 33.12 units | 28.45 units | 70.36% |
| LightGBM | 34.05 units | 29.10 units | 69.69% |

**Best hyperparameters (Gradient Boosting):** `learning_rate=0.1`, `n_estimators=150`, `max_depth=3`

**Most important features:** `dayofweek` (36.0%), followed by `is_holiday` (22.4%). Weekend sales run noticeably higher than weekday sales, and holidays produce immediate demand spikes.

> These figures come from the final exam notebook's summary. Re-run the notebook to confirm them.

### Athlete Weight Prediction and Churn Classification

Results for these notebooks are documented in each notebook's analysis cells. Open the notebooks to review the R² scores, confusion matrices, and classification reports.

## Project Structure

```
predictive-analysis/
├── Taller_EDA_resuelto-2.ipynb        # EDA: Telco customer churn
├── AP_Evaluación_1.ipynb              # Linear regression: athlete weight
├── Taller_regresión_logística.ipynb   # Logistic regression workshop (template)
├── examen_final.ipynb                 # Store sales forecasting with boosting
├── data/                              # Datasets (not included, see below)
│   ├── Telco_customer_churn.csv
│   ├── athlete_events.csv
│   └── [store sales dataset].csv
└── README.md
```

## Datasets

| Dataset | Used In | Source |
|---------|---------|--------|
| Telco Customer Churn | EDA, Logistic Regression | [IBM Community](https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113) |
| 120 Years of Olympic History (`athlete_events.csv`) | Linear Regression | [Kaggle](https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results) |
| [Store sales dataset name] | Forecasting | [Source] |

The datasets are not included in the repository. Download them and place them in the `data/` folder, then update the file paths in the notebooks if needed.

## Getting Started

### Prerequisites

- Python 3.10 or higher (3.12 was used for development)
- Jupyter Notebook or JupyterLab

### Installation

```bash
git clone https://github.com/andresmendoza59/predictive-analysis.git
cd predictive-analysis

python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows

pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm jupyter
```

### Running the Notebooks

```bash
jupyter notebook
```

Run the notebooks in this order, since each one is self-contained: run all cells from top to bottom, using **Restart & Run All** to confirm they execute without errors.

## Tech Stack

- **Language:** Python
- **Data manipulation:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine learning:** scikit-learn (Pipeline, ColumnTransformer, GridSearchCV, TimeSeriesSplit), XGBoost, LightGBM
- **Environment:** Jupyter Notebook

## Lessons Learned

- **Pipelines prevent leakage.** Fitting imputers and encoders inside a `Pipeline` keeps test-set statistics out of training.
- **Time series need time-aware validation.** Standard K-fold shuffles the data, which lets the model see the future. `TimeSeriesSplit` and a chronological test set give honest error estimates.
- **Missing values need a reason.** Imputing `Total Charges` with `0` made sense only after checking that the missing rows were new customers. Imputing `Churn Reason` would have been wrong.
- **Metric choice shapes the model.** Tuning for recall on churners and tuning for recall on non-churners lead to different models, so the business goal should drive the metric.
- **Baselines matter.** The naive `t-1` forecast gave a reference point that showed the boosting models' real improvement.

## Future Work

- [ ] Add the logistic regression workshop solutions
- [ ] Add class-weight balancing and compare results with and without it
- [ ] Try SHAP values for more reliable feature explanations
- [ ] Add a requirements.txt with pinned versions
- [ ] Export the best forecasting model for deployment

## License

This project is licensed under the [MIT License](LICENSE). Coursework datasets remain subject to their original licenses.
