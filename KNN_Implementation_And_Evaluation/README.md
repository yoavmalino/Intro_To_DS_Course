# Supervised KNN & Cross-Validation Evaluation

Object-oriented K-Nearest Neighbors (KNN) implementation from scratch for binary classification and continuous regression, evaluated via 5-fold cross-validation on `london_sample_2500.csv`.

## Folder Structure

```text
KNN_Implementation_And_Evaluation/
├── README.md
├── main.py                  # Runs Classification and Regression CV pipelines
├── data.py                  # Data loading, label encoding, StandardScaler, and fold splits
├── knn.py                   # Abstract KNN base class, ClassificationKNN, and RegressionKNN
├── evaluation.py            # F1-score, RMSE, and validation curve plotting
├── cross_validation.py      # K-Fold cross-validation scoring loop
├── london_sample_2500.csv   # 2,500-record dataset sample
└── plots.pdf                # Combined validation performance curves
```

## Architecture & Implementation

* **Preprocessing & Scaling (`data.py`):** Implements `StandardScaler` (`fit`, `transform`, `fit_transform` with `ddof=1`), binary season encoding (`0`: spring/summer, `1`: fall/winter), $\mathcal{N}(0, 0.0001^2)$ feature noise, and deterministic 5-fold splits (`KFold`, `random_state=42`).
* **Object-Oriented KNN (`knn.py`):**
  * `KNN(ABC)`: Fits an internal `StandardScaler` exclusively on `X_train` to prevent data leakage and retrieves $k$ nearest neighbors via Euclidean distance (`np.linalg.norm`).
  * `ClassificationKNN` / `RegressionKNN`: Inherit from `KNN` and predict via majority vote (`statistics.mode`) and neighbor mean (`np.mean`), respectively.
* **Cross-Validation & Metrics (`evaluation.py`, `cross_validation.py`, `main.py`):** Evaluates $k \in \{3, 5, 11, 25, 51, 75, 101\}$ across 5 folds using **F1-Score** for season classification (`['t1', 't2', 'wind_speed', 'hum']`) and **RMSE** for humidity regression (`['t1', 't2', 'wind_speed']`).

## Usage

```bash
conda activate hw2_ds_env
python main.py
```

* **Console Output:** Mean and sample standard deviation (`ddof=1`) of CV scores per $k$ for both tasks (4-decimal precision).
* **Generated Files:** `classification_plot.png` and `regression_plot.png`.
