# Coursework repository for Introduction to Data Science — Spring 2025

This repository contains end-to-end data science pipelines and core machine learning algorithms implemented from scratch in Python using NumPy, Pandas, and Matplotlib.

## Repository Structure

```text
intro-to-data-science/
├── README.md
├── env.yml
├── EDA_And_K-means_Implementation/
│   ├── README.md
│   ├── main.py
│   ├── data.py
│   └── clustering.py
└── KNN_Implementation_And_Evaluation/
    ├── README.md
    ├── main.py
    ├── data.py
    ├── knn.py
    ├── evaluation.py
    └── cross_validation.py
```

## Projects Overview

| Project | Core Concepts | Modules |
| :--- | :--- | :--- |
| **[EDA & K-Means Implementation](./EDA_And_K-means_Implementation)** | Data wrangling, temporal & domain feature engineering, Pearson correlation ranking, Min-Max scaling, and K-Means clustering from scratch | `data.py`, `clustering.py`, `main.py` |
| **[KNN Implementation & Evaluation](./KNN_Implementation_And_Evaluation)** | Object-oriented KNN (`ClassificationKNN`, `RegressionKNN`), custom `StandardScaler`, F1-score, RMSE, and 5-fold Cross-Validation | `data.py`, `knn.py`, `evaluation.py`, `cross_validation.py`, `main.py` |

## Environment Setup

Both projects run in the shared Conda environment (`hw2_ds_env`) defined in `env.yml` (`Python 3.8.8`, `NumPy 1.21.5`, `Pandas 1.3.4`, `Matplotlib 3.5.1`, `scikit-learn 1.0.2`).

1. Create the environment from the repository root:
   ```bash
   conda env create -f env.yml
   ```
2. Activate the environment:
   ```bash
   conda activate hw2_ds_env
   ```

## Dataset

All pipelines use the [London Bike Sharing Dataset](https://www.kaggle.com/hmavrodiev/london-bike-sharing-dataset) (`london.csv` and sampled subsets), containing ~17,400 hourly records of bike rental counts, weather measurements, and calendar indicators collected between January 2015 and January 2017.
