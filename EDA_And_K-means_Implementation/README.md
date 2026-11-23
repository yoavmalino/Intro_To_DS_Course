# Exploratory Data Analysis & K-Means Clustering

End-to-end data analysis and unsupervised clustering pipeline on the London Bike Sharing dataset (`london.csv`), combining Pandas feature engineering with a NumPy implementation of K-Means from scratch.

## Folder Structure

```text
EDA_K-means_Implementation/
├── README.md
├── main.py          # Orchestrates Part A (EDA) and Part B (K-Means)
├── data.py          # Data loading, feature engineering, and statistical analysis
├── clustering.py    # Min-Max scaling, K-Means algorithm, and cluster plotting
├── london.csv       # Hourly bike-sharing dataset (~17,400 records)
└── plots.pdf        # Combined cluster visualizations for k = 2, 3, 5
```

## Architecture & Implementation

* **Feature Engineering (`data.py`):** Extracts integer `Hour`, `Day`, `Month`, and `Year` from `timestamp`; maps meteorological seasons (`season_name`), combined holiday/weekend states (`is_weekend_holiday = 2 * is_holiday + is_weekend + 1`), and perceived temperature difference (`t_diff = t2 - t1`).
* **Statistical & Correlation Analysis (`data.py`):** Generates descriptive statistics (`describe`), ranks the top 5 strongest and weakest non-zero Pearson correlation pairs (6-decimal precision), and aggregates mean `t_diff` by season via `groupby`.
* **Preprocessing & K-Means (`clustering.py`):** Normalizes `cnt` and `t1` to $[0, 1]$ via Min-Max scaling and adds $\mathcal{N}(0, (10^{-5})^2)$ Gaussian noise. Initializes $k$ centroids randomly (`np.random.seed(2)`), assigns points via Euclidean distance, and updates centroids until convergence (`np.array_equal`).
* **Visualization (`clustering.py`):** Plots scaled `(cnt, t1)` clusters with centroid markers (`*`) for $k \in \{2, 3, 5\}$.

## Usage

```bash
conda activate hw2_ds_env
python main.py
```

* **Console Output:** Summary statistics, correlation rankings, seasonal `t_diff` averages, and converged centroids for $k = 2, 3, 5$.
* **Generated Files:** `k means 2.pdf`, `k means 3.pdf`, and `k means 5.pdf`.
