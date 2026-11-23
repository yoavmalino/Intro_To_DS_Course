import numpy as np
from data import load_data, adjust_labels, add_noise, get_folds
from knn import ClassificationKNN, RegressionKNN
from evaluation import f1_score, rmse, visualize_results
from cross_validation import cross_validation_scores


def main():
    csv_path = "london_sample_2500.csv"
    k_values = [3, 5, 11, 25, 51, 75, 101]
    folds = get_folds()
    df = load_data(csv_path)

    x_class = df[['t1', 't2', 'wind_speed', 'hum']].values
    y_class = adjust_labels(df['season'])
    x_class = add_noise(x_class)
    f1_scores_mean = []

    print("Part1 - Classification")
    for k in k_values:
        model = ClassificationKNN(k)
        scores = cross_validation_scores(model, x_class, y_class.values, folds, f1_score)
        mean_score = np.mean(scores)
        std_score = np.std(scores, ddof=1)
        f1_scores_mean.append(mean_score)
        print(f"k={k}, mean score: {mean_score:.4f}, std of scores: {std_score:.4f}")

    visualize_results(
        k_list=k_values,
        scores=f1_scores_mean,
        metric="f1 score",
        title="Classification",
        path="classification_plot.png"
    )

    x_reg = df[['t1', 't2', 'wind_speed']].values
    y_reg = df['hum'].values
    x_reg = add_noise(x_reg)
    rmse_scores_mean = []

    print("\nPart2 - Regression")
    for k in k_values:
        model = RegressionKNN(k)
        scores = cross_validation_scores(model, x_reg, y_reg, folds, rmse)
        mean_score = np.mean(scores)
        std_score = np.std(scores, ddof=1)
        rmse_scores_mean.append(mean_score)
        print(f"k={k}, mean score: {mean_score:.4f}, std of scores: {std_score:.4f}")

    visualize_results(
        k_list=k_values,
        scores=rmse_scores_mean,
        metric="RMSE",
        title="Regression",
        path="regression_plot.png"
    )
    print()


if __name__ == "__main__":
    main()
