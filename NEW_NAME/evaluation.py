import numpy as np
import matplotlib.pyplot as plt


def f1_score(y_true, y_pred):
    """
    returns f1_score of binary classification task with true labels y_true and predicted labels y_pred
    :param y_true: np array of true labels
    :param: y_pred: np array of predicted labels according to the model
    :return: f1 score
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def rmse(y_true, y_pred):
    """
    returns RMSE of regression task with true labels y_true and predicted labels y_pred
    :param y_true: np array of true labels
    :param y_pred: np array of predicted labels according to the model
    :return: RMSE value
    """
    rmse_formula = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse_formula


def visualize_results(k_list, scores, metric, title, path):
    """
    plot a results graph of cross validation scores
    :param k_list: list of k values
    :param scores: list of scores
    :param metric: name of metric
    :param title: title of graph
    :param path: path to save the graph
    """
    plt.figure()
    plt.plot(k_list, scores)

    plt.title(f" {title}")
    plt.xlabel("k")
    plt.ylabel(f" {metric}")
    plt.savefig(path)
    plt.close()
