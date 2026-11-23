import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
np.random.seed(42)


def load_data(path):
    """
    reads and returns the pandas DataFrame
    :param path: this is the path to the csv file
    :return: dataframe
    """
    df = pd.read_csv(path)
    return df


def adjust_labels(y):
    """
    adjust labels of season from {0,1,2,3} to {0,1}
    :param y: label vector
    :return: adjusted label vector
    """
    season_number = {
        0: 0,
        1: 0,
        2: 1,
        3: 1,
    }
    y_new = y.copy()
    y_new = y_new.apply(lambda x: season_number[x])
    return y_new


def add_noise(data):
    """
    :param data: dataset as np.array of shape (n, d) with n observations and d features
    :return: data + noise, where noise~N(0,0.0001^2)
    """
    noise = np.random.normal(loc=0, scale=0.0001, size=data.shape)
    return data + noise


def get_folds():
    """
    :return: sklearn KFold object that defines a specific partition to folds
    """
    return KFold(n_splits=5, shuffle=True, random_state=42)


class StandardScaler:
    def __init__(self):
        """
        object instantiation
        :param: self
        """
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        """
        fit scaler by learning the mean and standard deviation per feature
        :param X: np array of shape (n, d) with n observations and d features
        :param: self
        :return: self
        """
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0, ddof=1)
        return self

    def transform(self, X):
        """
        transform X by learned mean and standard deviation, and return it
        :param X: np array of shape (n, d) with n observations and d features
        :param: self
        :return: transformed dataframe according to mean and standard deviation
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted yet.")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        """
        fit scaler by learning the mean and standard deviation per feature, and then transform X
        :param X: np array of shape (n, d) with n observations and d features
        :param: self
        :return: fitted and transformed dataframe
        """

        return self.fit(X).transform(X)
