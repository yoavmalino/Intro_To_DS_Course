import numpy as np
from statistics import mode
from abc import abstractmethod, ABC
from data import StandardScaler


class KNN(ABC):

    def __init__(self, k):
        """
        object instantiation, save k and define a scaler object
        :param k: the number of neighbors
        :param self: self
        """
        self.k = k
        self.scaler = StandardScaler()
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """
        fit scaler and save X_train and y_train
        :param X_train: training data
        :param y_train: training labels
        """
        self.X_train = self.scaler.fit_transform(X_train)
        self.y_train = y_train

    @abstractmethod
    def predict(self, X_test):
        """
        predict labels for X_test and return predicted labels
        :param X_test: test data
        :param self: self
        """
        pass

    def neighbours_indices(self, x):
        """
        for a given point x, find indices of k closest points in the training set
        :param x: the point for which we want to find the indices
        :param self: self
        :return: sorted indices of k closest points in the training set
        """
        distances = [self.dist(x, x_train) for x_train in self.X_train]
        sorted_indices = np.argsort(distances)[:self.k]

        return sorted_indices

    @staticmethod
    def dist(x1, x2):
        """
        returns Euclidean distance between x1 and x2
        :param x1: first point
        :param x2: second point
        :return: Euclidean distance between x1 and x2
        """
        return np.linalg.norm(x1 - x2)


class ClassificationKNN(KNN):
    def __init__(self, k):
        """
        object instantiation, parent class instantiation
        :param k: number of neighbors
        :param self: self
        """
        super().__init__(k)

    def predict(self, X_test):
        """
        predict labels for X_test and return predicted labels
        :param X_test: test data
        :param self: self
        :return: predicted labels
        """
        x_test_scaled = self.scaler.transform(X_test)
        predictions = []

        for x in x_test_scaled:
            indices = self.neighbours_indices(x)
            neighbor_labels = [self.y_train[i] for i in indices]
            predictions.append(mode(neighbor_labels))

        return predictions


class RegressionKNN(KNN):
    def __init__(self, k):
        """
        object instantiation, parent class instantiation
        :param k: number of neighbors
        :param self: self
        """
        super().__init__(k)

    def predict(self, X_test):
        """
        predict labels for X_test and return predicted labels
        :param X_test: test data
        :param self: self
        :return: predicted labels
        """
        x_test_scaled = self.scaler.transform(X_test)
        predictions = []

        for x in x_test_scaled:
            indices = self.neighbours_indices(x)
            neighbor_values = [self.y_train[i] for i in indices]
            predictions.append(np.mean(neighbor_values))

        return predictions
