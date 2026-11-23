import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(2)

def add_noise(data):
    """
    :param data: dataset as numpy array of shape (n, 2)
    :return: data + noise, where noise~N(0,0.00001^2)
    """
    noise = np.random.normal(loc=0, scale=1e-5, size=data.shape)
    return data + noise

def choose_initial_centroids(data, k):
    """
    :param data: dataset as numpy array of shape (n, 2)
    :param k: number of clusters
    :return: numpy array of k random items from dataset
    """
    n = data.shape[0]
    indices = np.random.choice(range(n), k, replace=False)
    return data[indices]

def transform_data(df, features):
    """
    Performs the following transformations on df:
        - selecting relevant features
        - scaling
        - adding noise
    :param df: dataframe as was read from the original csv.
    :param features: list of 2 features from the dataframe
    :return: transformed data as numpy array of shape (n, 2)
    """
    x = df[features].to_numpy()

    # Min-Max Scaling
    min_vals = np.min(x, axis=0)  # Find the minimum value for each feature
    max_vals = np.max(x, axis=0)  # Find the maximum value for each feature
    x_scaled = (x - min_vals) / (max_vals - min_vals)  # Apply the scaling formula

    x_scaled_with_noise = add_noise(x_scaled)
    return x_scaled_with_noise

def kmeans(data, k):
    """
    Running kmeans clustering algorithm.
    :param data: numpy array of shape (n, 2)
    :param k: desired number of cluster
    :return:
    * labels - numpy array of size n, where each entry is the predicted label (cluster number)
    * centroids - numpy array of shape (k, 2), centroid for each cluster.
    """

    centroids = choose_initial_centroids(data, k)
    prev_centroids = np.zeros(centroids.shape)
    while True:
        labels = assign_to_clusters(data, centroids)
        centroids = recompute_centroids(data, labels, k)
        if np.array_equal(prev_centroids, centroids):
            break
        prev_centroids = centroids.copy()

    return labels, centroids

def visualize_results(data, labels, centroids, path):
    """
    Visualizing results of the kmeans model, and saving the figure.
    :param data: data as numpy array of shape (n, 2)
    :param labels: the final labels of kmeans, as numpy array of size n
    :param centroids: the final centroids of kmeans, as numpy array of shape (k, 2)
    :param path: path to save the figure to.
    """
    k = centroids.shape[0]  # number of clusters

    plt.scatter(data[:, 0], data[:, 1], c=labels, s=30, cmap='Set3')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='blue', s=30, marker='*')
    plt.title(f"Results for kmeans with k = {k}")
    plt.xlabel("cnt")
    plt.ylabel("t1")
    plt.savefig(path)
    plt.close('all')

def dist(x, y):
    """
    Euclidean distance between vectors x, y
    :param x: numpy array of size n
    :param y: numpy array of size n
    :return: the euclidean distance
    """
    dist_sum = 0
    for i in range(len(x)):
        dist_sum += (x[i] - y[i]) ** 2
    return np.sqrt(dist_sum)

def assign_to_clusters(data, centroids):
    """
    Assign each data point to a cluster based on current centroids
    :param data: data as numpy array of shape (n, 2)
    :param centroids: current centroids as numpy array of shape (k, 2)
    :return: numpy array of size n
    """
    n_samples = data.shape[0]
    n_clusters = len(centroids)
    labels = np.zeros(n_samples, dtype = 'int')
    for i in range(n_samples):
        distances = np.array([dist(data[i], centroids[j]) for j in range(n_clusters)])
        closest_centroid_index = np.argmin(distances)
        labels[i] = closest_centroid_index
    return labels

def recompute_centroids(data, labels, k):
    """
    Recomputes new centroids based on the current assignment
    :param data: data as numpy array of shape (n, 2)
    :param labels: current assignments to clusters for each data point, as numpy array of size n
    :param k: number of clusters
    :return: numpy array of shape (k, 2)
    """
    centroids = np.zeros((k, data.shape[1]))
    for j in range(k):
        cluster_points = data[labels == j]
        centroids[j] = np.mean(cluster_points, axis=0)
    return centroids