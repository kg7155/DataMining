import numpy as np
import pandas as pd
import math
from sklearn.neighbors import KDTree
import seaborn as sns
import matplotlib.pyplot as plt


class DBSCAN():
    def __init__(self, min_samples=4, eps=0.1, metric="euclidean"):
        """Initialize DBSCAN.

        Keyword arguments:
        min_samples -- required number of neighbor points for the seed point
        eps -- threshold distance
        metric -- distance measure
        """
        self.min_samples = min_samples
        self.eps = eps
        self.metric = metric

    def fit_predict(self, X):
        """Call dbscan() method.
        Return a vector containing group indices for each data point or -1 if the point is noise.
        """
        labels = self.dbscan(X)
        v = np.ones(len(X))
        for i in range(len(X)):
            v[i] = labels[tuple(X[i])]
        return v

    def dbscan(self, X):
        """Run DBSCAN algorithm on data X.
        Return a dictionary containing data points as keys and corresponding group indices as values.
        """
        c = -1
        labels = self.init_labels(X)
        """Labels meaning:
        -2: point has not been claimed yet
        -1: point is a noise
        0...k: cluster index the point was assigned to
        """
        for x in X:
            if (labels[tuple(x)] != -2):
                continue
            else:
                N = self.find_neighbors(x, X)
                if (len(N) < self.min_samples):
                    labels[tuple(x)] = -1
                else:
                    c = c + 1
                    labels[tuple(x)] = c

                    i = 0
                    while (i < len(N)):
                        n = N[i]
                        if (labels[tuple(X[n])] == -1):
                            labels[tuple(X[n])] = c
                        elif (labels[tuple(X[n])] == -2):
                            labels[tuple(X[n])] = c
                            N_n = self.find_neighbors(X[n], X)
                            if (len(N_n) >= self.min_samples):
                                N = N + N_n
                        i = i + 1
        return labels

    def init_labels(self, X):
        """ Initialize data points for DBSCAN.
        Return a dictionary containing data points as keys and corresponding labels as values.
        """
        labels = {}
        for x in X:
            labels[tuple(x)] = -2
        return labels

    def find_neighbors(self, x, X):
        """Find the x's neighbors (points within eps) using sklearn KDTree.
        Return the x's neighbors indices.
        """
        kdt = KDTree(X, leaf_size=5)
        ind = kdt.query_radius([x], r=self.eps)
        neighs = []
        for i in ind[0]:
            if (tuple(X[i]) != tuple(x)):
                neighs.append(i)
        return neighs


def k_dist(X, k, metric="euclidean"):
    """Calculate the distances to k-th neighbour for each data point from X.
    Return a vector of distances sorted descending
    """
    kdt = KDTree(X, metric=metric)
    all_dist = []
    for x in X:
        dist, _ = kdt.query([x], k=k)
        all_dist.append(dist[0][k - 1])
    return sorted(all_dist, reverse=True)


def draw_clusters(X, clusters):
    """Plot clusters of data points each in its own color; noise is presented with black dots."""
    plt.subplot(2, 1, 2)
    plt.title("DBSCAN | min_samples=4, eps=0.065")
    num_clusters = int(np.max(clusters)) + 1
    for c in range(num_clusters):
        C = X[clusters == c]
        sns.regplot(x=C[:, 0], y=C[:, 1], fit_reg=False, label=str(c))
    C = X[clusters == -1]
    if len(C):
        sns.regplot(x=C[:, 0], y=C[:, 1], fit_reg=False,
                    marker=".", color="k", label="noise")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    """Simple data
    X = [[3,2], [4,2], [5,7], [1,3], [9,2], [5,3], [4,4], [6,8], [5,8], [6,9], [10,1]]
    X = np.asarray(X)
    db = DBSCAN(min_samples=3, eps=3)
    """
    X = pd.read_csv("data/data.csv", header=0)
    X = X.values

    plt.subplot(2, 1, 1)
    plt.title("Distances to k-th neighbor")
    plt.xlabel("points")
    plt.ylabel("distance")
    ks = [4, 8, 12]
    for k in ks:
        v = k_dist(X, k)
        plt.plot(v, label="k=" + str(k))
    plt.legend()

    db = DBSCAN(min_samples=4, eps=0.065)
    clusters = db.fit_predict(X)
    draw_clusters(X, clusters)
