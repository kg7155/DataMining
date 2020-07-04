import numpy as np
import pandas as pd
from scipy import io
from sklearn import datasets
import sklearn.utils
from sklearn.cluster import KMeans
from sklearn.cluster.bicluster import SpectralBiclustering
from itertools import permutations
import math
np.seterr(divide='ignore', invalid='ignore')


class ConsensusClustering:
    """Consensus Clustering based on Monti et al. article from 2003.
    """

    def __init__(self, num_clusters, num_iter, sample_size):
        """Initialize ConsensusClustering.

        Keyword arguments:
        num_cluster -- number of clusters to try with K = {k_1, k_2,... k_max}
        num_iter -- number of resampling iterations
        sample_size -- number of samples to keep in resampling
        """
        self.num_clusters = num_clusters
        self.num_iter = num_iter
        self.sample_size = sample_size

    def fit_predict(self, D):
        """Run ConsensusClustering algorithm on data D.
        Return partition of input data and consensus matrix for best k.
        """
        # number of samples
        n = D.shape[0]

        # AUC score for each k
        AUC_scores = np.zeros(len(self.num_clusters))
        i = 0

        for k in self.num_clusters:
            M = self.calc_consensus(n, D, k)
            AUC_scores[i] = self.calc_auc(M)
            i = i + 1

        # find best number of clusters (k_best)
        idx_k_best = np.argmax(AUC_scores)
        k_best = K[idx_k_best]

        # uncomment to see the best k for given input data
        #print("Best number of clusters (k): ", k_best)

        M_k_best = self.calc_consensus(n, D, k_best)

        # partition D into K-best clusters based on M_k_best using
        # SpectralBiclustering
        model = SpectralBiclustering(n_clusters=k_best, method='bistochastic')
        model.fit(M_k_best)
        P = model.row_labels_

        return P, M_k_best

    def calc_consensus(self, n, D, k):
        """Calculate and return consensus matrix for given k (number of clusters).
        """
        M = np.zeros((n, n), dtype=np.float32)
        I = np.zeros((n, n), dtype=np.uint8)
        for h in range(self.num_iter):
            D_h, indices = self.resample(D)
            y = self.cluster(D_h, k)
            M_h = self.calc_connectivity(n, indices, y)
            I_h = self.calc_indicator(n, indices)
            M = np.add(M, M_h, dtype=np.float32)
            I = np.add(I, I_h)
        M = np.divide(M, I)
        M[np.isnan(M)] = 0
        return M

    def resample(self, D):
        """Resample input data.
        Return array of samples and corresponding indices from D.
        """
        idx = list(range(D.shape[0]))
        D_h, indices = sklearn.utils.resample(
            D, idx, replace=False, n_samples=self.sample_size)
        return D_h, indices

    def cluster(self, D_h, k):
        """Run K-means algorithm on data D_h using k clusters.
        Return predicted cluster index for each sample.
        """
        model = KMeans(n_clusters=k)
        model.fit(D_h)
        y = model.predict(D_h)
        return y

    def calc_connectivity(self, n, indices, y):
        """Calculate and return connectivity matrix filled with ones
        if samples are in the same cluster and zeros otherwise.
        """
        M_h = np.zeros((n, n))
        for i in range(len(indices)):
            for j in range(len(indices)):
                if (y[i] == y[j]):
                    M_h[indices[i]][indices[j]] = 1
        return M_h

    def calc_indicator(self, n, indices):
        """Calculate and return indicator matrix filled with ones
        if samples are in resampled data D_h and zeros otherwise.
        """
        I_h = np.zeros((n, n))
        for i in indices:
            for j in indices:
                I_h[i][j] = 1
        return I_h

    def calc_auc(self, M):
        """Calculate and return AUC (Area Under Curve) from CDF for matrix M.
        """
        N = M.shape[0]
        CDFs, unique_vals = self.calc_cdf(M)
        auc = 0
        for i in range(1, len(CDFs)):
            diff = unique_vals[i] - unique_vals[i - 1]
            auc = auc + diff * CDFs[i]
        return auc

    def calc_cdf(self, M):
        """Calculate and return CDF (empirical Cummulative Distribution)
        and unique values for matrix M.
        """
        N = M.shape[0]
        num_unique_vals = int(N * (N - 1) / 2)
        CDFs = np.zeros(num_unique_vals)
        unique_vals = np.zeros(num_unique_vals)
        val_num = 0
        for i in range(0, N):
            for j in range(i + 1, N):
                CDFs[val_num] = np.sum(np.sum(M <= M[i][j])) / num_unique_vals
                unique_vals[val_num] = M[i][j]
                val_num = val_num + 1
        return CDFs, unique_vals


class KMeansClustering:
    """Cluster input data using K-Means Clustering.
    """

    def __init__(self, num_clusters):
        """Initialize KMeansClustering.

        Keyword arguments:
        num_clusters -- number of clusters
        """
        self.num_clusters = num_clusters

    def fit_predict(self, X):
        """Run K-means algorithm on data X using k clusters.
        Return predicted cluster index for each sample.
        """
        y = self.cluster(X)
        return y

    def cluster(self, X):
        """Run K-Means Clustering on data X.
        Return predicted cluster index for each sample.
        """
        model = KMeans(n_clusters=self.num_clusters)
        model.fit(X)
        y = model.predict(X)
        return y


def preprocess(X_path):
    """Preprocess input data X_path (path to the file containing data).
    """
    # Load data
    X = io.mmread(X_path)
    X = X.tocsc()

    # Select attributes (columns) with most different values
    n = X.shape[1]
    attrs = np.zeros(X.shape[1])
    for i in range(n):
        attrs[i] = len(np.unique(X[:, i].toarray()))

    # Take only 550 columns with most different features
    top_features = np.argsort(attrs)[::-1][:550]
    X_p = X[:, top_features]

    # Convert to dense form for K-means
    X_p = X_p.toarray()

    # Binarize
    X_p = X_p != 0

    return X_p


if __name__ == '__main__':

    """ Iris data
    X, y = datasets.load_iris(return_X_y=True)
    K = np.array([2,4,3,5,10,120])
    cc = ConsensusClustering(num_clusters=K, num_iter=10, sample_size=int(0.8*X.shape[0]))
    P, M_k_best = cc.fit_predict(X)
    """

    """ Genom data
    """
    X = preprocess("data/train.mtx")
    K = [43]
    kc = KMeansClustering(num_clusters=43)
    y = kc.fit_predict(X)
    #cc = ConsensusClustering(num_clusters=K, num_iter=10, sample_size=int(0.8*X.shape[0]), n_jobs=-2)
    #y, _ = cc.fit_predict(X)
    pd.DataFrame(y).to_csv("k_means.txt", index=False, header=False)
