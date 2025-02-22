from time import time
import numpy as np
import pandas as pd
import Orange
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class EigenPCA():
    """ Calculate full PCA transformation by using Numpy's np.linalg.eigh.
    """

    def __init__(self, n_components=2, iterations=1000, eps=1e-10):
        self.n_components = n_components
        self.iterations = iterations
        self.eps = eps

    def fit(self, X):
        """ Fit the model with X.
        Args:
            X (np.ndarray): Data matrix of shape [n_examples, n_features]
        """
        # center data
        self.mean_ = X.mean(axis=0)
        X_c = X - self.mean_

        # calc covariance matrix S
        X_c = X_c.transpose()
        S = np.cov(X_c)
        self.trace_ = np.trace(S)

        # calc eigenvectors (components_) and eigenvalues (explained_variance_)
        self.explained_variance_, self.components_ = np.linalg.eigh(S)

        # sort eigenvectors and eigenvalue in descending order
        self.explained_variance_ = self.explained_variance_[
            ::-1][:self.n_components]
        self.components_ = np.flipud(self.components_.T)[:self.n_components]

        # calc eigenvalues ratio (explained_variance_ratio)
        self.explained_variance_ratio_ = []
        for i in self.explained_variance_:
            self.explained_variance_ratio_.append(i / self.trace_)

    def transform(self, X):
        """ Apply the dimensionality reduction on X.
        Args:
            X (np.ndarray): Data matrix of shape [n_examples, n_features]
        """
        return np.dot(X - self.mean_, self.components_.T)


class PowerPCA():
    """ Calculate PCA transformation by using Power method.

    Parameters
    -----------
    n_components : int > 0
        Number of components for PCA.

    iterations : int >= 0
        Max. number of iterations for the power method.

    eps: float >= 0, (default: 1e-10)


    Attributes
    -----------
    components_ : array, shape (n_components, n_features)
        Principal axes in feature space, representing the directions of
        maximum variance in the data. The components are sorted by
        ``explained_variance_``.

    explained_variance_ : array, shape (n_components,)
        The amount of variance explained by each of the selected components.
        Equal to n_components largest eigenvalues of the covariance matrix of X.

    explained_variance_ratio_ : array, shape (n_components,)
        Percentage of variance explained by each of the selected components.
    """

    def __init__(self, n_components=2, iterations=1000, eps=1e-10):
        self.n_components = n_components
        self.iterations = iterations
        self.eps = eps

    def fit(self, X):
        """ Fit the model with X.
        Args:
            X (np.ndarray): Data matrix of shape [n_examples, n_features]
        """
        # center data
        self.mean_ = np.mean(X, axis=0)
        X = X - self.mean_

        # calc covariance matrix S
        X_t = X.transpose()
        S = np.dot(X_t, X) / X.shape[0]
        self.trace_ = np.trace(S)

        self.components_ = np.random.rand(self.n_components, S.shape[0])
        self.explained_variance_ = np.random.rand(self.n_components)
        self.explained_variance_ratio_ = np.random.rand(self.n_components)

        # calculate PCA components
        for i in range(self.n_components):
            comp_vec = self.find_eigen_vec(S)
            comp_val, comp_var_ratio = self.find_eigen_val(comp_vec, S)

            # save data
            self.components_[i] = comp_vec
            self.explained_variance_[i] = comp_val
            self.explained_variance_ratio_[i] = comp_var_ratio

            S_new = S - comp_val * np.outer(comp_vec, comp_vec.T)
            S = S_new

    def find_eigen_vec(self, S):
        """ Find the eigen vector of the matrix S using the power method.
        """
        a = np.random.rand(S.shape[0])

        i = 0
        while True:
            b = np.dot(S, a)
            #b /= np.sqrt(np.dot(b, b))
            b /= np.linalg.norm(b)
            e = np.sqrt(np.dot(b - a, b - a))

            a = b
            i = i + 1
            if (e < self.eps or i > self.iterations):
                break

        # print(i)
        return a

    def find_eigen_val(self, a, S):
        """ Find the eigenvalue (explained variance) and the explained variance ratio
        of a given eigen vector a and matrix S.
        """
        e_val = np.dot(np.dot(a, S), a)
        e_val_rate = e_val / self.trace_
        return e_val, e_val_rate

    def transform(self, X):
        """Apply dimensionality reduction on X.
        """
        return np.dot(X - self.mean_, self.components_.T)


class OrtoPCA():
    """ Calculate PCA transformation by using Power method with Gramm-Schmidt ortogonalization.

    Parameters
    -----------
    n_components : int > 0
        Number of components for PCA.

    iterations : int >= 0
        Max. number of iterations for the power method.

    eps: float >= 0, (default: 1e-10)

    Attributes
    -----------
    components_ : array, shape (n_components, n_features)
        Principal axes in feature space, representing the directions of
        maximum variance in the data. The components are sorted by
        ``explained_variance_``.

    explained_variance_ : array, shape (n_components,)
        The amount of variance explained by each of the selected components.
        Equal to n_components largest eigenvalues of the covariance matrix of X.

    explained_variance_ratio_ : array, shape (n_components,)
        Percentage of variance explained by each of the selected components.
    """

    def __init__(self, n_components=2, iterations=1000, eps=1e-10):
        self.n_components = n_components
        self.iterations = iterations
        self.eps = eps

    def fit(self, X):
        """ Fit the model with X.
        Args:
            X (np.ndarray): Data matrix of shape [n_examples, n_features]
        """
        # center data
        self.mean_ = np.mean(X, axis=0)
        X = X - self.mean_

        # calc covariance matrix
        X_t = X.transpose()
        S = np.dot(X_t, X) / X.shape[0]
        self.trace_ = np.trace(S)

        # calculate PCA components
        temp = np.random.rand(S.shape[0], self.n_components)

        i = 0
        while True:
            prev = temp.copy()
            temp = S.dot(temp)
            temp = self.gram_schmidt_orthogonalize(temp)
            temp = temp / np.linalg.norm(temp, axis=0)

            e = np.linalg.norm(temp - prev)

            if (e < self.eps or i > self.iterations):
                break

            i = i + 1

        # save data
        self.components_ = temp.T

        self.explained_variance_ = []
        self.explained_variance_ratio_ = []
        for i in self.components_:
            e_var, e_var_rate = self.find_eigen_val(i, S)
            self.explained_variance_.append(e_var)
            self.explained_variance_ratio_.append(e_var_rate)

        self.explained_variance_ = np.asarray(self.explained_variance_)
        self.explained_variance_ratio_ = np.asarray(
            self.explained_variance_ratio_)

    def find_eigen_val(self, a, S):
        """ Find the eigen value (explained variance) and the explained variance ratio
        of a given eigen vector a and matrix S.
        """
        e_val = np.dot(np.dot(a, S), a)
        e_val_rate = e_val / self.trace_
        return e_val, e_val_rate

    def transform(self, X):
        """Apply dimensionality reduction on X.
        """
        return np.dot(X - self.mean_, self.components_.T)

    def column(self, matrix, i):
        """Get the i-th column of the matrix
        """
        return [row[i] for row in matrix]

    def gram_schmidt_orthogonalize(self, vecs):
        """ Gram-Schmidt orthonormalization of column vectors.

        Args:
            vecs (np.asarray): Array of shape [n_features, k] with column
                vectors to orthogonalize.

        Returns:
            Orthonormalized vectors of the same shape as on input.
        """
        u = self.column(vecs, 0)
        e = u / np.linalg.norm(u)

        # length of each vector
        n = vecs.shape[0]

        # number of vectors
        k = vecs.shape[1]

        # matrix that will hold orthonormal vectors
        U = np.zeros((n, k))

        U[:, 0] = self.column(vecs, 0) / np.linalg.norm(self.column(vecs, 0))
        for i in range(1, k):
            U[:, i] = self.column(vecs, i)
            for j in range(0, i):
                U[:, i] = U[:, i] - \
                    np.dot((np.dot(U[:, i], U[:, j]) /
                            np.dot(U[:, j], U[:, j])), U[:, j])

            U[:, i] = U[:, i] / np.linalg.norm(U[:, i])

        return np.asarray(U)


if __name__ == '__main__':
    """
    iris = Orange.data.Table('iris.tab')
    X = iris.X
    train, test = X[:145], X[145:]

    my_pca = OrtoPCA(n_components=2)
    my_pca.fit(train)
    Z = my_pca.transform(test)
    """
    data = pd.read_csv('./data/train.csv', delimiter=',')
    num_classes = len(data.label.unique())
    y = data['label'].values
    X = X = data.drop(['label'], axis=1).values

    pca = PowerPCA(n_components=2)
    pca.fit(X)
    X = pca.transform(X)

    colors_palette = cm.brg(np.linspace(0, 1, num_classes))

    for i in range(0, num_classes):
        plt.scatter(X[y == i, 0], X[y == i, 1],
                    label=i, color=colors_palette[i])

    plt.legend()
    plt.show()
