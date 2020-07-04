import glob
import numpy as np
import zlib
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

"""Linear kernel implementation
""" 
def kernel_linear(X, x):
    return x @ X.T

class KernelRBF():
    """RBF (radial basis function) kernel implementation
    """ 

    """Cache values for RBF kernel using X as training data.
    """
    def cache(self, X):
        self.cache_ = dict()
        for i in range(len(X)):
            self.cache_[X[i].tostring()] = self.__call__(X, X[i])

    """Calculate RBF kernel value for x using X (training data).
    """
    def __call__(self, X, x):
        key = x.tostring()
        if key in self.cache_:
            return self.cache_[key]
        else:
            dsts = np.sum((x - X)**2, axis=1)
            return np.exp(-dsts / 2) #for simiplicty assume gamma = 1

class KernelText():
    """Text kernel implementation
    """

    """Cache values for text kernel using X as training data.
    """
    def cache(self, X):
        n = len(X)
        self.cache_idx_ = dict()
        self.cache_ = np.zeros((n, n))

        for i in range(n):
            self.cache_idx_[X[i]] = i
            for j in range(i + 1, n):
                result = self.dist(X[i], X[j])
                self.cache_[i, j] = 10 - result
                self.cache_[j, i] = 10 - result
        self.cache_ -= np.mean(self.cache_) #account for missing bias

    """Calculate compression ratio between 2 strings (a and b).
    """
    def dist(self, a, b):
        A = len(zlib.compress(bytes(a, "utf8")))
        B = len(zlib.compress(bytes(b, "utf8")))
        AB = len(zlib.compress(bytes(a + b, "utf8")))
        BA = len(zlib.compress(bytes(b + a, "utf8")))
        return ((AB - A) / A + (BA - B) / B) / 2

    """Calculate text kernel value for x using X (training data).
    """
    def __call__(self, X, x):
        if x in self.cache_idx_:
            return -self.cache_[self.cache_idx_[x]]
        else:
            res = list()
            for i in X:
                res.append(self.dist(i, x))
            return -np.array(res)
        
class SVM:
    """SVM (Support Vector Machine) implementation
    """

    """Initialize SVM.

    Arguments:
    kernel {string} -- type of kernel to use
    epochs {int} -- max. number of iterations
    rate {float} -- learning rate
    C {int} -- penalty for wrong classified training data
    """
    def __init__(self, kernel, epochs, rate, C):
        self.C = C

        if kernel == 'rbf':
            self.kernel_ = KernelRBF()
        elif kernel == 'linear':
            self.kernel_ = kernel_linear
        elif kernel == 'text':
            self.kernel_ = KernelText()

        self.epochs_ = epochs
        self.rate_ = rate
        self.coef_ = None

    """Predict X's classes.
    """
    def predict(self, X):
        res = np.array([np.sum(self.coef_ * self.y_ * self.kernel_(self.X_, X[i])) for i in range(len(X))])
        return (res >= 0).astype(int)

    """Calculate attributes' weights.
    """
    def get_weights(self):
        return self.X_.T * self.y_ @ self.coef_

    """Fit data X to classes y.
    """
    def fit(self, X, y):
        if hasattr(self.kernel_, 'cache'):
            self.kernel_.cache(X)
        
        self.y_ = y.copy()
        self.y_[y == 0] = -1
        self.X_ = X

        self.coef_ = np.zeros(len(X))
        for _ in range(self.epochs_):
            for i in range(self.coef_.size):
                total = np.sum(self.coef_ * self.y_ * self.kernel_(self.X_, self.X_[i]))
                delta = self.rate_ * (1 - self.y_[i] * total)
                self.coef_[i] = max(0, min(self.coef_[i] + delta, self.C))

"""Generate and return data (blobs or circle).
"""
def generate_data(data_type, n_samples=100):
    if data_type == 'blobs':
        X, y = datasets.make_blobs(
            n_samples=n_samples,
            centers=[[-1, -1], [1, 1]],
            cluster_std=0.4
        )
    elif data_type == 'circle':
        X = (np.random.rand(500, 2) - 0.5) * 20
        y = (np.sqrt(np.sum(X ** 2, axis=1)) > 8).astype(int)

    return X, y

"""Fit the model using data X and classes y. Save it to PDF.
"""
def save_to_pdf(model, X, y, filename, n=20):
    model.fit(X, y)
    minx, miny = np.min(X, axis=0)
    maxx, maxy = np.max(X, axis=0)

    coor = list()
    for i in np.arange(minx, maxx, (maxx - minx) / n):
        for j in np.arange(miny, maxy, (maxy - miny) / n):
            coor.append((i, j))

    coor = np.array(coor)
    y_hat = model.predict(coor)

    plt.figure(figsize=(8, 8))
    plt.scatter(coor[:, 0], coor[:, 1], s=y_hat * 10, color='k', alpha=0.4)
    colors = np.array(['r', 'g'])
    plt.scatter(X[:, 0], X[:, 1], color=colors[y], s=model.coef_* 200 + 20)
    plt.savefig(filename, format='pdf')

"""Get text data from folders "text-data/*".
"""
def get_text_data(origin="text-data"):
    dirs = glob.glob(origin + "/*")
    X, y = [], []
    for i, d in enumerate(dirs):
        files = glob.glob(d + "/*")
        for file_name in files:
            with open(file_name, "rt", encoding="utf8") as file:
                X.append(" ".join(file.readlines()))
        y.extend([i] * len(files))
    return np.array(X), np.array(y)

if __name__ == '__main__':
    #SVM model for "blobs" data using linear kernel
    X, y = generate_data('blobs')
    model = SVM(C=1, kernel='linear', epochs=5000, rate=0.001)
    save_to_pdf(model, X, y, 'blob.pdf')
    
    #SVM model for "circle" data using RBF kernel
    X, y = generate_data('circle')
    model = SVM(C=1, kernel='rbf', epochs=5000, rate=0.001)
    save_to_pdf(model, X, y, 'circle.pdf')

    #SVM model for "text" data using text kernel
    X, y = get_text_data('podatki')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=13)
    model = SVM(C=1, kernel='text', epochs=500, rate=0.001)
    model.fit(X_train, y_train)
    print("Text accuracy score: " + str(accuracy_score(y_test, model.predict(X_test))))