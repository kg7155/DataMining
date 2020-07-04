import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.metrics import f1_score
from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

def sigmoid(x):
    """Calculate logistic function on matrix x
    
    Arguments:
        x {np.array} -- values
    """
    return 1 / (1 + np.exp(-x))

class NeuralNetwork(MLPClassifier):
    """Neural Network ML class """
    def __init__(self, hidden_layer_sizes, alpha):
        """Initialize Neural Network class
        
        Arguments:
            hidden_layer_sizes {list} -- hidden layer sizes
            alpha {float} -- regularization parameter
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha

    def get_params(self, deep=True):
        """Scipy compatibility
        
        Keyword Arguments:
            deep {bool} -- copy also sub networks (default: {True})
        """
        return { "alpha": self.alpha, "hidden_layer_sizes": self.hidden_layer_sizes }

    def set_params(self, **parameters):
        """Scipy compatibility  """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        """Train model
        
        Arguments:
            X {np.array} -- training data
            y {np.array} -- correct predictions
        """
        self.set_data_(X, y)
        self.coefs = fmin_l_bfgs_b(func=self.cost, x0=self.coefs, fprime=self.grad)[0]

    def predict(self, X):
        """Predict classes for X
        
        Arguments:
            X {np.array} -- data
        """
        y = self.predict_proba(X)
        return np.argmax(y, axis=1)

    def predict_proba(self, X):
        """Return predictions probabilites for each class
        
        Arguments:
            X {np.array} -- data
        """
        return self.predict_proba_coefs(X, self.coefs)
    
    def predict_proba_coefs(self, X, coefs):
        """Return predictions probabilites for each class with given coefficients
        
        Arguments:
            X {np.array} -- data
            coefs {np.array} -- coefficients
        """

        a = X
        for w in self.unflatten_coefs(coefs):
            a = sigmoid(self.add_ones(a).dot(w))
        return a

    def grad(self, coefs):
        """Calculate gradients for function optimization
        
        Arguments:
            coefs {np.array} -- coefficients
        """
        A = [self.X]
        m = self.X.shape[0]
        coefs = self.unflatten_coefs(coefs)
        for w in coefs:
            z = self.add_ones(A[-1])
            A.append(sigmoid(np.dot(z, w)))

        d = (A[-1] - self.y) * A[-1] * (1 - A[-1])
        D = [self.add_ones(A[-2]).T.dot(d) / m]
        D[-1][1:, :] += self.alpha * coefs[-1][1:, :] # Apply regularization
        for i in range(len(coefs) - 2, - 1, -1):
            w = coefs[i + 1]
            d = d.dot(w[1:, :].T) * A[i + 1] * (1 - A[i + 1])
            D.append(self.add_ones(A[i]).T.dot(d) / m)
            D[-1][1:, :] += self.alpha * coefs[i][1:, :] # Apply regularization

        return self.flatten_coefs(D[::-1])

    def grad_approx(self, coefs, e):
        """Calculate numeric gradients for epsilon (e)
        
        Arguments:
            coefs {np.array} -- coefficients
            e {float} -- epsilon
        """
        grad = np.zeros(coefs.shape)
        cost = self.cost(coefs)
        for i in range(coefs.size):
            coefs[i] -= e
            grad[i] = (cost - self.cost(coefs)) / e
            coefs[i] += e

        return grad

    def flatten_coefs(self, coefs):
        """Flatten array of matrices into 1xN matrix
        
        Arguments:
            coefs {np.array} -- Coefficients
        """
        return np.concatenate([x.reshape(-1) for x in coefs])

    def unflatten_coefs(self, coefs):
        """Convert flat array of coefficients into a list of coefficients
        
        Arguments:
            coefs {np.array} -- coefficients
        """
        i = 0
        coef_new = []
        for shape in self.shapes:
            size = shape[0] * shape[1]
            coef_new.append(coefs[i:i + size].reshape(shape))
            i += size

        return coef_new

    def init_weights_(self):
        """Initialize weight matrix for given architecture
        """
        self.shapes = []
        coefs = []
        x = self.X.shape[1]
        y = self.y.shape[1]
        
        # Hidden layers
        sizes = [x] + self.hidden_layer_sizes + [y]
        for i in range(1, len(sizes)):
            self.shapes.append([sizes[i - 1] + 1, sizes[i]])
            coefs.append(np.random.uniform(size=self.shapes[-1], low=-0.33, high=0.33))

        return self.flatten_coefs(coefs)

    def cost(self, coefs):
        """Cost function
        
        Arguments:
            coefs {np.array} -- coefficients
        """
        reg = 0 
        for i in self.unflatten_coefs(coefs):
            reg += np.sum(i[1:,:]**2)
        m = self.X.shape[0]
        J = np.sum((self.predict_proba_coefs(self.X, coefs) - self.y)**2)
        return  J / (2 * m) + self.alpha / 2 * reg

    def set_data_(self, X, y):
        """Set the data and initialize coefficients
        
        Arguments:
            X {np.array} -- data
            y {np.array} -- real classes
        """
        self.X = X
        self.y = np.eye(y.max() + 1)[y]

        self.coefs = self.init_weights_()

    def add_ones(self, x):
        """Add bias to matrix 
        
        Arguments:
            x {np.array} -- Data
        """
        return np.hstack([np.ones((x.shape[0], 1)), x])


def cross_validation():
    """Run cross validation and compare 3 different classifiers
    """
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    kf = KFold(n_splits=5, shuffle=True)
    models = [NeuralNetwork([8, 2], alpha=1e-5), LogisticRegression(), GradientBoostingClassifier()]
    predictions = [np.zeros(y.shape) for i in models]
    for i, m in enumerate(models):
        for train_index, test_index in kf.split(X):
            m.fit(X[train_index], y[train_index])
            predictions[i][test_index] = m.predict(X[test_index])

    return [f1_score(y, predict, average='micro') for predict in  predictions]

if __name__ == '__main__':
    print(cross_validation())

    