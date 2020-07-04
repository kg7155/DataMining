import unittest
import sklearn
import numpy as np
from sklearn import decomposition
from sklearn import datasets

from pca import EigenPCA, PowerPCA, OrtoPCA


class PCATest(unittest.TestCase):
    def setUp(self):
        X = datasets.load_iris().data
        self.X = X - X.mean(axis=0)
        self.n_comp = 3

    def fit_test(self, method):
        s_pca = sklearn.decomposition.PCA(self.n_comp)
        s_pca.fit(self.X)

        my_pca = method(n_components=self.n_comp)
        my_pca.fit(self.X)

        self.assertEqual(my_pca.explained_variance_.shape, (self.n_comp, ))
        np.testing.assert_array_almost_equal(my_pca.explained_variance_,
                                             s_pca.explained_variance_, decimal=1)
        np.testing.assert_array_almost_equal(my_pca.explained_variance_ratio_,
                                             s_pca.explained_variance_ratio_, decimal=1)

        diff = sum(min(np.linalg.norm(r1 - r2), np.linalg.norm(r1 + r2))
                   for r1, r2 in zip(my_pca.components_, s_pca.components_))
        self.assertAlmostEqual(diff, 0.)

    def transform_test(self, method):
        n_sample = 5
        train, test = self.X[:-n_sample], self.X[-n_sample:]

        s_pca = sklearn.decomposition.PCA(self.n_comp)
        s_pca.fit(train)
        sk_T = s_pca.transform(test)

        my_pca = method(self.n_comp)
        my_pca.fit(train)

        my_T = my_pca.transform(self.X[-n_sample:])
        self.assertEqual(my_T.shape, (n_sample, self.n_comp))
        for c1, c2 in zip(sk_T.T, my_T.T):
            np.testing.assert_almost_equal(min(np.linalg.norm(c1 - c2), np.linalg.norm(c1 + c2)), 0.)

    def method_test(self, method):
        self.fit_test(method)
        self.transform_test(method)

    def test_EigenPCA(self):
        self.method_test(EigenPCA)

    def test_PowerPCA(self):
        self.method_test(PowerPCA)

    def test_OrtoPCA(self):
        self.method_test(OrtoPCA)

if __name__ == "__main__":
    unittest.main(verbosity=2)