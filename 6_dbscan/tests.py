
import unittest
import sklearn
import sklearn.datasets
import numpy as np
import dbscan


class DBSCANTest(unittest.TestCase):
    def setUp(self):
        self.X = sklearn.datasets.load_iris().data[:, (2, 3)]

    def test_dbscan(self):
        dbs = dbscan.DBSCAN(eps=0.4, min_samples=5)
        clusters = dbs.fit_predict(self.X)
        self.assertEqual(len(np.unique(clusters[clusters >= 0])), 2)

        dbs = dbscan.DBSCAN(eps=0.2)
        clusters = dbs.fit_predict(self.X)
        self.assertGreater(len(np.unique(clusters[clusters >= 0])), 3)

        self.assertGreater(np.sum(clusters < 0), 3)

    def test_kdist(self):
        d = dbscan.k_dist(self.X, metric="euclidean", k=4)
        self.assertGreater(min(d), -0.01)
        self.assertGreater(0.6, max(d))
        self.assertEqual(len(d), len(self.X))


if __name__ == "__main__":
    unittest.main(verbosity=2)