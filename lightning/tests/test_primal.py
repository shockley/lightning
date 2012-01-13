import numpy as np
import scipy.sparse as sp

from numpy.testing import assert_array_equal, assert_array_almost_equal, \
                          assert_almost_equal
from nose.tools import assert_raises, assert_true

from sklearn.svm import LinearSVC
from sklearn.svm.sparse import LinearSVC as SparseLinearSVC
from sklearn.datasets import load_iris
from sklearn.datasets.samples_generator import make_classification
from sklearn.utils import check_random_state

from lightning.primal import PrimalClassifier

bin_dense, bin_target = make_classification(n_samples=200, n_features=100,
                                            n_informative=5,
                                            n_classes=2, random_state=0)
bin_sparse = sp.csr_matrix(bin_dense)

mult_dense, mult_target = make_classification(n_samples=300, n_features=100,
                                              n_informative=5,
                                              n_classes=3, random_state=0)
mult_sparse = sp.csr_matrix(mult_dense)


def test_primal_fit_binary():
    for X in (bin_dense, bin_sparse):
        # Caution: use a dense LinearSVC even on sparse data!
        clf = PrimalClassifier(LinearSVC())
        y_pred = clf.fit(X, bin_target).predict(X)
        assert_true(np.mean(y_pred == bin_target) >= 0.98)


def test_primal_fit_multiclass():
    for X in (mult_dense, mult_sparse):
        clf = PrimalClassifier(LinearSVC())
        y_pred = clf.fit(X, mult_target).predict(X)
        assert_true(np.mean(y_pred == mult_target) >= 0.8)
