from __future__ import division
import pytest
import numpy as np


@pytest.fixture
def dataset():
    import datasets
    return datasets.KaggleDR(
        filename_targets='/home/cl/Downloads/data_kaggle_dr/trainLabels.csv')


def test_train_test_split(dataset):
    idx_train, idx_test = dataset.train_test_split(0.1, shuffle=True)
    y = dataset.y
    classes = np.unique(y)
    priors = np.array([np.count_nonzero(y == c_k) for c_k in classes])/len(y)
    priors_train = np.array([np.count_nonzero(y[idx_train] == c_k) for c_k in
                             classes])/len(idx_train)
    priors_test = np.array([np.count_nonzero(y[idx_test] == c_k) for c_k in
                             classes])/len(idx_test)
    np.testing.assert_almost_equal(priors, priors_train, decimal=3)
    np.testing.assert_almost_equal(priors, priors_test, decimal=3)



