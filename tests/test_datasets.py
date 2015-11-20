from __future__ import division
import pytest
import numpy as np


@pytest.fixture
def dataset():
    import datasets
    return datasets.KaggleDR(
        filename_targets='/home/cl/Downloads/data_kaggle_dr/trainLabels.csv')


def test_train_test_split(dataset):

    test_sizes = [0.01, 0.1]
    train_sizes = [0.02, None]

    for test_size, train_size in zip(test_sizes, train_sizes):
        idx_train, idx_test = dataset.train_test_split(test_size,
                                                       train_size=train_size)
        if train_size is None:
            train_size = 1 - test_size
        assert abs(len(idx_train) - train_size*dataset.n_samples) < 3
        assert abs(len(idx_test) - test_size*dataset.n_samples) < 3

        y = dataset.y
        classes = np.unique(y)
        rel_freq = np.array([np.count_nonzero(y == c_k) for c_k in classes])\
                   /len(y)
        rel_freq_train = np.array([np.count_nonzero(y[idx_train] == c_k)
                                   for c_k in classes])/len(idx_train)
        rel_freq_test = np.array([np.count_nonzero(y[idx_test] == c_k)
                                  for c_k in classes])/len(idx_test)
        np.testing.assert_almost_equal(rel_freq, rel_freq_train, decimal=3)
        np.testing.assert_almost_equal(rel_freq, rel_freq_test, decimal=3)



