from __future__ import division
import pytest
import numpy as np


class TestDataset:
    @pytest.fixture
    def dataset(self):
        import datasets

        class SimpleDataset(datasets.Dataset):
            def __init__(self, y):
                self._y = y
                self._n_samples = len(y)

            @property
            def n_samples(self):
                return self._n_samples

            @property
            def y(self):
                return self._y

            def load_batch(self, indices):
                pass

        n_samples = 1000
        rel_freq = [0.73, 0.07, 0.15, 0.03, 0.02]
        y = np.concatenate([np.ones(rf*n_samples, dtype=np.int32)*i
                            for i, rf in enumerate(rel_freq)])
        np.random.shuffle(y)
        return SimpleDataset(y)

    def test_train_test_split(self, dataset):
            test_size = 0.1
            train_size = None
            idx_train, idx_test = dataset.train_test_split(
                                                         test_size=test_size,
                                                         train_size=train_size,
                                                         deterministic=True)
            if train_size is None:
                train_size = 1 - test_size
            assert abs(len(idx_train) - train_size*dataset.n_samples) < 3
            assert abs(len(idx_test) - test_size*dataset.n_samples) < 3

            y = dataset.y
            classes = np.unique(y)
            rel_freq = np.array([np.count_nonzero(y == c_k)
                                 for c_k in classes])/len(y)
            rel_freq_train = np.array([np.count_nonzero(y[idx_train] == c_k)
                                       for c_k in classes])/len(idx_train)
            rel_freq_test = np.array([np.count_nonzero(y[idx_test] == c_k)
                                      for c_k in classes])/len(idx_test)
            np.testing.assert_almost_equal(rel_freq, rel_freq_train,
                                           decimal=3)
            np.testing.assert_almost_equal(rel_freq, rel_freq_test, decimal=3)

            test_idx_train, test_idx_test = dataset.train_test_split(
                                                           test_size=test_size,
                                                         train_size=train_size,
                                                         deterministic=True)
            np.testing.assert_array_equal(idx_train, test_idx_train)
            np.testing.assert_array_equal(idx_test, test_idx_test)