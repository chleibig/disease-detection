from __future__ import division
import pytest
import numpy as np
import datasets


class TestDataset:
    @pytest.fixture
    def dataset(self):
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

class TestKaggleDR:
    @pytest.fixture
    def dataset(self):
        dataset = datasets.KaggleDR(
                        path_data='tests/ref_data/KDR/sample_JF_512',
                        filename_targets='tests/ref_data/KDR/sampleLabels.csv')
        return dataset

    def test_preprocessing_is_applied(self, dataset):
        dataset.preprocessing = datasets.KaggleDR.standard_normalize
        sn_image_0, _ = dataset.load_batch([0])
        dataset.preprocessing = datasets.KaggleDR.jf_trafo
        jf_image_0, _ = dataset.load_batch([0])
        dataset.preprocessing = datasets.KaggleDR.standard_normalize
        sn_image_1, _ = dataset.load_batch([0])
        dataset.preprocessing = datasets.KaggleDR.jf_trafo
        jf_image_1, _ = dataset.load_batch([0])

        np.testing.assert_array_equal(sn_image_0, sn_image_1)
        np.testing.assert_array_equal(jf_image_0, jf_image_1)

class TestOptRetina:
    @pytest.fixture
    def dataset(self):
        dataset = datasets.OptRetina(
                         path_data='tests/ref_data/OR/sample_JF_512',
                         filename_targets='tests/ref_data/OR/sampleLabels.csv',
                         preprocessing=datasets.KaggleDR.standard_normalize)
        return dataset

    def test_build_unique_filenames(self):
        import pandas as pd
        labels = pd.read_csv('tests/ref_data/OR/sampleLabels.csv', dtype={
            'level': np.int32})
        unique_filenames = datasets.OptRetina.build_unique_filenames(labels)
        assert unique_filenames[0] == '21/linked/anonymized_3558'

    def test_build_absolute_filename(self, dataset):
        abs_fn = 'tests/ref_data/OR/sample_JF_512/10/linked/anonymized_3566' \
                 '.jpeg'
        unique_fn = '10/linked/anonymized_3566'
        assert abs_fn == dataset.build_absolute_filename(unique_fn)

    def test_load_batch(self, dataset):
        indices = [3, 4]
        X, y = dataset.load_batch(indices)
        assert len(X) == len(y) == len(indices)
        assert X.shape == (len(indices), 3, 512, 512)




