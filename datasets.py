from abc import ABCMeta, abstractmethod
import os
import numpy as np
import pandas as pd
from PIL import Image
from lasagne.utils import floatX
import theano
import sklearn.cross_validation as skcv
import warnings


class Dataset(object):
    """Base class for actual datasets"""

    __metaclass__ = ABCMeta

    def __init__(self):
        """Just for documenting the properties"""
        self._n_samples = None
        self._y = None

    @property
    def n_samples(self):
        """Number of samples in the entire dataset"""
        return self._n_samples

    @property
    def y(self):
        """Labels"""
        return self._y

    @abstractmethod
    def load_batch(self, indices):
        """
        Load a batch of data samples together with labels

        Parameters
        ----------
        indices: array-like, shape = (n_samples,)
            with respect to the entire data set

        Returns
        -------
        X : numpy array, shape = (n_samples, n_channels, n_rows, n_columns)
            batch of data
        y : numpy array, shape = (n_samples,)

        """

    def generate_indices(self, train_frac, val_frac, test_frac, shuffle=False):
        """Generate indices for training, validation and test data.

        Parameters
        ----------
        train_frac, val_frac, test_frac : scalars, values between 0 and 1
            fraction of data that should be used for training, validation
            and testing respectively
        shuffle : boolean (False by default)
            shuffle indices

        Returns
        -------
        train_indices, val_indices, test_indices : numpy arrays,
                                                    shape = (n_samples,)

        """

        warnings.warn('This function does not provide the same relative '
                      'class frequencies in the splits. You might want to '
                      'check the function train_test_split')

        assert train_frac + val_frac + test_frac <= 1.0

        train_size = int(np.ceil(self.n_samples*train_frac))
        val_size = int(np.ceil(self.n_samples*val_frac))
        test_size = int(np.ceil(self.n_samples*test_frac))

        indices = np.arange(self.n_samples)
        if shuffle:
            np.random.shuffle(indices)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:min(train_size +
                                                         val_size + test_size,
                                                         self.n_samples)]

        return train_indices, val_indices, test_indices

    def train_test_split(self, test_size=0.1, train_size=None,
                         deterministic=True):
        """Return a single split into training and test data
           Both parts have approximately the same relative class frequencies

        Parameters
        ----------
        test_size : float (default 0.1), int, or None
            If float, should be between 0.0 and 1.0 and represent the
            proportion of the dataset to include in the test split. If
            int, represents the absolute number of test samples. If None,
            the value is automatically set to the complement of the train size.

        train_size : float, int, or None (default is None)
            If float, should be between 0.0 and 1.0 and represent the
            proportion of the dataset to include in the train split. If
            int, represents the absolute number of train samples. If None,
            the value is automatically set to the complement of the test size.

        deterministic: Boolean

        Returns
        -------
        train_indices, test_indices : nd-arrays

        """
        if deterministic:
            seed = 1234
        else:
            seed = None

        return next(skcv.StratifiedShuffleSplit(self.y, n_iter=1,
                                                test_size=test_size,
                                                train_size=train_size,
                                                random_state=seed)
                    .__iter__())

    def iterate_minibatches(self, indices, batch_size, shuffle=False):
        """
        Generator that yields a batch of data together with labels

        Parameters
        ----------
        indices : array-like, shape = (n_samples,)
            with respect to the entire data set
        batch_size : int
        shuffle : boolean (False by default)
            shuffle indices

        Returns
        -------
        X : numpy array, shape = (n_samples, n_channels, n_rows, n_columns)
            batch of data
        y : numpy array, shape = (n_samples,)

        """

        if shuffle:
            np.random.shuffle(indices)
        for start_idx in range(0, len(indices), batch_size):
            excerpt = indices[start_idx:min(start_idx + batch_size,
                                            len(indices))]
            yield self.load_batch(excerpt)


class KaggleDR(Dataset):
    """
    Provides access to data from Kaggle's Diabetic Retinopathy competition.

    """

    @staticmethod
    def standard_normalize(image):
        """Normalize image to have zero mean and unit variance.

        Subtracts channel MEAN and divides by channel STD

        Parameters
        ----------
        image : numpy array, shape = (n_colors, n_rows, n_columns), dtype =
                                                           theano.config.floatX

        Returns
        -------
        image : numpy array, shape = (n_colors, n_rows, n_columns), dtype =
                                                           theano.config.floatX
        """

        # channel standard deviations (calculated by team o_O)
        STD = np.array([70.53946096, 51.71475228, 43.03428563],
                   dtype=theano.config.floatX)
        # channel means (calculated by team o_O)
        MEAN = np.array([108.64628601, 75.86886597, 54.34005737],
                    dtype=theano.config.floatX)

        return np.divide(np.subtract(image,
                                     MEAN[:, np.newaxis, np.newaxis]),
                                     STD[:, np.newaxis, np.newaxis])

    @staticmethod
    def jf_trafo(image):
        """Apply Jeffrey de Fauw's transformation"""

        #Recovered from model_dump['data_loader_params'].zmuv_mean
        # and *.zmuv_std of 2015_07_17_123003.pkl in Jeffrey's repo:
        ZMUV_MEAN = 0.04166667
        ZMUV_STD = 0.20412415

        image /= 255
        return (image - ZMUV_MEAN) / (0.05 + ZMUV_STD)

    def __init__(self, path_data=None, filename_targets=None,
                 preprocessing=standard_normalize):
        self.path_data = path_data
        self.filename_targets = filename_targets
        labels = pd.read_csv(self.filename_targets, dtype={'level': np.int32})
        self.image_filenames = labels['image']
        # we store all labels
        self._y = np.array(labels['level'])
        self._n_samples = len(self.y)
        # we might cache some data later on
        self.X = None
        # because self.X might be a subset of the entire data set, we track
        # wich samples we have cached
        self.indices_in_X = None
        self.preprocessing = preprocessing

    def load_image(self, filename):
        """
        Load image.

        Parameters
        ----------
        filename : string
            relative filename (path to image folder gets prefixed)

        Returns
        -------
        image : numpy array, shape = (n_rows, n_columns, n_channels)

        """

        filename = os.path.join(self.path_data, filename + '.jpeg')
        return np.array(Image.open(filename))

    def prepare_image(self, im):
        """
        Prepare image.

        Parameters
        ----------
        im : numpy array, shape = (n_rows, n_columns, n_channels)

        Returns
        -------
        processed image : numpy array, shape = (n_channels, n_rows, n_columns)
                                       dtype = floatX

        """

        im = floatX(np.transpose(im, (2, 0, 1)))
        return self.preprocessing(im)

    def load_batch(self, indices):
        """
        Load batch of preprocessed data samples together with labels

        Parameters
        ----------

        indices : array_like, shape = (batch_size,)
            absolute index values refer to position in trainLabels.csv

        """

        if self.indices_in_X is not None and self.X is not None:
            # map indices [0, n_all_samples] to [0, n_stored_samples] while
            # preserving order
            select_from_cached = np.array(
                  [np.where(self.indices_in_X == idx)[0][0] for idx in indices]
            )
            assert len(select_from_cached) == len(indices)
            return self.X[select_from_cached], self.y[indices]

        else:
            X = np.array([self.prepare_image(self.load_image(fn)) for fn in
                          self.image_filenames[indices]])
            y = self.y[indices]
            assert len(X) == len(y) == len(indices)
            return X, y

    def load_data(self, indices):
        """
        Load data, preprocess and cache data

        Parameters
        ----------

        indices : array_like, shape = (n_samples,)
            absolute index values refer to position in trainLabels.csv

        """

        self.X = np.array([self.prepare_image(self.load_image(fn)) for fn in
                           self.image_filenames[indices]])
        self.indices_in_X = indices


class OptRetina(Dataset):
    """
    Provides access to FUNDUS images from OptRetina.

    """

    def __init__(self, path_data=None, filename_targets=None,
                 preprocessing=KaggleDR.jf_trafo, exclude_path=None):
        self.path_data = path_data
        self.filename_targets = filename_targets
        labels = pd.read_csv(self.filename_targets,
                             dtype={'diseased': np.int32})

        if exclude_path is not None:
            labels = OptRetina.exclude_samples(exclude_path,
                                               labels)

        self.image_filenames = OptRetina.build_unique_filenames(labels)
        self.extension = '.jpeg'
        # we store all labels
        self._y = np.array(labels['diseased'])
        self._n_samples = len(self.y)
        # we might cache some data later on
        self.X = None
        # because self.X might be a subset of the entire data set, we track
        # wich samples we have cached
        self.indices_in_X = None
        self.preprocessing = preprocessing

    @staticmethod
    def exclude_samples(exclude_path, labels):
        exclude_filenames = [fn.split('.')[0] for fn in
                             os.listdir(exclude_path)]

        images = labels.filename.apply(lambda fn: fn.split('.')[0]).values
        centre_ids = labels.centre_id.values.astype(str)
        filenames = pd.Series(['_'.join(centre_id_and_image) for
                               centre_id_and_image in
                               zip(centre_ids, images)])

        return labels[~filenames.isin(exclude_filenames)]

    @staticmethod
    def build_unique_filenames(data_frame):
        centre_ids = data_frame.centre_id.values.astype(str)
        n_images = len(data_frame)
        image_filenames = data_frame.filename.values

        filenames = [''.join(parts).split('.')[0] for parts in
                     zip(centre_ids, ['/linked/']*n_images, image_filenames)]

        return np.array(filenames)

    def load_batch(self, indices):
        if self.indices_in_X is not None and self.X is not None:
            # map indices [0, n_all_samples] to [0, n_stored_samples] while
            # preserving order
            select_from_cached = np.array(
                  [np.where(self.indices_in_X == idx)[0][0] for idx in indices]
            )
            assert len(select_from_cached) == len(indices)
            return self.X[select_from_cached], self.y[indices]

        else:
            X = np.array([self.prepare_image(self.load_image(fn)) for fn in
                          self.image_filenames[indices]])
            y = self.y[indices]
            assert len(X) == len(y) == len(indices)
            return X, y

    def prepare_image(self, im):
        """
        Prepare image.

        Parameters
        ----------
        im : numpy array, shape = (n_rows, n_columns, n_channels)

        Returns
        -------
        processed image : numpy array, shape = (n_channels, n_rows, n_columns)
                                       dtype = floatX

        """

        im = floatX(np.transpose(im, (2, 0, 1)))
        return self.preprocessing(im)

    def load_image(self, filename):
        filename = self.build_absolute_filename(filename)
        image = np.array(Image.open(filename))
        assert len(image.shape) == 3, 'Suspicious image ' + filename
        return image

    def build_absolute_filename(self, filename):
        return os.path.join(self.path_data, filename + self.extension)


