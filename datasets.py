from abc import ABCMeta, abstractmethod, abstractproperty
import os
import numpy as np
import pandas as pd
from PIL import Image
from lasagne.utils import floatX
import theano


class Dataset(object):
    """Base class for actual datasets"""

    __metaclass__ = ABCMeta

    @abstractproperty
    def n_samples(self):
        pass

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

    # channel standard deviations (calculated by team o_O)
    STD = np.array([70.53946096, 51.71475228, 43.03428563],
                   dtype=theano.config.floatX)
    # channel means (calculated by team o_O)
    MEAN = np.array([108.64628601, 75.86886597, 54.34005737],
                    dtype=theano.config.floatX)

    def __init__(self, path_data=None, filename_targets=None):
        self.path_data = path_data
        self.filename_targets = filename_targets
        labels = pd.read_csv(self.filename_targets, dtype={'level': np.int32})
        self.image_filenames = labels['image']
        # we store all labels
        self.y = np.array(labels['level'])
        self._n_samples = len(self.y)
        # we might cache some data later on
        self.X = None
        # because self.X might be a subset of the entire data set, we track
        # wich samples we have cached
        self.indices_in_X = None

    @property
    def n_samples(self):
        """Number of samples in the entire dataset"""
        return self._n_samples

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

        image = np.subtract(image, KaggleDR.MEAN[:, np.newaxis, np.newaxis])
        image = np.divide(image, KaggleDR.STD[:, np.newaxis, np.newaxis])
        return image

    @staticmethod
    def prepare_image(im):
        """
        Prepare image.

        Dimensions get reordered according to theano/lasagne conventions
        Colour channels are inverted: RGB -> BGR
        cast to floatX

        Parameters
        ----------
        im : numpy array, shape = (n_rows, n_columns, n_channels)

        Returns
        -------
        processed image : numpy array, shape = (n_channels, n_rows, n_columns)
                                       dtype = floatX

        """

        # Returned image should be (n_channels, n_rows, n_columns)
        im = np.transpose(im, (2, 0, 1))
        # Convert to BGR
        im = im[::-1, :, :]
        return floatX(im)

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
