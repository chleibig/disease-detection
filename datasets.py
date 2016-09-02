from __future__ import print_function
from abc import ABCMeta, abstractmethod
import glob
import os
import threading
import warnings

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, array_to_img
from keras import backend as K
from lasagne.utils import floatX
import numpy as np
import pandas as pd
from PIL import Image
import theano
import sklearn.cross_validation as skcv


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

        train_size = int(np.ceil(self.n_samples * train_frac))
        val_size = int(np.ceil(self.n_samples * val_frac))
        test_size = int(np.ceil(self.n_samples * test_frac))

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

        # Recovered from model_dump['data_loader_params'].zmuv_mean
        # and *.zmuv_std of 2015_07_17_123003.pkl in Jeffrey's repo:
        ZMUV_MEAN = 0.04166667
        ZMUV_STD = 0.20412415

        image /= 255
        return (image - ZMUV_MEAN) / (0.05 + ZMUV_STD)

    def __init__(self, path_data=None, filename_targets=None,
                 preprocessing=standard_normalize,
                 require_both_eyes_same_label=False):
        self.path_data = path_data
        self.filename_targets = filename_targets
        labels = pd.read_csv(filename_targets, dtype={'level': np.int32})
        if require_both_eyes_same_label:
            labels = KaggleDR.contralateral_agreement(labels)
        self.image_filenames = labels['image'].values
        # we store all labels
        self._y = np.array(labels['level'])
        self._n_samples = len(self.y)
        # we might cache some data later on
        self.X = None
        # because self.X might be a subset of the entire data set, we track
        # wich samples we have cached
        self.indices_in_X = None
        self.preprocessing = preprocessing

    @staticmethod
    def contralateral_agreement(df):
        """Get only the samples for which the contralateral image had been
           assigned the same label

        Parameters
        ==========
        df: pandas data frame
            all samples

        Returns
        =======

        df: pandas data frame
            just the samples with contralateral label agreement

        """

        left = df.image.str.contains(r'\d+_left')
        right = df.image.str.contains(r'\d+_right')
        df[left].level == df[right].level
        accepted_patients = (df[left].level == df[right].level).values
        accepted_images_left = df[left].image[accepted_patients]
        accepted_images_right = df[right].image[accepted_patients]
        accepted_images = pd.concat((accepted_images_left,
                                     accepted_images_right))
        return df[df.image.isin(accepted_images)]

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
                [np.where(self.indices_in_X == idx)[0][0] for idx in indices])
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


class Messidor(KaggleDR):

    def __init__(self, path_data=None,
                 filename_targets='data/messidor/messidor.csv',
                 preprocessing=KaggleDR.standard_normalize):
        super(Messidor, self).__init__(path_data=path_data,
                                       filename_targets=filename_targets,
                                       preprocessing=preprocessing,
                                       require_both_eyes_same_label=False)

    @staticmethod
    def prepare_labels():
        """ Prepare csv labels file from messidor's excel sheets

        With the resulting labels file, one should be able to use Messidor data
        the same way as KaggleDR data

        """

        labels_file = 'data/messidor/messidor.csv'

        if os.path.exists(labels_file):
            print(labels_file, 'already exists.')
            labels = pd.read_csv(labels_file)
        else:
            labels = pd.DataFrame({'image': pd.Series(dtype='str'),
                                   'level': pd.Series(dtype='int32')})
            filenames = glob.glob('data/messidor/Annotation*Base*.xls')
            for fn in filenames:
                df = pd.read_excel(fn, converters={'Retinopathy grade': np.int32})
                chunk = pd.DataFrame(
                    {'image': df['Image name'].apply(lambda x: x.split('.tif')[0]),
                     'level': df['Retinopathy grade']})
                labels = labels.append(chunk)
            labels.to_csv(labels_file, index=False)

        assert len(labels) == 1200

        labels_file_R0vsR1 = 'data/messidor/messidor_R0vsR1.csv'

        if os.path.exists(labels_file_R0vsR1):
            print(labels_file_R0vsR1, 'already exists.')
            labels_ROvsR1 = pd.read_csv(labels_file_R0vsR1)

        else:
            labels_ROvsR1 = labels[(labels.level == 0) | (labels.level == 1)]
            labels_ROvsR1.to_csv(labels_file_R0vsR1, index=False)

        assert len(labels_ROvsR1) == 699

        return labels, labels_ROvsR1

    @staticmethod
    def contralateral_agreement(df):
        raise NotImplementedError('Undefined for Messidor.')


class DatasetImageDataGenerator(ImageDataGenerator):

    def flow_from_dataset(self, dataset, indices,
                          target_size=(512, 512),
                          dim_ordering='default',
                          batch_size=32,
                          shuffle=True,
                          seed=None,
                          save_to_dir=None,
                          save_prefix='',
                          save_format='jpeg'):
        return DatasetIterator(dataset, indices, self,
                               target_size=target_size,
                               dim_ordering=dim_ordering,
                               batch_size=batch_size,
                               shuffle=shuffle,
                               seed=seed,
                               save_to_dir=save_to_dir,
                               save_prefix=save_prefix,
                               save_format=save_format)


class DatasetIterator(object):
    """Inspired by keras.preprocessing.image.(NumpyArray)Iterator"""
    def __init__(self, dataset, indices, image_data_generator,
                 target_size=(512, 512),
                 dim_ordering='default',
                 batch_size=32, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg'):
        self.dataset = dataset
        self.indices = indices
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.dim_ordering = dim_ordering
        if self.dim_ordering == 'tf':
            self.image_shape = self.target_size + (3,)
        else:
            self.image_shape = (3,) + self.target_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(indices, batch_size, shuffle,
                                                seed)
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

    def _flow_index(self, indices, batch_size=32, shuffle=False, seed=None):
        self.batch_index = 0
        while 1:
            if self.batch_index == 0:
                index_array = indices
                N = len(index_array)
                if shuffle:
                    if seed is not None:
                        np.random.seed(seed + self.total_batches_seen)
                    np.random.shuffle(index_array)

            current_index = (self.batch_index * batch_size) % N
            if N >= current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = N - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index +
                               current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        # needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = \
                next(self.index_generator)
        # The transformation of images is not under thread lock so it can be
        # done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape)
        for i, j in enumerate(index_array):
            img = self.dataset.load_image(self.dataset.image_filenames[j])
            x = img_to_array(img, dim_ordering=self.dim_ordering)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            # instead of via standardize we have our own preprocessing routine
            # attached to the dataset with cached dataset statistics:
            x = self.dataset.preprocessing(x)
            batch_x[i] = x
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=current_index + i,
                    hash=np.random.randint(1e4),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        if self.dataset.y is None:
            return batch_x
        batch_y = self.dataset.y[index_array]
        return batch_x, batch_y

