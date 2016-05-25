from __future__ import print_function
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from datasets import KaggleDR

AUGMENTATION_PARAMS = {'featurewise_center': False,
                       'samplewise_center': False,
                       'featurewise_std_normalization': False,
                       'samplewise_std_normalization': False,
                       'zca_whitening': False,
                       'rotation_range': 180.,
                       'width_shift_range': 0.05,
                       'height_shift_range': 0.05,
                       'shear_range': 0.,
                       'zoom_range': 0.10,
                       'channel_shift_range': 0.,
                       'fill_mode': 'constant',
                       'cval': 0.,
                       'horizontal_flip': True,
                       'vertical_flip': True,
                       'dim_ordering': 'th'}

NO_AUGMENTATION_PARAMS = {'featurewise_center': False,
                          'samplewise_center': False,
                          'featurewise_std_normalization': False,
                          'samplewise_std_normalization': False,
                          'zca_whitening': False,
                          'rotation_range': 0.,
                          'width_shift_range': 0.,
                          'height_shift_range': 0.,
                          'shear_range': 0.,
                          'zoom_range': 0.,
                          'channel_shift_range': 0.,
                          'fill_mode': 'nearest',
                          'cval': 0.,
                          'horizontal_flip': False,
                          'vertical_flip': False,
                          'dim_ordering': 'th'}

datagen_aug = ImageDataGenerator(**AUGMENTATION_PARAMS)
datagen_no_aug = ImageDataGenerator(**NO_AUGMENTATION_PARAMS)

size = 512
batch_size = 5

ds = KaggleDR(path_data='data/kaggle_dr/sample_JF_' + str(size),
              filename_targets='data/kaggle_dr/sampleLabels.csv',
              preprocessing=KaggleDR.jf_trafo)

outer_batch = 0
for Xb_outer, yb_outer in ds.iterate_minibatches(np.arange(ds.n_samples), 10,
                                                 shuffle=False):
    print('outer_batch', outer_batch)
    outer_batch += 1
    inner_samples = 0
    for Xb, yb in datagen_aug.flow(Xb_outer, yb_outer,
                                   batch_size=3,
                                   shuffle=False,
                                   seed=None,
                                   save_to_dir='data/kaggle_dr/'
                                               'sample_JF_512/aug',
                                   save_prefix="",
                                   save_format="jpeg"):
        inner_samples += Xb.shape[0]
        print('inner samples', inner_samples)
        if inner_samples >= len(Xb_outer):
            print('inner samples >= outer batch size, stopping.')
            break
