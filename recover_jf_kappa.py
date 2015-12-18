from __future__ import print_function, division

import os
import numpy as np
import theano
import theano.tensor as T
import lasagne
from keras.utils.generic_utils import Progbar

from datasets import KaggleDR
from util import quadratic_weighted_kappa

import cPickle as pickle
import models

X = T.tensor4('X')
y = T.ivector('y')
img_dim = T.matrix('img_dim')

weights = '/home/cl/Downloads/kdr_solutions/JeffreyDF/' \
          'kaggle_diabetic_retinopathy/dumps/' \
          '2015_07_17_123003_PARAMSDUMP.pkl'

#Recovered from model_dump['data_loader_params'].zmuv_mean and *.zmuv_std
ZMUV_MEAN = np.array([[[[ 0.04166667]]]], dtype=np.float32)
ZMUV_STD = np.array([[[[ 0.20412415]]]], dtype=np.float32)

network = models.jeffrey_df(width=512, height=512, filename=weights)

network['0'].input_var = X
network['22'].input_var = img_dim
prob = network['31']

path = '/home/cl/Downloads/data_kaggle_dr'
batch_size = 2
n_epoch = 1
fn_labels = 'trainLabels_partial.csv'

kdr = KaggleDR(path_data=os.path.join(path, 'train_partial_JF_512'),
               filename_targets=os.path.join(path, fn_labels))

output = lasagne.layers.get_output(prob, deterministic=True)
y_pred = T.argmax(output, axis=1)
acc = T.mean(T.eq(y_pred, y), dtype=theano.config.floatX)

fn = theano.function([X, y, img_dim], [acc, y_pred])

all_y_pred = np.zeros((kdr.n_samples,), dtype=np.int32)
all_y = np.zeros((kdr.n_samples,), dtype=np.int32)
idx = 0
progbar = Progbar(kdr.n_samples)
for batch in kdr.iterate_minibatches(np.arange(kdr.n_samples),
                                     batch_size, shuffle=False):
    inputs, targets = batch
    # Reverse standard_normalize transformation performed by
    # KaggleDR.standard_normalize
    inputs  = (inputs * KaggleDR.STD[:, np.newaxis, np.newaxis]) + \
              KaggleDR.MEAN[:, np.newaxis, np.newaxis]
    #  and apply instead:
    inputs /= 255
    inputs = (inputs - ZMUV_MEAN) / (0.05 + ZMUV_STD) # values from
    n_s = len(targets)
    _img_dim = np.concatenate(
        (np.full((n_s, 1), inputs.shape[2], dtype=np.float32),
         np.full((n_s, 1), inputs.shape[3], dtype=np.float32)),
        axis=1)/700. #jeffrey does that under load_image_and_process?!
    acc, labels = fn(inputs, targets, _img_dim)
    progbar.add(inputs.shape[0], values=[("test accuracy", acc)])
    all_y_pred[idx:idx+len(targets)] = labels
    all_y[idx:idx+len(targets)] = targets
    idx += len(targets)

acc_all = np.mean(np.equal(all_y_pred, all_y))
kp_all = quadratic_weighted_kappa(all_y_pred, all_y, 5)

print('Accuracy:', acc_all)
print('Kappa:', kp_all)

results = {'y_pred': all_y_pred, 'y': all_y,
           'acc': acc_all, 'kappa': kp_all}

# with open('results.pkl', 'wb') as h:
#     pickle.dump(results, h)
