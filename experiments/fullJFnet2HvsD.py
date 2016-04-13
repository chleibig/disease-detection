from __future__ import print_function, division

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

weights = 'models/jeffrey_df/2015_07_17_123003_PARAMSDUMP.pkl'

network = models.jeffrey_df(width=512, height=512, filename=weights)

network['0'].input_var = X
network['22'].input_var = img_dim
prob = network['31']

kdr = KaggleDR(path_data='data/kaggle_dr/test_JF_512',
               filename_targets='data/kaggle_dr/retinopathy_solution.csv',
               preprocessing=KaggleDR.jf_trafo)

output = lasagne.layers.get_output(prob, deterministic=True)
forward_pass = theano.function([X, img_dim], output)

probs = np.zeros((kdr.n_samples, 5), dtype=np.float32)
idx = 0
progbar = Progbar(kdr.n_samples)
for batch in kdr.iterate_minibatches(np.arange(kdr.n_samples),
                                     batch_size=2, shuffle=False):
    inputs, targets = batch
    # Replace the following by using the correct image sizes
    n_s = len(targets)
    _img_dim = np.concatenate(
        (np.full((n_s, 1), inputs.shape[2], dtype=np.float32),
         np.full((n_s, 1), inputs.shape[3], dtype=np.float32)),
        axis=1)/700.  # jeffrey does that under load_image_and_process?!
    probs[idx:idx+len(targets)] = forward_pass(inputs, _img_dim)
    progbar.add(inputs.shape[0])
    idx += len(targets)

y_pred = np.argmax(probs, axis=1)
acc = np.mean(np.equal(y_pred, kdr.y))
kappa = quadratic_weighted_kappa(y_pred, kdr.y, 5)

probs_bin = np.zeros((probs.shape[0], 2))
probs_bin[:, 0] = probs[:, 0]
probs_bin[:, 1] = np.sum(probs[:, 1:], axis=1)

results = {'probabilities': probs}

with open('results.pkl', 'wb') as h:
    pickle.dump(results, h)
