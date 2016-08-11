from __future__ import print_function, division

import numpy as np
import pandas as pd
import theano
import theano.tensor as T
import lasagne
from keras.utils.generic_utils import Progbar

from datasets import KaggleDR
from util import quadratic_weighted_kappa

import cPickle as pickle
from models import JFnet


X = T.tensor4('X')
y = T.ivector('y')
img_dim = T.matrix('img_dim')

T = 100  # Number of MC dropout samples

weights = 'models/jeffrey_df/2015_07_17_123003_PARAMSDUMP.pkl'

network = JFnet.build_model(width=512, height=512, filename=weights)

network['0'].input_var = X
network['22'].input_var = img_dim
l_out = network['31']

fname_labels = 'data/kaggle_dr/trainLabels_wh.csv'

kdr = KaggleDR(path_data='data/kaggle_dr/train_JF_512',
               filename_targets=fname_labels,
               preprocessing=KaggleDR.jf_trafo)

df = pd.read_csv(fname_labels)
width = df.width.values.astype(theano.config.floatX)
height = df.height.values.astype(theano.config.floatX)

det_fn = theano.function([X, img_dim],
                         lasagne.layers.get_output(l_out,
                                                   deterministic=True))
stoch_fn = theano.function([X, img_dim],
                           lasagne.layers.get_output(l_out,
                                                     deterministic=False))

det_out = np.zeros((kdr.n_samples, 5), dtype=np.float32)
stoch_out = np.zeros((kdr.n_samples, 5, T), dtype=np.float32)

idx = 0
progbar = Progbar(kdr.n_samples)
for batch in kdr.iterate_minibatches(np.arange(kdr.n_samples),
                                     batch_size=512, shuffle=False):
    inputs, targets = batch
    n_s = len(targets)

    _img_dim = JFnet.get_img_dim(width, height, idx, n_s)

    det_out[idx:idx + n_s] = det_fn(inputs, _img_dim)
    for t in range(T):
        stoch_out[idx:idx + n_s, :, t] = stoch_fn(inputs, _img_dim)
    progbar.add(inputs.shape[0])
    idx += n_s

det_y_pred = np.argmax(det_out, axis=1)
det_acc = np.mean(np.equal(det_y_pred, kdr.y))
det_kappa = quadratic_weighted_kappa(det_y_pred, kdr.y, 5)

results = {'det_out': det_out,
           'stoch_out': stoch_out,
           'det_kappa': det_kappa,
           'det_acc': det_acc}

with open('jfnet_with_uncertainty_results_KaggleDR_train.pkl', 'wb') as h:
    pickle.dump(results, h)
