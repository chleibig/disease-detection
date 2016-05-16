from __future__ import print_function, division

import os
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import DenseLayer
from lasagne.regularization import regularize_network_params, l2
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score

from keras.utils.generic_utils import Progbar

import models
from datasets import KaggleDR

batch_size = 10
n_epoch = 0
# ToDo:
# * different learning rates for FC layer and lower layers!
# * l1 regularization for softmax layer
# * optimize for AUC
# * check which decision the recommendation from British diabetic association 
#   refers to
learning_rate = 0.0001
l2_lambda = 0.001
size = 512

weights_init = 'models/jeffrey_df/2015_07_17_123003_PARAMSDUMP.pkl'
best_weights = 'best_weights.npz'

# Initialize symbolic variables
X = T.tensor4('X')
y = T.ivector('y')

kdr = KaggleDR(path_data='data/kaggle_dr/train_JF_' + str(size),
               filename_targets='data/kaggle_dr/trainLabels_bin.csv',
               preprocessing=KaggleDR.jf_trafo)
kdr_test = KaggleDR(path_data='data/kaggle_dr/test_JF_' + str(size),
                    filename_targets='data/kaggle_dr/'
                                     'retinopathy_solution_bin.csv',
                    preprocessing=KaggleDR.jf_trafo)

network = models.jeffrey_df(input_var=X, width=size, height=size,
                            filename=weights_init, untie_biases=True)

n_classes = len(np.unique(kdr.y))
network['logreg'] = DenseLayer(network['conv_combined'],
                               num_units=n_classes,
                               nonlinearity=lasagne.nonlinearities.softmax)

l_out = network['logreg']

if os.path.exists(best_weights):
    models.load_weights(l_out, best_weights)

# Scalar loss expression to be minimized during training:
predictions = lasagne.layers.get_output(l_out)
loss = lasagne.objectives.categorical_crossentropy(predictions, y)
# Introduce class weight here?
loss = loss.mean()
# Besides global l2 regularization, l1 might be reasonable for the FC layer
# once we have the histogram implementation
l2_penalty = l2_lambda * regularize_network_params(l_out, l2)
loss = loss + l2_penalty

params = lasagne.layers.get_all_params(l_out, trainable=True)

updates = lasagne.updates.nesterov_momentum(loss, params,
                                            learning_rate=learning_rate,
                                            momentum=0.9)

predictions_det = lasagne.layers.get_output(l_out, deterministic=True)
loss_det = lasagne.objectives.categorical_crossentropy(predictions_det, y)
loss_det = loss_det.mean()

train_iter = theano.function([X, y], [loss, predictions], updates=updates)
eval_iter = theano.function([X, y], [loss_det, predictions_det])

idx_train, idx_val = train_test_split(np.arange(kdr.n_samples), stratify=kdr.y,
                                      test_size=0.2, random_state=1234)


###########################################################################
# Training
###########################################################################
start_time = time.time()

loss_train = np.zeros(len(idx_train))
predictions_train = np.zeros((len(idx_train), 2))

loss_val = np.zeros(len(idx_val))
predictions_val = np.zeros((len(idx_val), 2))

best_auc = 0.0

for epoch in range(n_epoch):
    print('-' * 40)
    print('Epoch', epoch)
    print('-' * 40)
    print("Training...")
    pos = 0
    predictions_train[:] = 0
    loss_train[:] = 0
    progbar = Progbar(len(idx_train))
    perm = np.random.permutation(len(idx_train))
    y_train = kdr.y[idx_train[perm]]
    for Xb, yb in kdr.iterate_minibatches(idx_train[perm], batch_size,
                                          shuffle=False):
        [loss, predictions] = train_iter(Xb, yb)
        loss_train[pos:pos + Xb.shape[0]] = loss
        predictions_train[pos:pos + Xb.shape[0]] = predictions

        progbar.add(Xb.shape[0], values=[("train loss", loss)])
        pos += Xb.shape[0]

    print('Training loss: ', loss_train.mean())
    print('Training AUC: ', roc_auc_score(y_train, predictions_train[:, 1]))

    print('-' * 40)
    print('Epoch', epoch)
    print('-' * 40)
    print("Validating...")
    pos = 0
    loss_val[:] = 0
    predictions_val[:] = 0
    progbar = Progbar(len(idx_val))
    y_val = kdr.y[idx_val]
    for Xb, yb in kdr.iterate_minibatches(idx_val, batch_size,
                                          shuffle=False):
        [loss, predictions] = eval_iter(Xb, yb)
        loss_val[pos:pos + Xb.shape[0]] = loss
        predictions_val[pos:pos + Xb.shape[0]] = predictions

        progbar.add(Xb.shape[0], values=[("val. loss", loss)])
        pos += Xb.shape[0]

    auc_val = roc_auc_score(y_val, predictions_val[:, 1])

    if auc_val > best_auc:
        best_auc = auc_val
        print('Saving currently best weights...')
        save_weights(l_out, 'best_weights.npz')

    print('Validation loss: ', loss_val.mean())
    print('Validation AUC: ', auc_val)


print("Training took {:.3g} sec.".format(time.time() - start_time))


###########################################################################
# Testing
###########################################################################

loss_test = np.zeros(kdr_test.n_samples)
predictions_test = np.zeros((kdr_test.n_samples, 2))

print("Testing...")
pos = 0
progbar = Progbar(kdr_test.n_samples)
for Xb, yb in kdr_test.iterate_minibatches(np.arange(kdr_test.n_samples),
                                           batch_size, shuffle=False):
    loss, predictions = eval_iter(Xb, yb)
    progbar.add(Xb.shape[0], values=[("test loss", loss)])
    loss_test[pos:pos + Xb.shape[0]] = loss
    predictions_test[pos:pos + Xb.shape[0]] = predictions

    progbar.add(Xb.shape[0], values=[("test loss", loss)])
    pos += Xb.shape[0]

print('Test loss: ', loss_test.mean())
print('Test AUC: ', roc_auc_score(kdr_test.y, predictions_test[:, 1]))
