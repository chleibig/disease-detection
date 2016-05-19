from __future__ import print_function, division

from collections import defaultdict
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import DenseLayer
from lasagne.regularization import regularize_network_params, l2
from lasagne.regularization import regularize_layer_params, l1
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score

from keras.utils.generic_utils import Progbar

import models
from datasets import KaggleDR

batch_size = 4
n_epoch = 10
lr_logreg = 0.001
lr_conv = 0.0001
l2_lambda = 0.0005  # entire network
l1_lambda = 0.0005  # only last layer
size = 2048

weights_init = 'models/jeffrey_df/2015_07_17_123003_PARAMSDUMP.pkl'
load_previous_weights = True
best_auc = 0.89435

X = T.tensor4('X')
y = T.ivector('y')

kdr = KaggleDR(path_data='data/kaggle_dr/train_JF_' + str(size),
               filename_targets='data/kaggle_dr/trainLabels_bin.csv',
               preprocessing=KaggleDR.jf_trafo)
kdr_test = KaggleDR(path_data='data/kaggle_dr/test_JF_' + str(size),
                    filename_targets='data/kaggle_dr/'
                                     'retinopathy_solution_bin.csv',
                    preprocessing=KaggleDR.jf_trafo)

untie_biases = defaultdict(lambda: False, {512: True})

network = models.jeffrey_df(input_var=X, width=size, height=size,
                            filename=weights_init,
                            untie_biases=untie_biases[size])

n_classes = len(np.unique(kdr.y))
network['logreg'] = DenseLayer(network['conv_combined'],
                               num_units=n_classes,
                               nonlinearity=lasagne.nonlinearities.softmax)

l_out = network['logreg']

if load_previous_weights:
    models.load_weights(l_out, 'best_weights.npz')

# Scalar loss expression to be minimized during training:
predictions = lasagne.layers.get_output(l_out)
loss = lasagne.objectives.categorical_crossentropy(predictions, y)
# Introduce class weight here?
loss = loss.mean()

l2_penalty = l2_lambda * regularize_network_params(l_out, l2)
l1_penalty = l1_lambda * regularize_layer_params(l_out, l1)
loss = loss + l2_penalty + l1_penalty


def build_updates(network, lr_conv, lr_logreg):
    """Build updates with different learning rates for the conv stack
       and the logistic regression layer

    """

    params_conv = lasagne.layers.get_all_params(network['conv_combined'],
                                                trainable=True)
    updates_conv = lasagne.updates.sgd(loss, params_conv,
                                       learning_rate=lr_conv)
    params_logreg = network['logreg'].get_params(trainable=True)
    updates_logreg = lasagne.updates.sgd(loss, params_logreg,
                                         learning_rate=lr_logreg)
    updates = updates_conv
    for k, v in updates_logreg.items():
        updates[k] = v

    return lasagne.updates.apply_nesterov_momentum(updates, momentum=0.9)


predictions_det = lasagne.layers.get_output(l_out, deterministic=True)
loss_det = lasagne.objectives.categorical_crossentropy(predictions_det, y)
loss_det = loss_det.mean()

updates = build_updates(network, lr_conv, lr_logreg)
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
	if (pos % 1000) == 0:
            params_old = lasagne.layers.get_all_param_values(l_out)
            [loss, predictions] = train_iter(Xb, yb)
            params_new = lasagne.layers.get_all_param_values(l_out)
            params_scale = np.array([np.linalg.norm(p_old.ravel())
                                     for p_old in params_old])
            updates_scale = np.array([np.linalg.norm((p_new - p_old).ravel())
                                      for p_new, p_old in
                                      zip(params_new, params_old)])
            print('update_scale/param_scale: ',
                  np.divide(updates_scale, params_scale))
        else:
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

    print('Validation loss: ', loss_val.mean())
    print('Validation AUC: ', auc_val)

    if auc_val > best_auc:
        best_auc = auc_val
        print('Saving currently best weights...')
        models.save_weights(l_out, 'best_weights.npz')


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
    loss_test[pos:pos + Xb.shape[0]] = loss
    predictions_test[pos:pos + Xb.shape[0]] = predictions

    progbar.add(Xb.shape[0], values=[("test loss", loss)])
    pos += Xb.shape[0]

print('Test loss: ', loss_test.mean())
print('Test AUC: ', roc_auc_score(kdr_test.y, predictions_test[:, 1]))
