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
from keras.preprocessing.image import ImageDataGenerator

import models
from datasets import KaggleDR, OptRetina
from util import SelectiveSampler

batch_size = 10
n_epoch = 10
lr_logreg = 0.005
lr_conv = 0.005
l2_lambda = 0.001  # entire network
l1_lambda = 0.001  # only last layer
size = 512
dataset = 'optretina'

weights_init = 'models/jeffrey_df/2015_07_17_123003_PARAMSDUMP.pkl'
load_previous_weights = False
best_auc = 0.0

# TODO: consider updating keras as data augmentation code has evolved
datagen = ImageDataGenerator(featurewise_center=False,
                             samplewise_center=False,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=False,
                             zca_whitening=False, # apply ZCA whitening
                             rotation_range=10., # degrees (0 to 180)
                             width_shift_range=0.05, # fraction of total width
                             height_shift_range=0.05, # fraction of tot. height
                             horizontal_flip=True,
                             vertical_flip=True)

X = T.tensor4('X')
y = T.ivector('y')

if dataset == 'KaggleDR':
    ds = KaggleDR(path_data='data/kaggle_dr/train_JF_' + str(size),
                  filename_targets='data/kaggle_dr/trainLabels_bin.csv',
                  preprocessing=KaggleDR.jf_trafo)
    ds_test = KaggleDR(path_data='data/kaggle_dr/test_JF_' + str(size),
                       filename_targets='data/kaggle_dr/'
                                        'retinopathy_solution_bin.csv',
                       preprocessing=KaggleDR.jf_trafo)

if dataset == 'optretina':
    ds = OptRetina(path_data='data/optretina/data_JF_' + str(size),
                   filename_targets='data/optretina/OR_diseased_labels.csv',
                   preprocessing=KaggleDR.jf_trafo,
                   exclude_path='data/optretina/data_JF_' + str(size) +
                                '_exclude')

untie_biases = defaultdict(lambda: False, {512: True})

network = models.jeffrey_df(input_var=X, width=size, height=size,
                            filename=weights_init,
                            untie_biases=untie_biases[size])

n_classes = len(np.unique(ds.y))
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
pred_iter = theano.function([X], predictions_det)

if dataset == 'KaggleDR':
    idx_train, idx_val = train_test_split(np.arange(ds.n_samples),
                                          stratify=ds.y,
                                          test_size=0.2,
                                          random_state=1234)
if dataset == 'optretina':
    idx_train_val, idx_test = train_test_split(np.arange(ds.n_samples),
                                               stratify=ds.y,
                                               test_size=0.2,
                                               random_state=1234)
    idx_train, idx_val = train_test_split(idx_train_val,
                                          stratify=ds.y[idx_train_val],
                                          test_size=0.2,
                                          random_state=1234)

y_train = ds.y[idx_train]
N_DISEASED = np.sum(y_train == 1)
IDX_HEALTHY = np.where(y_train == 0)[0]
selective_sampler = SelectiveSampler(M=N_DISEASED, y=y_train)

###########################################################################
# Training
###########################################################################
start_time = time.time()

for epoch in range(n_epoch):
    print('-' * 40)
    print('Epoch', epoch)
    print('-' * 40)
    print("Training...")

    # if epoch == 0:
    print('Select all training data...')
    selection = np.arange(len(idx_train))
    # else:
    #     print('Prediction on diseased images for selective sampling...')
    #     progbar = Progbar(len(IDX_HEALTHY))
    #     probs_neg = np.zeros((len(IDX_HEALTHY),))
    #     pos = 0
    #     for Xb, _ in ds.iterate_minibatches(idx_train[IDX_HEALTHY],
    #                                         batch_size,
    #                                         shuffle=False):
    #         prob_neg = pred_iter(Xb)[:, 0]
    #         probs_neg[pos:pos + Xb.shape[0]] = prob_neg
    #         progbar.add(Xb.shape[0], values=[("prob_neg", prob_neg.mean())])
    #         pos += Xb.shape[0]
    #     selection = selective_sampler.sample(probs_neg=probs_neg, shuffle=True)

    progbar = Progbar(len(selection))
    loss_train = np.zeros((len(selection),))
    predictions_train = np.zeros((len(selection), 2))
    y_train_sel = ds.y[idx_train[selection]]
    pos = 0
    for Xb, yb in ds.iterate_minibatches(idx_train[selection], batch_size=100,
                                         shuffle=False):
        # real-time data augmentation
        for Xb, yb in datagen.flow(Xb, yb,
                               batch_size=batch_size,
                               shuffle=False,
                               seed=None,
                               save_to_dir=None,
                               save_prefix="",
                               save_format="jpeg"):
            if (pos % 40000) == 0:
                params_old = lasagne.layers.get_all_param_values(l_out)
                [loss, predictions] = train_iter(Xb, yb)
                params_new = lasagne.layers.get_all_param_values(l_out)
                params_scale = np.array([np.linalg.norm(p_old.ravel())
                                         for p_old in params_old])
                updates_scale = np.array([np.linalg.norm((p_new - 
                                                          p_old).ravel())
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
    print('Training AUC: ', roc_auc_score(y_train_sel,
                                          predictions_train[:, 1]))

    print('-' * 40)
    print('Epoch', epoch)
    print('-' * 40)
    print("Validating...")
    progbar = Progbar(len(idx_val))
    loss_val = np.zeros(len(idx_val))
    predictions_val = np.zeros((len(idx_val), 2))
    y_val = ds.y[idx_val]
    pos = 0
    for Xb, yb in ds.iterate_minibatches(idx_val, batch_size,
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

print("Loading best weights for testing...")
models.load_weights(l_out, 'best_weights.npz')

print("Testing...")

if dataset == 'KaggleDR':
    ds = ds_test
    idx_test = np.arange(ds_test.n_samples)
if dataset == 'optretina':
    pass

loss_test = np.zeros(len(idx_test))
predictions_test = np.zeros((len(idx_test), 2))
progbar = Progbar(len(idx_test))
pos = 0
for Xb, yb in ds.iterate_minibatches(idx_test,
                                     batch_size, shuffle=False):
    loss, predictions = eval_iter(Xb, yb)
    loss_test[pos:pos + Xb.shape[0]] = loss
    predictions_test[pos:pos + Xb.shape[0]] = predictions

    progbar.add(Xb.shape[0], values=[("test loss", loss)])
    pos += Xb.shape[0]

print('Test loss: ', loss_test.mean())
print('Test AUC: ', roc_auc_score(ds.y[idx_test], predictions_test[:, 1]))
