from __future__ import print_function, division

import gc
import pickle
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.regularization import regularize_network_params, l2
from lasagne.regularization import regularize_layer_params, l1
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score

from keras.utils.generic_utils import Progbar

if __name__ == '__main__':
    import os
    os.sys.path.append('.')

import models
from models import JFnet, JFnetMono
from datasets import KaggleDR
from datasets import DatasetImageDataGenerator
from training import generator_queue
from util import Progplot

# TODO:
# - track true labels due to generator internal shuffling for correct train auc
# - command line script with config files to run a whole set of configurations
# - move as much code as possible to modules for easier reuse

p = 0.2
last_layer = '17'  # from JFnet
batch_size = 32
n_epoch = 30
lr_schedule = {0: 0.005, 1: 0.005, 2: 0.001, 3: 0.001, 4: 0.0005, 5: 0.0001}
change_every = 5
l2_lambda = 0.001  # entire network
l1_lambda = 0.001  # only last layer
size = 512
dataset = 'KaggleDR'
seed = 1234

previous_weights = None

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

datagen_aug = DatasetImageDataGenerator(**AUGMENTATION_PARAMS)

if dataset == 'KaggleDR':
    ds = KaggleDR(path_data='data/kaggle_dr/train_JF_BG_' + str(size),
                  filename_targets='data/kaggle_dr/trainLabels_bin.csv',
                  preprocessing=KaggleDR.standard_normalize,
                  require_both_eyes_same_label=False)
    ds_test = KaggleDR(path_data='data/kaggle_dr/test_JF_BG_' + str(size),
                       filename_targets='data/kaggle_dr/'
                                        'retinopathy_solution_bin.csv',
                       preprocessing=KaggleDR.standard_normalize,
                       require_both_eyes_same_label=False)
    idx_train, idx_val = train_test_split(np.arange(ds.n_samples),
                                          stratify=ds.y,
                                          test_size=0.2,
                                          random_state=seed)

n_classes = len(np.unique(ds.y))

print('-' * 40)
print('JFnet layers: ', last_layer)
print('-' * 40)

best_auc = None

###########################################################################
# Setup network

model = JFnetMono(p_conv=p, last_layer=last_layer, weights=None, n_classes=2)

l_out = model.get_output_layer()
X = model.inputs['X']
y = model.targets['y']

if previous_weights is not None:
    models.load_weights(l_out, previous_weights)
###########################################################################

###########################################################################
# Theano functions


def bayes_cross_entropy(y, ce_loss, n_classes):
    """Dalyac et al. (2014), eq. (17)"""
    priors = T.extra_ops.bincount(y) / y.shape[0]
    weights = 1.0 / (priors[y] * y.shape[0] * n_classes)
    bce_loss = ce_loss * weights
    return bce_loss.sum()


predictions = lasagne.layers.get_output(l_out, deterministic=False)
loss = lasagne.objectives.categorical_crossentropy(predictions, y)
loss = bayes_cross_entropy(y, loss, n_classes)


l2_penalty = l2_lambda * regularize_network_params(l_out, l2)
l1_penalty = l1_lambda * regularize_layer_params(l_out, l1)
loss = loss + l2_penalty + l1_penalty

predictions_det = lasagne.layers.get_output(l_out, deterministic=True)
loss_det = lasagne.objectives.categorical_crossentropy(predictions_det, y)
loss_det = bayes_cross_entropy(y, loss_det, n_classes)

params = lasagne.layers.get_all_params(l_out, trainable=True)
train_iter = {k: theano.function([X, y], [loss, predictions],
                                 updates=lasagne.updates.nesterov_momentum(
                                 loss, params,
                                 learning_rate=lr_schedule[k]))
              for k in lr_schedule.keys()}

eval_iter = theano.function([X, y], [loss_det, predictions_det])
pred_iter = theano.function([X], predictions_det)
###########################################################################

###########################################################################
# Training
###########################################################################
start_time = time.time()
progplot = Progplot(n_epoch, "epochs (batch_size " + str(batch_size) + ")",
                    names=['loss (train)', 'loss (val.)',
                           'AUC (train)', 'AUC (val.)'],
                    title='Finetuning Bayesian JFnet' + last_layer)

y_train = ds.y[idx_train]
N_DISEASED = np.sum(y_train == 1)
IDX_HEALTHY = np.where(y_train == 0)[0]

wait_time = 0.01  # in seconds
multiprocessing = False
data_gen_queue, _stop = generator_queue(datagen_aug.flow_from_dataset(
                                        ds, idx_train,
                                        target_size=(size, size),
                                        batch_size=batch_size,
                                        shuffle=True,
                                        seed=seed),
                                        max_q_size=10,
                                        nb_worker=8,
                                        pickle_safe=multiprocessing)

for epoch in range(n_epoch):
    print('-' * 40)
    print('Epoch', epoch)
    print('-' * 40)
    print("Training...")

    samples_per_epoch = len(idx_train)
    progbar = Progbar(samples_per_epoch)
    loss_train = np.zeros((samples_per_epoch,))
    predictions_train = np.zeros((samples_per_epoch, 2))

    samples_seen = 0
    while samples_seen < samples_per_epoch:
            Xb = yb = None
            while not _stop.is_set():
                if not data_gen_queue.empty():
                    Xb, yb = data_gen_queue.get()
                    Xb = Xb.astype('float32')
                    break
                else:
                    time.sleep(wait_time)
            if samples_seen == 0:
                # check scale of parameter updates at the beginning of
                # each epoch.
                # TODO: refactor the following into a decorator?
                params_old = lasagne.layers.get_all_param_values(l_out)
                [loss, predictions] = train_iter[epoch // change_every](Xb, yb)
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
                # Recompile training function if learning rate has changed
                [loss, predictions] = train_iter[epoch // change_every](Xb, yb)

            loss_train[samples_seen:samples_seen + Xb.shape[0]] = loss
            predictions_train[samples_seen:samples_seen + Xb.shape[0]] = \
                predictions
            
            progbar.add(Xb.shape[0], values=[("train loss", loss)])
            samples_seen += Xb.shape[0]

            if samples_seen > samples_per_epoch:
                Warning('Generated more samples (%d) than expected per epoch'
                        ' (%d).' % (samples_seen, samples_per_epoch))

            if ((samples_seen // batch_size) % (1000 // batch_size)) == 0:
                gc.collect()

    print('Training loss: ', loss_train.mean())
    auc_train = roc_auc_score(ds.y[idx_train], predictions_train[:, 1])
    print('Training AUC: ', auc_train)

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

    progplot.add(values=[("loss (train)", loss_train.mean()),
                         ("AUC (train)", auc_train),
                         ("loss (val.)", loss_val.mean()),
                         ("AUC (val.)", auc_val)])
    if best_auc is None:
        best_auc = auc_val

    if auc_val > best_auc:
        best_auc = auc_val
        print('Saving current best weights...')
        models.save_weights(l_out, 'best_weights' + last_layer + '.npz')

_stop.set()
if multiprocessing:
    data_gen_queue.close()

print("Training took {:.3g} sec.".format(time.time() - start_time))

###########################################################################
# Testing
###########################################################################

print("Loading best weights for testing...")
models.load_weights(l_out, 'best_weights' + last_layer + '.npz')

print("Testing...")

if dataset == 'KaggleDR':
    ds = ds_test
    idx_test = np.arange(ds_test.n_samples)

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

res = {'history': progplot.y,
       'y_test': ds.y[idx_test],
       'pred_test': predictions_test,
       'param values': lasagne.layers.get_all_param_values(l_out)}
pickle.dump(res, open('results' + last_layer + '.pkl', 'wb'))
