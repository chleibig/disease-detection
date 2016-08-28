from __future__ import print_function, division

import gc

import pickle
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

if __name__ == '__main__':
    import os
    os.sys.path.append('.')

import models
from models import JFnet
from datasets import KaggleDR
from util import SelectiveSampler
from util import Progplot


p = 0.2
last_layer = '17'  # from JFnet
batch_size = 32
n_epoch = 30
lr_schedule = {0: 0.005, 1: 0.005, 2: 0.005, 3: 0.005, 4: 0.001, 5: 0.001}
change_every = 5
l2_lambda = 0.001  # entire network
l1_lambda = 0.001  # only last layer
size = 512
dataset = 'KaggleDR'

weights_init = 'models/jeffrey_df/2015_07_17_123003_PARAMSDUMP.pkl'
load_previous_weights = False

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

X = T.tensor4('X')
y = T.ivector('y')

if dataset == 'KaggleDR':
    ds = KaggleDR(path_data='data/kaggle_dr/train_JF_BG_' + str(size),
                  filename_targets='data/kaggle_dr/trainLabels_bin.csv',
                  preprocessing=KaggleDR.standard_normalize,
                  require_both_eyes_same_label=True)
    ds_test = KaggleDR(path_data='data/kaggle_dr/test_JF_BG_' + str(size),
                       filename_targets='data/kaggle_dr/'
                                        'retinopathy_solution_bin.csv',
                       preprocessing=KaggleDR.standard_normalize,
                       require_both_eyes_same_label=True)
    idx_train, idx_val = train_test_split(np.arange(ds.n_samples),
                                          stratify=ds.y,
                                          test_size=0.2,
                                          random_state=1234)

n_classes = len(np.unique(ds.y))

print('-' * 40)
print('JFnet layers: ', last_layer)
print('-' * 40)

best_auc = None

###########################################################################
# Setup network

network = JFnet.build_model(width=512, height=512,
                            filename=weights_init, p_conv=p)
network['0'].input_var = X
mean_pooled = lasagne.layers.GlobalPoolLayer(network[last_layer],
                                             pool_function=T.mean)
max_pooled = lasagne.layers.GlobalPoolLayer(network[last_layer],
                                            pool_function=T.max)
network['global_pool'] = lasagne.layers.ConcatLayer([mean_pooled, max_pooled],
                                                    axis=1)
network['logreg'] = DenseLayer(network['global_pool'],
                               num_units=n_classes,
                               nonlinearity=lasagne.nonlinearities.softmax)
l_out = network['logreg']

if load_previous_weights:
    models.load_weights(l_out, 'best_weights.npz')
###########################################################################

###########################################################################
# Theano functions

predictions = lasagne.layers.get_output(l_out, deterministic=False)
loss = lasagne.objectives.categorical_crossentropy(predictions, y)
loss = loss.mean()

l2_penalty = l2_lambda * regularize_network_params(l_out, l2)
l1_penalty = l1_lambda * regularize_layer_params(l_out, l1)
loss = loss + l2_penalty + l1_penalty

predictions_det = lasagne.layers.get_output(l_out, deterministic=True)
loss_det = lasagne.objectives.categorical_crossentropy(predictions_det, y)
loss_det = loss_det.mean()

params = lasagne.layers.get_all_params(network['logreg'], trainable=True)
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
selective_sampler = SelectiveSampler(M=N_DISEASED, y=y_train)

for epoch in range(n_epoch):
    print('-' * 40)
    print('Epoch', epoch)
    print('-' * 40)
    print("Training...")

    if True:
        print('Select all training data...')
        selection = np.arange(len(idx_train))
        np.random.shuffle(selection)
    else:
        print('Prediction on diseased images for selective sampling...')
        progbar = Progbar(len(IDX_HEALTHY))
        probs_neg = np.zeros((len(IDX_HEALTHY),))
        pos = 0
        for Xb, _ in ds.iterate_minibatches(idx_train[IDX_HEALTHY],
                                            batch_size,
                                            shuffle=False):
            prob_neg = pred_iter(Xb)[:, 0]
            probs_neg[pos:pos + Xb.shape[0]] = prob_neg
            progbar.add(Xb.shape[0],
                        values=[("prob_neg", prob_neg.mean())])
            pos += Xb.shape[0]
        selection = selective_sampler.sample(probs_neg=probs_neg,
                                             shuffle=True)

    progbar = Progbar(len(selection))
    loss_train = np.zeros((len(selection),))
    predictions_train = np.zeros((len(selection), 2))
    y_train_sel = ds.y[idx_train[selection]]
    pos = 0
    bs_outer = batch_size * 5
    for Xb_outer, yb_outer in ds.iterate_minibatches(idx_train[selection],
                                                     batch_size=bs_outer,
                                                     shuffle=False):
        augment_data = np.random.randint(2)  # augment 50 % of the data
        if augment_data:
            datagen = datagen_aug
        else:
            datagen = datagen_no_aug

        n_samples_inner = 0
        for Xb, yb in datagen.flow(Xb_outer, yb_outer,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   seed=None,
                                   save_to_dir=None):
            Xb = Xb.astype('float32', copy=False)
            n_samples_inner += Xb.shape[0]
            if n_samples_inner > Xb_outer.shape[0]:
                Warning('Generated more samples than we provided as '
                        'input.')

            if (pos % 40000) == 0:
                params_old = lasagne.layers.get_all_param_values(l_out)
                [loss, predictions] = train_iter[epoch//change_every](Xb, yb)
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
                [loss, predictions] = train_iter[epoch//change_every](Xb, yb)

            loss_train[pos:pos + Xb.shape[0]] = loss
            predictions_train[pos:pos + Xb.shape[0]] = predictions

            progbar.add(Xb.shape[0], values=[("train loss", loss)])
            pos += Xb.shape[0]

            if ((pos//batch_size) % (1000//batch_size)) == 0:
                gc.collect()

            if n_samples_inner == Xb_outer.shape[0]:
                break  # datagen.flow loop is an infinite generator
    print('Training loss: ', loss_train.mean())
    auc_train = roc_auc_score(y_train_sel, predictions_train[:, 1])
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
        print('Saving currently best weights...')
        models.save_weights(l_out, 'best_weights' + last_layer + '.npz')

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



