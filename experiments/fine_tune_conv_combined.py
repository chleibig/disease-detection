from __future__ import print_function, division

from collections import defaultdict
import itertools
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
from util import Progplot

batch_size = 2
n_epoch = 30
lr_logreg = 0.005
lr_conv = 0.005
lr_schedule = {0: 0.005, 1: 0.001, 2: 0.0005, 3: 0.0001, 4: 0.00005, 5: 0.000001}
change_every = 5
l2_lambda = 0.001  # entire network
l1_lambda = 0.001  # only last layer
size = 512
dataset = 'KaggleDR'

weights_init = 'models/vgg19/vgg19_normalized.pkl'
load_previous_weights = False
best_auc = 0.0

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
                  filename_targets='data/kaggle_dr/trainLabels_01vs234.csv',
                  preprocessing=KaggleDR.standard_normalize)
    ds_test = KaggleDR(path_data='data/kaggle_dr/test_JF_BG_' + str(size),
                       filename_targets='data/kaggle_dr/'
                                        'retinopathy_solution_01vs234.csv',
                       preprocessing=KaggleDR.standard_normalize)

if dataset == 'optretina':
    ds = OptRetina(path_data='data/optretina/data_JF_' + str(size),
                   filename_targets='data/optretina/OR_diseased_labels.csv',
                   preprocessing=KaggleDR.standard_normalize,
                   exclude_path='data/optretina/data_JF_' + str(size) +
                                '_exclude')

untie_biases = defaultdict(lambda: False, {512: True})

network = models.vgg19(input_var=X, height=size, width=size)
models.load_weights(network['pool5'], weights_init)

n_classes = len(np.unique(ds.y))

######################################################
# Construct view on network
######################################################
# TODO: write an expression layer that computes the
#       correlation between feature maps

selection = ['pool1', 'pool2', 'pool3', 'pool4', 'pool5']

mean_pooled_features = [lasagne.layers.GlobalPoolLayer(network[k],
                                                       pool_function=T.mean)
                        for k in selection]

network['conv_combined'] = lasagne.layers.ConcatLayer(pooled_features, axis=1)


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


#def build_updates(network, lr_conv, lr_logreg):
#    """Build updates with different learning rates for the conv stack
#       and the logistic regression layer
#    """
#
#    params_conv = lasagne.layers.get_all_params(network['conv_combined'],
#                                                trainable=True)
#    updates_conv = lasagne.updates.sgd(loss, params_conv,
#                                       learning_rate=lr_conv)
#    params_logreg = network['logreg'].get_params(trainable=True)
#    updates_logreg = lasagne.updates.sgd(loss, params_logreg,
#                                         learning_rate=lr_logreg)
#    updates = updates_conv
#    for k, v in updates_logreg.items():
#        updates[k] = v
#
#    return lasagne.updates.apply_nesterov_momentum(updates, momentum=0.9)


predictions_det = lasagne.layers.get_output(l_out, deterministic=True)
loss_det = lasagne.objectives.categorical_crossentropy(predictions_det, y)
loss_det = loss_det.mean()

#updates = build_updates(network, lr_conv, lr_logreg)
params = lasagne.layers.get_all_params(network['logreg'], 
                                       trainable=True)
#updates = lasagne.updates.nesterov_momentum(loss, params,
#                                            learning_rate=lr_logreg)
train_iter = {k: theano.function([X, y], [loss, predictions],
                                 updates=lasagne.updates.nesterov_momentum(
                                    loss, params,
                                    learning_rate=lr_schedule[k]))
              for k in lr_schedule.keys()}

#train_iter = theano.function([X, y], [loss, predictions], updates=updates)
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
progplot = Progplot(n_epoch, "epochs (batch_size " + str(batch_size) + ")",
                    names=['loss (train)', 'loss (val.)',
                           'AUC (train)', 'AUC (val.)'])

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
            progbar.add(Xb.shape[0], values=[("prob_neg", prob_neg.mean())])
            pos += Xb.shape[0]
        selection = selective_sampler.sample(probs_neg=probs_neg, shuffle=True)

    progbar = Progbar(len(selection))
    loss_train = np.zeros((len(selection),))
    predictions_train = np.zeros((len(selection), 2))
    y_train_sel = ds.y[idx_train[selection]]
    pos = 0
    bs_outer = batch_size * 10
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
                Warning('Generated more samples than we provided as input.')

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

            if n_samples_inner == Xb_outer.shape[0]:
                break # datagen.flow loop is an infinite generator

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
