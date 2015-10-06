from __future__ import print_function

import time

import theano
import theano.tensor as T
import lasagne
from lasagne.layers import DenseLayer, NonlinearityLayer
from lasagne.nonlinearities import softmax


from modelzoo import vgg19
from datasets import KaggleDR

# TODO: - resolve memory issue -> check theano docu on python memory management
# TODO: - set trainable tag of convnet params to false
# --> in build_model() set tags trainable=False!
# TODO: - use both left and right eye data for each patient

###############################################################################
# Parameters
###############################################################################
path_data = '/home/cl/Downloads/data_kaggle_dr/train'
filename_targets = '/home/cl/Downloads/data_kaggle_dr/trainLabels.csv'
n_epoch = 500
batch_size = 1
train_frac = 0.08
val_frac = 0.01
test_frac = 0.01

# Prepare theano variables for inputs and targets
input_var = T.tensor4('inputs')
target_var = T.ivector('targets')

###############################################################################
# Setup pretrained network: Here VGG19
###############################################################################
network = vgg19.build_model(load_weights=True)
###############################################################################
# Transfer Learning: Adjust output layer to new task
###############################################################################
# network['fc8'] = DenseLayer(network['fc7'], num_units=5, nonlinearity=None)
# network['prob'] = NonlinearityLayer(network['fc8'], softmax)
output_layer = network['prob']

# plug symbolic input to network
network['input'].input_var = input_var

# later on this will be filled with values from the actual dataset:
kdr = KaggleDR(path_data=path_data,
               filename_targets=filename_targets)

# Scalar loss expression to be minimized during training:
prediction = lasagne.layers.get_output(output_layer)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()

params = lasagne.layers.get_all_params(output_layer, trainable=True)
updates = lasagne.updates.nesterov_momentum(loss, params,
                                            learning_rate=0.01, momentum=0.9)

# Scalar loss expression for testing - only necessary if stochasticity such
# as dropout is involved during training
test_prediction = lasagne.layers.get_output(output_layer, deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                        target_var)
test_loss = test_loss.mean()
# Theano expression for classification accuracy
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                  dtype=theano.config.floatX)

# Function for one training step on minibatch
train_fn = theano.function([input_var, target_var], loss, updates=updates)

# Function to compute val loss and accuracy
val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

train_indices, val_indices, test_indices = kdr.generate_indices(train_frac,
                                                                val_frac,
                                                                test_frac)

###############################################################################
# Training
###############################################################################
print("Starting training...")
for epoch in range(n_epoch):
    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_batches = 0
    start_time = time.time()
    for batch in kdr.iterate_minibatches(train_indices, batch_size):
        inputs, targets = batch
        train_err += train_fn(inputs, targets)
        train_batches += 1

    # And a full pass over the validation data:
    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch in kdr.iterate_minibatches(val_indices, batch_size,
                                         shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1

    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, n_epoch, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(
        val_acc / val_batches * 100))

###############################################################################
# Testing
###############################################################################
test_err = 0
test_acc = 0
test_batches = 0
for batch in kdr.iterate_minibatches(test_indices, batch_size, shuffle=False):
    inputs, targets = batch
    err, acc = val_fn(inputs, targets)
    test_err += err
    test_acc += acc
    test_batches += 1
print("Final results:")
print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
print("  test accuracy:\t\t{:.2f} %".format(
    test_acc / test_batches * 100))
