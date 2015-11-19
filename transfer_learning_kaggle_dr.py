from __future__ import print_function, division
from lasagne.layers import DenseLayer, NonlinearityLayer
from lasagne.nonlinearities import softmax

import click
from util import Progplot

@click.command()
@click.option('--config_file', default='config.json', show_default=True,
              help="(JSON) configuration file.")
def main(config_file):
    """Perform transfer learning on Kaggle's Diabetic Retinopathy competition.

    \b
    Parameters
    ----------
    \b
    config_file: json file
        content of example.json:
        {
            // Path to data (labels, image folders or feature activations)
            "path": "/home/cl/data_kaggle_dr/o_O" ,
            "batch_size": 2,
            // Number of epochs for training and validation
            "n_epoch": 10,
            // Learning rate used throughout "n_epochs":
            "learning_rate": 1e-4,
            // Fraction of training data used for training and validation
            // respectively:
            "train_size": 0.9,
            "val_size": 0.1,
            // Fraction of test data used for testing:
            "test_size": 1.0,
            // File with initial weights under "path":
            "weights_init": "vgg19.pkl",
            // File to dump weights to under "path":
            "weights_dump": "weights_dump.npz",
            // File with training labels under "path". "val_frac" of those is
            // taken for validation:
            "labels_train": "trainLabels.csv",
            // File with test labels under "path":
            "labels_test": "retinopathy_solution.csv",
            // Number of classes to adopt the original architecture to:
            "n_classes": 5
            // Priors (if uniform they have no influence):
            "val_priors": [0.2, 0.2, 0.2, 0.2, 0.2],
            "test_priors": [0.2, 0.2, 0.2, 0.2, 0.2]
        }

    """

    import os
    import time
    import numpy as np
    import theano
    import theano.tensor as T
    import lasagne
    from keras.utils.generic_utils import Progbar
    import json

    import models
    from datasets import KaggleDR
    from util import quadratic_weighted_kappa

    ###########################################################################
    # Parse configuration
    ###########################################################################
    with open(config_file) as json_file:
        config = json.load(json_file)

    path = config['path']
    batch_size = config['batch_size']
    n_epoch = config['n_epoch']
    learning_rate = config['learning_rate']
    train_size = config['train_size'] # will be used once
    # StratifiedShuffleSplit is used as implementation of train_test_split
    val_size = config['val_size']
    test_size = config['test_size']
    weights_init = config['weights_init']
    weights_dump = config['weights_dump']
    labels_train = config['labels_train']
    labels_test = config['labels_test']
    n_classes = config['n_classes']
    val_priors = np.array(config['val_priors'], dtype=theano.config.floatX)
    test_priors = np.array(config['test_priors'], dtype=theano.config.floatX)
    assert len(val_priors) == len(test_priors) == n_classes, \
        'Mismatch between number of classes and priors'
    ###########################################################################

    # Initialize symbolic variables
    X = T.tensor4('X')
    y = T.ivector('y')
    y_pred = T.ivector('y_pred')

    # Initialize DAOs (data access objects)
    kdr = KaggleDR(path_data=os.path.join(path, "train"),
                   filename_targets=os.path.join(path, labels_train))
    kdr_test = KaggleDR(path_data=os.path.join(path, "test"),
                        filename_targets=os.path.join(path, labels_test))

    network = models.vgg19(input_var=X,
                           filename=os.path.join(path, weights_init), p=0.5)

    network['fc8'] = DenseLayer(network['dropout2'], num_units=n_classes,
                                nonlinearity=None)
    network['prob'] = NonlinearityLayer(network['fc8'], softmax)

    l_out = network['prob']

    # Scalar loss expression to be minimized during training:
    train_posteriors = lasagne.layers.get_output(l_out)
    loss = lasagne.objectives.categorical_crossentropy(train_posteriors, y)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(l_out, trainable=True)

    updates = lasagne.updates.nesterov_momentum(loss, params,
                                                learning_rate=learning_rate,
                                                momentum=0.9)

    # Scalar loss expression for validation - only necessary if
    # stochasticity such
    # as dropout is involved during training
    val_posteriors = lasagne.layers.get_output(l_out, deterministic=True)
    val_posteriors_bal = val_posteriors * val_priors / \
        T.sum(val_posteriors * val_priors, axis=1).dimshuffle(0, 'x')
    val_y = T.argmax(val_posteriors_bal, axis=1)
    val_loss = lasagne.objectives.categorical_crossentropy(val_posteriors_bal,
                                                           y)
    val_loss = val_loss.mean()

    # Scalar loss expression for testing - only necessary if stochasticity such
    # as dropout is involved during training
    test_posteriors = lasagne.layers.get_output(l_out, deterministic=True)
    test_posteriors_bal = test_posteriors * test_priors / \
        T.sum(test_posteriors * test_priors, axis=1).dimshuffle(0, 'x')
    test_labels = T.argmax(test_posteriors_bal, axis=1)
    test_loss = lasagne.objectives.categorical_crossentropy(
        test_posteriors_bal, y)
    test_loss = test_loss.mean()

    # Function for one training step on minibatch
    train_fn = theano.function([X, y], loss, updates=updates)

    val_fn = theano.function([X, y], [val_loss, val_y])
    acc = T.mean(T.eq(y_pred, y), dtype=theano.config.floatX)
    acc_fn = theano.function([y_pred, y], [acc])

    # Function to compute val loss and accuracy
    test_fn = theano.function([X, y], [test_loss, test_labels])

    idx_train, idx_val = kdr.train_test_split(val_size, shuffle=True)
    idx_test = np.arange(min(test_size*kdr_test.n_samples,
                             kdr_test.n_samples), dtype=np.int32)
    # idx_train, idx_val, _ = kdr.generate_indices(train_size, val_size,
    #                                              test_size)

    ###########################################################################
    # Setup progression plot
    ###########################################################################
    progplot = Progplot(n_epoch,
                        "epochs (batch_size "+str(batch_size)+")")
    ###########################################################################

    ###########################################################################
    # Training
    ###########################################################################
    start_time = time.time()

    train_loss = np.zeros(len(idx_train))
    val_loss = np.zeros(len(idx_val))
    val_y_pred = np.zeros(len(idx_val), dtype=np.int32)
    val_y_hum = np.zeros(len(idx_val), dtype=np.int32)

    for epoch in range(n_epoch):
        print('-'*40)
        print('Epoch', epoch)
        print('-'*40)
        print("Training...")

        #######################################################################
        # Training loop
        #######################################################################
        current = 0
        train_loss[:] = 0
        progbar = Progbar(len(idx_train))
        for batch in kdr.iterate_minibatches(idx_train, batch_size,
                                             shuffle=True):
            X_batch, y_hum = batch
            loss = train_fn(X_batch, y_hum)
            progbar.add(X_batch.shape[0], values=[("train loss", loss)])
            train_loss[current:current+X_batch.shape[0]] = \
                [loss]*X_batch.shape[0]
            current += X_batch.shape[0]

        #######################################################################
        # Validation loop
        #######################################################################
        current = 0
        val_loss[:] = 0
        val_y_pred[:] = 0
        val_y_hum[:] = 0
        progbar = Progbar(len(idx_val))
        for batch in kdr.iterate_minibatches(idx_val, batch_size,
                                             shuffle=False):
            X_batch, y_hum = batch
            loss, y_pred = val_fn(X_batch, y_hum)
            progbar.add(X_batch.shape[0], values=[("val. loss", loss)])
            val_loss[current:current+X_batch.shape[0]] = \
                [loss]*X_batch.shape[0]
            val_y_pred[current:current+X_batch.shape[0]] = y_pred
            val_y_hum[current:current+X_batch.shape[0]] = y_hum
            current += X_batch.shape[0]

        val_acc = acc_fn(val_y_hum, val_y_pred)[0]
        val_kp = quadratic_weighted_kappa(val_y_hum, val_y_pred, n_classes)
        print("Validation accuracy: ", val_acc)
        print("Validation kappa: ", val_kp)
        progplot.add(values=[("train loss", np.mean(train_loss)),
                             ("val. loss", np.mean(val_loss)),
                             ("val. accuracy", val_acc),
                             ("val. kappa", val_kp)])

    print("Training took {:.3g} sec.".format(time.time() - start_time))

    del kdr

    ###########################################################################
    # Testing
    ###########################################################################

    test_loss = np.zeros(kdr_test.n_samples)
    test_y_pred = np.zeros(kdr_test.n_samples, dtype=np.int32)
    test_y_hum = np.zeros(kdr_test.n_samples, dtype=np.int32)

    print("Testing...")
    current = 0
    test_loss[:] = 0
    test_y_pred[:] = 0
    test_y_hum[:] = 0

    progbar = Progbar(kdr_test.n_samples)
    for batch in kdr_test.iterate_minibatches(idx_test,
                                              batch_size, shuffle=False):
        X_batch, y_hum = batch
        loss, y_pred = test_fn(X_batch, y_hum)
        progbar.add(X_batch.shape[0], values=[("test loss", loss)])
        test_loss[current:current+X_batch.shape[0]] = \
            [loss]*X_batch.shape[0]
        test_y_pred[current:current+X_batch.shape[0]] = y_pred
        test_y_hum[current:current+X_batch.shape[0]] = y_hum
        current += X_batch.shape[0]

    test_acc = acc_fn(test_y_hum, test_y_pred)[0]
    test_kp = quadratic_weighted_kappa(test_y_hum, test_y_pred, n_classes)
    print("Test accuracy: ", test_acc)
    print("Test kappa: ", test_kp)

    progplot.save()
    np.savez(os.path.join(path, weights_dump),
             *lasagne.layers.get_all_param_values(l_out))

if __name__ == '__main__':
    main()
