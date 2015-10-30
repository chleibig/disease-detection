from __future__ import print_function

import click


@click.command()
@click.option('--path', default=None, show_default=True,
              help="Path to trainLabels.csv and feature_activations.npy.")
@click.option('--batch_size', default=2, show_default=True)
@click.option('--n_epoch', default=100, show_default=True,
              help="Number of epochs for training and validation.")
@click.option('--split', nargs=3, type=float, default=(0.9, 0.1, 0.0),
              help="Fraction of samples to be used for train, val and test "
                   "respectively.")
@click.option('--model_file', default='model.npz', show_default=True,
              help="Filename for model dump.")
def main(path, batch_size, n_epoch, split, model_file):
    """Perform transfer learning on Kaggle's Diabetic Retinopathy competition.
    """

    import os
    import time
    import numpy as np
    import theano
    import theano.tensor as T
    import lasagne
    from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
    from lasagne.utils import floatX
    from keras.utils.generic_utils import Progbar

    from datasets import KaggleDR
    from util import quadratic_weighted_kappa

    ###########################################################################
    # Parameters - once the API for this code is clear, move these to a config
    # file
    ###########################################################################
    cnf = {
        'labels_train_val': 'trainLabels_aug.csv',
        'features_train_val': 'feature_activations_train_aug.npy',
        'labels_test': 'retinopathy_solution.csv',
        'features_test': 'feature_activations_test.npy',
        'priors': np.array([0.73478335,  0.06954962,  0.15065763,  0.02485338,
                            0.02015601], dtype=theano.config.floatX)
    }

    X = T.matrix('X')
    y = T.ivector('y')

    ###########################################################################
    # Load features obtained via forward pass through pretrained network
    ###########################################################################
    kdr = KaggleDR(filename_targets=os.path.join(path,
                                                 cnf['labels_train_val']))
    kdr.X = floatX(np.load(os.path.join(path,
                                        cnf['features_train_val'])))

    n_samples, n_features = kdr.X.shape
    # assert that we have features for all labels stored in kdr.y
    assert n_samples == kdr.n_samples
    kdr.indices_in_X = np.arange(n_samples)

    ###########################################################################
    # Transfer Learning: Train logistic regression on extracted features
    ###########################################################################
    l_in = InputLayer((batch_size, n_features), input_var=X)
    l_hidden = DenseLayer(l_in, num_units=5, nonlinearity=None)
    l_out = NonlinearityLayer(l_hidden, lasagne.nonlinearities.softmax)

    # Scalar loss expression to be minimized during training:
    y_pred = lasagne.layers.get_output(l_out)
    loss = lasagne.objectives.categorical_crossentropy(y_pred, y)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(l_out, trainable=True)

    updates = lasagne.updates.nesterov_momentum(loss, params,
                                                learning_rate=1e-4,
                                                momentum=0.9)

    # Scalar loss expression for testing - only necessary if stochasticity such
    # as dropout is involved during training
    test_y_pred = lasagne.layers.get_output(l_out, deterministic=True)
    y_pred_labels = T.argmax(test_y_pred*cnf['priors'], axis=1)
    test_loss = lasagne.objectives.categorical_crossentropy(test_y_pred, y)
    test_loss = test_loss.mean()
    # Theano expression for classification accuracy
    test_acc = T.mean(T.eq(y_pred_labels, y),
                      dtype=theano.config.floatX)

    # Function for one training step on minibatch
    train_fn = theano.function([X, y], loss, updates=updates)

    # Function to compute val loss and accuracy
    val_fn = theano.function([X, y], [test_loss, test_acc, y_pred_labels])

    idx_train, idx_val, _ = kdr.generate_indices(*split, shuffle=True)

    print('Validation accuracy before training:')
    progbar = Progbar(len(idx_val))
    for batch in kdr.iterate_minibatches(idx_val, batch_size,
                                         shuffle=False):
        inputs, targets = batch
        err, acc, _ = val_fn(inputs, targets)
        progbar.add(inputs.shape[0], values=[("validation loss", err),
                                             ("validation accuracy", acc)])



    ###########################################################################
    # Training
    ###########################################################################
    start_time = time.time()
    for epoch in range(n_epoch):
        print('-'*40)
        print('Epoch', epoch)
        print('-'*40)
        print("Training...")

        progbar = Progbar(len(idx_train))
        for batch in kdr.iterate_minibatches(idx_train, batch_size,
                                             shuffle=True):
            inputs, targets = batch
            err = train_fn(inputs, targets)
            progbar.add(inputs.shape[0], values=[("train loss", err)])

        progbar = Progbar(len(idx_val))
        for batch in kdr.iterate_minibatches(idx_val, batch_size,
                                             shuffle=False):
            inputs, targets = batch
            err, acc, predicted = val_fn(inputs, targets)
            kp = quadratic_weighted_kappa(targets, predicted, 5)
            progbar.add(inputs.shape[0], values=[("validation loss", err),
                                                 ("validation accuracy",
                                                  acc),
                                                 ("validation kappa", kp)])

    print("Training took {:.3g} sec.".format(time.time() - start_time))

    del kdr

    ###########################################################################
    # Testing
    ###########################################################################
    print("Testing...")
    kdr = KaggleDR(filename_targets=os.path.join(path,
                                                 cnf['labels_test']))
    kdr.X = floatX(np.load(os.path.join(path,
                                        cnf['features_test'])))

    n_samples = kdr.X.shape[0]
    # assert that we have features for all labels stored in kdr.y
    assert n_samples == kdr.n_samples
    kdr.indices_in_X = np.arange(n_samples)

    progbar = Progbar(kdr.n_samples)
    for batch in kdr.iterate_minibatches(kdr.indices_in_X, batch_size,
                                         shuffle=False):
        inputs, targets = batch
        err, acc, predicted = val_fn(inputs, targets)
        kp = quadratic_weighted_kappa(targets, predicted, 5)
        progbar.add(inputs.shape[0], values=[("test loss", err),
                                             ("test accuracy", acc),
                                             ("test kappa", kp)])

    np.savez(os.path.join(path, model_file),
             *lasagne.layers.get_all_param_values(l_out))

if __name__ == '__main__':
    main()
