from __future__ import print_function

import click


@click.command()
@click.option('--path', default=None, show_default=True,
              help="Path to trainLabels.csv and feature_activations.npy.")
@click.option('--batch_size', default=2, show_default=True)
@click.option('--n_epoch', default=100, show_default=True,
              help="Number of epochs for training and validation.")
@click.option('--split', nargs=3, type=float, default=(0.8, 0.1, 0.1),
              help="Fraction of samples to be used for train, val and test "
                   "respectively.")
@click.option('--model_file', default='model.npz', show_default=True,
              help="Filename for model dump.")
def main(path, batch_size, n_epoch, split, model_file):
    """Perform transfer learning on Kaggle's Diabetic Retinopathy competition.
    """

    import os
    import numpy as np
    import theano
    import theano.tensor as T
    import lasagne
    from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
    from lasagne.utils import floatX
    from keras.utils.generic_utils import Progbar

    from datasets import KaggleDR

    X = T.matrix('X')
    y = T.ivector('y')

    ###########################################################################
    # Load features obtained via forward pass through pretrained network
    ###########################################################################
    kdr = KaggleDR(filename_targets=os.path.join(path, 'trainLabels_aug.csv'))

    # kdr.X = np.memmap(os.path.join(path,
    #                                'feature_activations_train_aug.npy'),
    #                   dtype=theano.config.floatX, mode='r',
    #                   shape=(kdr.n_samples, 3, 224, 224))
    kdr.X = floatX(np.load(os.path.join(path,
                                     'feature_activations_train_aug.npy')))

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
                                                learning_rate=1e-6,
                                                momentum=0.9)

    # Scalar loss expression for testing - only necessary if stochasticity such
    # as dropout is involved during training
    test_y_pred = lasagne.layers.get_output(l_out, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_y_pred, y)
    test_loss = test_loss.mean()
    # Theano expression for classification accuracy
    test_acc = T.mean(T.eq(T.argmax(test_y_pred, axis=1), y),
                      dtype=theano.config.floatX)

    # Function for one training step on minibatch
    train_fn = theano.function([X, y], loss, updates=updates)

    # Function to compute val loss and accuracy
    val_fn = theano.function([X, y], [test_loss, test_acc])

    idx_train, idx_val, idx_test = kdr.generate_indices(*split, shuffle=True)

    ###########################################################################
    # Training
    ###########################################################################
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
            err, acc = val_fn(inputs, targets)
            progbar.add(inputs.shape[0], values=[("validation loss", err),
                                                 ("validation accuracy", acc)])

    ###########################################################################
    # Testing
    ###########################################################################
    progbar = Progbar(len(idx_test))
    for batch in kdr.iterate_minibatches(idx_test, batch_size,
                                         shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        progbar.add(inputs.shape[0], values=[("test loss", err),
                                             ("test accuracy", acc)])

    np.savez(os.path.join(path, model_file),
             *lasagne.layers.get_all_param_values(l_out))

if __name__ == '__main__':
    main()
