from __future__ import print_function, division

import click
from util import Progplot


@click.command()
@click.option('--path', default=None, show_default=True,
              help="Path to trainLabels.csv and feature_activations.npy or training images.")
@click.option('--batch_size', default=2, show_default=True)
@click.option('--n_epoch', default=100, show_default=True,
              help="Number of epochs for training and validation.")
@click.option('--split', nargs=3, type=float, default=(0.9, 0.1, 0.0),
              help="Fraction of samples to be used for train, val and test "
                   "respectively.")
@click.option('--weights', default='vgg19.pkl', show_default=True,
              help="Filename for pre-trained weights.")
@click.option('--model_file', default='model.npz', show_default=True,
              help="Filename for model dump.")
def main(path, batch_size, n_epoch, split, weights, model_file):
    """Perform transfer learning on Kaggle's Diabetic Retinopathy competition.
    """

    import os
    import time
    import numpy as np
    import theano
    import theano.tensor as T
    import lasagne
    from keras.utils.generic_utils import Progbar

    import models
    from datasets import KaggleDR
    from util import quadratic_weighted_kappa

    ###########################################################################
    # Parameters - once the API for this code is clear, move these to a config
    # file
    ###########################################################################
    cnf = {
        'labels_train': 'trainLabels.csv',
        'labels_train_val': 'trainLabels_aug.csv',
        'features_train_val': 'feature_activations_train_aug.npy',
        'labels_test': 'retinopathy_solution.csv',
        'features_test': 'feature_activations_test.npy',
        'val_priors': np.array(5*[1/5.], dtype=theano.config.floatX),
        'test_priors': np.array(5*[1/5.], dtype=theano.config.floatX),
        # 'test_priors': np.array([0.73478335,  0.06954962,  0.15065763,
        #                          0.02485338,  0.02015601],
        #                         dtype=theano.config.floatX)
        'trained_weights': 'weights',
        'val_frac': split[1],
        'n_classes': 5
    }

    X = T.tensor4('X')
    y = T.ivector('y')
    y_pred = T.ivector('y_pred')

    ###########################################################################
    # Load features obtained via forward pass through pretrained network
    ###########################################################################

    kdr = KaggleDR(path_data=os.path.join(path, "train"),
                   filename_targets=os.path.join(path,
                                                 cnf['labels_train']))

    kdr_test = KaggleDR(path_data=os.path.join(path, "test"),
                        filename_targets=os.path.join(path,
                                                      cnf['labels_test']))

    ###########################################################################
    # Transfer Learning: Train logistic regression on extracted features
    ###########################################################################
    network = models.vgg19(input_var=X,
                           filename=os.path.join(path, cnf['trained_weights']))
    l_out = network['prob']

    # Scalar loss expression to be minimized during training:
    train_posteriors = lasagne.layers.get_output(l_out)
    loss = lasagne.objectives.categorical_crossentropy(train_posteriors, y)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(l_out, trainable=True)

    updates = lasagne.updates.nesterov_momentum(loss, params,
                                                learning_rate=1e-4,
                                                momentum=0.9)

    val_posteriors = lasagne.layers.get_output(l_out)
    val_posteriors_bal = val_posteriors * cnf['val_priors'] / \
        T.sum(val_posteriors * cnf['val_priors'], axis=1).dimshuffle(0, 'x')
    val_y = T.argmax(val_posteriors_bal, axis=1)
    val_loss = lasagne.objectives.categorical_crossentropy(val_posteriors_bal,
                                                           y)
    val_loss = val_loss.mean()

    # Scalar loss expression for testing - only necessary if stochasticity such
    # as dropout is involved during training
    test_posteriors = lasagne.layers.get_output(l_out, deterministic=True)
    test_posteriors_bal = test_posteriors * cnf['test_priors'] / \
        T.sum(test_posteriors * cnf['test_priors'], axis=1).dimshuffle(0, 'x')
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

    idx_train, idx_val = kdr.train_test_split(cnf['val_frac'], shuffle=True)
    idx_test = np.arange(kdr_test.n_samples)
    # idx_train, idx_val, _ = kdr.generate_indices(0.01, 0.005, 0.005)
    # idx_test = np.arange(100)

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
        val_kp = quadratic_weighted_kappa(val_y_hum, val_y_pred,
                                          cnf['n_classes'])
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
    test_kp = quadratic_weighted_kappa(test_y_hum, test_y_pred,
                                       cnf['n_classes'])
    print("Test accuracy: ", test_acc)
    print("Test kappa: ", test_kp)

    progplot.save()
    np.savez(os.path.join(path, model_file),
             *lasagne.layers.get_all_param_values(l_out))

if __name__ == '__main__':
    main()
