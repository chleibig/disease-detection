from __future__ import print_function, division

import click
from util import Progplot


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
        'val_priors': np.array(5*[1/5.], dtype=theano.config.floatX),
        'test_priors': np.array(5*[1/5.], dtype=theano.config.floatX)
        # 'test_priors': np.array([0.73478335,  0.06954962,  0.15065763,
        #                          0.02485338,  0.02015601],
        #                         dtype=theano.config.floatX)
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
    del n_samples

    kdr_test = KaggleDR(filename_targets=os.path.join(path,
                                                      cnf['labels_test']))
    kdr_test.X = floatX(np.load(os.path.join(path,
                                        cnf['features_test'])))
    n_samples = kdr_test.X.shape[0]
    # assert that we have features for all labels stored in kdr_test.y
    assert n_samples == kdr_test.n_samples
    kdr_test.indices_in_X = np.arange(n_samples)

    ###########################################################################
    # Transfer Learning: Train logistic regression on extracted features
    ###########################################################################
    l_in = InputLayer((batch_size, n_features), input_var=X)
    l_hidden = DenseLayer(l_in, num_units=5, nonlinearity=None)
    l_out = NonlinearityLayer(l_hidden, lasagne.nonlinearities.softmax)

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
    val_labels = T.argmax(val_posteriors_bal, axis=1)
    val_loss = lasagne.objectives.categorical_crossentropy(val_posteriors_bal,
                                                           y)
    val_loss = val_loss.mean()
    val_acc = T.mean(T.eq(val_labels, y), dtype=theano.config.floatX)

    # Scalar loss expression for testing - only necessary if stochasticity such
    # as dropout is involved during training
    test_posteriors = lasagne.layers.get_output(l_out, deterministic=True)
    test_posteriors_bal = test_posteriors * cnf['test_priors'] / \
        T.sum(test_posteriors * cnf['test_priors'], axis=1).dimshuffle(0, 'x')
    test_labels = T.argmax(test_posteriors_bal, axis=1)
    test_loss = lasagne.objectives.categorical_crossentropy(
        test_posteriors_bal, y)
    test_loss = test_loss.mean()
    # Theano expression for classification accuracy
    test_acc = T.mean(T.eq(test_labels, y),
                      dtype=theano.config.floatX)

    # Function for one training step on minibatch
    train_fn = theano.function([X, y], loss, updates=updates)

    val_fn = theano.function([X, y], [val_loss, val_acc, val_labels])

    # Function to compute val loss and accuracy
    test_fn = theano.function([X, y], [test_loss, test_acc, test_labels])

    idx_train, idx_val, _ = kdr.generate_indices(*split, shuffle=True)

    print('Validation accuracy before training:')
    progbar = Progbar(len(idx_val))
    for batch in kdr.iterate_minibatches(idx_val, batch_size,
                                         shuffle=False):
        inputs, targets = batch
        loss, acc, _ = val_fn(inputs, targets)
        progbar.add(inputs.shape[0], values=[("val. loss", loss),
                                             ("val. accuracy", acc)])

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
    test_loss = np.zeros(kdr_test.n_samples)
    test_acc = np.zeros(kdr_test.n_samples)
    test_kp = np.zeros(kdr_test.n_samples)

    for epoch in range(n_epoch):
        print('-'*40)
        print('Epoch', epoch)
        print('-'*40)
        print("Training...")

        current = 0
        train_loss[:] = 0
        progbar = Progbar(len(idx_train))
        for batch in kdr.iterate_minibatches(idx_train, batch_size,
                                             shuffle=True):
            inputs, targets = batch
            loss = train_fn(inputs, targets)
            progbar.add(inputs.shape[0], values=[("train loss", loss)])
            train_loss[current:current+inputs.shape[0]] = \
                [loss]*inputs.shape[0]
            current += inputs.shape[0]

        # progbar = Progbar(len(idx_val))
        # for batch in kdr.iterate_minibatches(idx_val, batch_size,
        #                                      shuffle=False):
        #     inputs, targets = batch
        #     loss, acc, labels = val_fn(inputs, targets)
        #     kp = quadratic_weighted_kappa(targets, labels, 5)
        #     progbar.add(inputs.shape[0], values=[("val. loss", loss),
        #                                          ("val. accuracy",
        #                                           acc),
        #                                          ("val. kappa", kp)])
        #     progplot.add(values=[("val. loss", loss),
        #                          ("val. accuracy", acc),
        #                          ("val. kappa", kp)])

        print("Testing...")
        current = 0
        test_loss[:] = 0
        test_acc[:] = 0
        test_kp[:] = 0
        progbar = Progbar(kdr_test.n_samples)
        for batch in kdr_test.iterate_minibatches(kdr_test.indices_in_X,
                                                  batch_size, shuffle=False):
            inputs, targets = batch
            loss, acc, labels = test_fn(inputs, targets)
            kp = quadratic_weighted_kappa(targets, labels, 5)
            progbar.add(inputs.shape[0], values=[("test loss", loss),
                                                 ("test accuracy", acc),
                                                 ("test kappa", kp)])
            test_loss[current:current+inputs.shape[0]] = \
                [loss]*inputs.shape[0]
            test_acc[current:current+inputs.shape[0]] = \
                [acc]*inputs.shape[0]
            test_kp[current:current+inputs.shape[0]] = \
                [kp]*inputs.shape[0]
            current += inputs.shape[0]

        progplot.add(values=[("train loss", np.mean(train_loss)),
                             ("test loss", np.mean(test_loss)),
                             ("test accuracy", np.mean(test_acc)),
                             ("test kappa", np.mean(test_kp))])

        print("Training took {:.3g} sec.".format(time.time() - start_time))

    progplot.save()
    np.savez(os.path.join(path, model_file),
             *lasagne.layers.get_all_param_values(l_out))

if __name__ == '__main__':
    main()
