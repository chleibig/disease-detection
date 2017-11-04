"""Performance of original JFnet on Kaggle's level prediction task"""
from __future__ import print_function
import cPickle as pickle
import pandas as pd
import numpy as np
import scipy

from util import quadratic_weighted_kappa


def load_labels():
    df_train = pd.read_csv('data/kaggle_dr/trainLabels.csv')
    df_test = pd.read_csv('data/kaggle_dr/retinopathy_solution.csv')
    y_train = df_train.level.values
    y_test = df_test.level.values
    return df_train, df_test, y_train, y_test


def load_softmax_outputs():
    # train predictions obtained with e.g.
    # scripts/predict.py, commit 1234ce0
    with open('data/processed/1234ce0_jfnet_100MCdropout_KaggleDR_train.pkl',
              'rb') as h:
        pred_train = pickle.load(h)
    softmax_train = pred_train['det_out']
    # test predictions obtained with e.g.
    # scripts/predict.py, commit 40d8265
    with open('data/processed/40d8265_jfnet_100MCdropout_KaggleDR_test.pkl',
              'rb') as h:
        pred_test = pickle.load(h)
    softmax_test = pred_test['det_out']
    return softmax_train, softmax_test


def optimal_thresholds(y_train, softmax_train):

    def neg_kappa_from_probs(thresholds, labels_true, probs):
        n_samples, n_classes = probs.shape
        assert len(labels_true) == n_samples
        assert n_classes == 5
        t1, t2, t3, t4 = thresholds
        scores = probs[:, 1] + probs[:, 2] * 2 + \
            probs[:, 3] * 3 + probs[:, 4] * 4
        # The following could be a one-liner with np.digitize, but this would
        # require to introduce constraints on the ordering of the thresholds in
        # turn
        labels_pred = 999 * np.ones_like(labels_true)
        labels_pred[scores <= t1] = 0
        labels_pred[(t1 < scores) & (scores <= t2)] = 1
        labels_pred[(t2 < scores) & (scores <= t3)] = 2
        labels_pred[(t3 < scores) & (scores <= t4)] = 3
        labels_pred[t4 < scores] = 4
        assert labels_pred.max() <= 4

        return -quadratic_weighted_kappa(labels_true, labels_pred, n_classes)

    print('Determining optimal thresholds...')
    x0 = np.array([0.5, 1.5, 2.5, 3.5])
    res = scipy.optimize.minimize(neg_kappa_from_probs, x0,
                                  args=(y_train, softmax_train),
                                  method='Powell')
    return res['x']


def probs_to_labels(probs,
                    thresholds=[0.52909613, 1.6936609,
                                2.75167356, 3.58791778]):
    scores = np.dot(probs, np.arange(5))
    return np.digitize(scores, thresholds)


def main(compute_optimal_thresholds=True):
    df_train, df_test, y_train, y_test = load_labels()
    softmax_train, softmax_test = load_softmax_outputs()

    if compute_optimal_thresholds:
        thr = optimal_thresholds(y_train, softmax_train)
    else:
        thr = [0.52909613, 1.6936609, 2.75167356, 3.58791778]

    print('Opt. train kappa:',
          quadratic_weighted_kappa(y_train,
                                   probs_to_labels(softmax_train, thr), 5))
    print('Opt. test kappa:',
          quadratic_weighted_kappa(y_test,
                                   probs_to_labels(softmax_test, thr), 5))

    print('Argmax train kappa:',
          quadratic_weighted_kappa(y_train,
                                   np.argmax(softmax_train, axis=1), 5))
    print('Argmax test kappa:',
          quadratic_weighted_kappa(y_test,
                                   np.argmax(softmax_test, axis=1), 5))

    private = (df_test.Usage == 'Private').values
    public = (df_test.Usage == 'Public').values
    print('Opt. test (private leaderboard) kappa:',
          quadratic_weighted_kappa(y_test[private],
                                   probs_to_labels(softmax_test[private], thr),
                                   5))
    print('Opt. test (public leaderboard) kappa:',
          quadratic_weighted_kappa(y_test[public],
                                   probs_to_labels(softmax_test[public], thr),
                                   5))


if __name__ == '__main__':
    main()
