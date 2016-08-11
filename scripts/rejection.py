"""Analysis of how the rejection of test samples based on model uncertainty
   affects the performance of the rest of the data that remains automatically
   classified"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score

from util import roc_curve_plot

plt.ion()
sns.set_context('paper', font_scale=1.4)

A4_WIDTH_SQUARE = (8.27, 8.27)


def load_labels():
    df_test = pd.read_csv('data/kaggle_dr/retinopathy_solution.csv')
    y_test = df_test.level.values
    return y_test


def load_predictions():
    """Load test predictions obtained with scripts/predict.py,
       e.g. based on commit 40d8265"""
    with open('data/processed/40d8265_jfnet_100MCdropout_KaggleDR_test.pkl',
              'rb') as h:
        pred_test = pickle.load(h)
    probs = pred_test['det_out']
    probs_mc = pred_test['stoch_out']
    return probs, probs_mc


def binary_labels(labels, min_positive_level=1):
    labels_bin = np.zeros_like(labels)
    labels_bin[labels < min_positive_level] = 0
    labels_bin[labels >= min_positive_level] = 1
    return labels_bin


def binary_probs(probs, min_positive_level=1):
    return probs[:, min_positive_level:].sum(axis=1)


def detection_task(y, probs, probs_mc, disease_level):
    y_diseased = binary_labels(y, disease_level)
    probs_diseased = binary_probs(probs, disease_level)
    probs_mc_diseased = binary_probs(probs_mc, disease_level)
    return y_diseased, probs_diseased, probs_mc_diseased


def posterior_statistics(probs_mc_bin):
    predictive_mean = probs_mc_bin.mean(axis=1)
    predictive_std = probs_mc_bin.std(axis=1)
    return predictive_mean, predictive_std


def argmax_labels(probs):
    return (probs >= 0.5).astype(int)


def accuracy(y_true, probs):
    y_pred = argmax_labels(probs)
    assert len(y_true) == len(y_pred)
    return (y_true == y_pred).sum() / float(len(y_true))


def stratified_mask(y, y_prior, shuffle=False):
    """Get mask such that y[mask] has the same size and class freq. as y_prior

    Parameters
    ==========
    y : array with labels
    y_prior: subset of y, defining the relative class frequencies
    shuffle: bool, False by default
        random selection from the pool of each class

    Returns
    =======
    select: bool. array of the same size as y with len(y_prior) True entries


    """
    classes = np.unique(y_prior)
    k_n = {k: (y_prior == k).sum() for k in classes}
    mask = np.array(len(y) * [False])
    for k, n in k_n.iteritems():
        idx_k = np.where(y == k)[0]
        if shuffle:
            np.random.shuffle(idx_k)
        select_n_from_k = idx_k[:n]
        mask[select_n_from_k] = True
    return mask


def performance_over_uncertainty_tol(uncertainty, y, probs, measure):
    uncertainty_tol = np.linspace(np.percentile(uncertainty, 10),
                                  uncertainty.max(),
                                  100)
    p = np.zeros_like(uncertainty_tol)
    p_rand = np.zeros_like(uncertainty_tol)
    p_strat = np.zeros_like(uncertainty_tol)
    frac_retain = np.zeros_like(uncertainty_tol)
    n_samples = len(uncertainty)
    for i, ut in enumerate(uncertainty_tol):
        accept = (uncertainty <= ut)
        rand_sel = np.random.permutation(accept)
        strat_sel = stratified_mask(y, y[accept], shuffle=True)
        p[i] = measure(y[accept], probs[accept])
        p_rand[i] = measure(y[rand_sel], probs[rand_sel])
        p_strat[i] = measure(y[strat_sel], probs[strat_sel])
        frac_retain[i] = accept.sum() / float(n_samples)

    return uncertainty_tol, frac_retain, p, p_rand, p_strat


def acc_rejection_figure(y, y_score, uncertainty, disease_onset,
                         save=False, format='.svg'):
    plt.figure(figsize=A4_WIDTH_SQUARE)
    plt.suptitle('Accuracy under rejection (disease onset: {})'.format(
                 disease_onset))
    y_pred = argmax_labels(y_score)
    corr = (y_pred == y)
    error = (y_pred != y)

    plt.subplot(2, 2, 1)
    sns.distplot(uncertainty[corr], label='correct')
    sns.distplot(uncertainty[error], label='error')
    plt.xlabel('model uncertainty')
    plt.ylabel('density')
    plt.legend(loc='best')

    plt.subplot(2, 2, 2)
    bins = np.linspace(np.min(uncertainty), np.max(uncertainty), num=100)
    sns.distplot(uncertainty[corr], bins=bins, kde=False, norm_hist=False,
                 label='correct')
    sns.distplot(uncertainty[error], bins=bins, kde=False, norm_hist=False,
                 label='error')
    plt.xlabel('model uncertainty')
    plt.ylabel('counts')
    plt.legend(loc='best')

    uncertainty_tol, frac_retain, acc, acc_rand, acc_strat = \
        performance_over_uncertainty_tol(uncertainty, y, y_score, accuracy)

    plt.subplot(2, 2, 3)
    plt.plot(uncertainty_tol, acc, label='dropout uncertainty')
    plt.plot(uncertainty_tol, acc_rand, label='randomly rejected')
    plt.plot(uncertainty_tol, acc_strat, label='prior preserved')
    plt.xlabel('tolerated model uncertainty')
    plt.ylabel('accuracy')
    plt.legend(loc='best')

    plt.subplot(2, 2, 4)
    plt.plot(frac_retain, acc, label='dropout uncertainty')
    plt.plot(frac_retain, acc_rand, label='randomly rejected')
    plt.plot(frac_retain, acc_strat, label='prior preserved')
    plt.xlabel('fraction of retained data')
    plt.ylabel('accuracy')
    plt.legend(loc='best')

    if save:
        plt.savefig('acc_' + str(disease_onset) + format)


def roc_auc_rejection_figure(y, y_score, uncertainty, disease_onset,
                             save=False, format='.svg'):
    plt.figure(figsize=A4_WIDTH_SQUARE)
    plt.suptitle('ROC under rejection (disease onset: {})'.format(
                 disease_onset))

    uncertainty_tol, frac_retain, auc, auc_rand, auc_strat = \
        performance_over_uncertainty_tol(uncertainty, y, y_score,
                                         roc_auc_score)

    plt.subplot2grid((2, 2), (0, 0))
    plt.plot(uncertainty_tol, auc, label='dropout uncertainty')
    plt.plot(uncertainty_tol, auc_rand, label='randomly rejected')
    plt.plot(uncertainty_tol, auc_strat, label='prior preserved')
    plt.xlabel('tolerated model uncertainty')
    plt.ylabel('roc_auc')
    plt.legend(loc='best')

    plt.subplot2grid((2, 2), (0, 1))
    plt.plot(frac_retain, auc, label='dropout uncertainty')
    plt.plot(frac_retain, auc_rand, label='randomly rejected')
    plt.plot(frac_retain, auc_strat, label='prior preserved')
    plt.xlabel('fraction of retained data')
    plt.ylabel('roc_auc')
    plt.legend(loc='best')

    plt.subplot2grid((2, 2), (1, 0), colspan=2)
    roc_curve_plot(y, y_score, legend_prefix='without rejection',
                   plot_BDA=True)
    fractions = [0.9, 0.8, 0.7]
    for f in fractions:
        thr = uncertainty_tol[frac_retain >= f][0]
        roc_curve_plot(y[uncertainty <= thr],
                       y_score[uncertainty <= thr],
                       legend_prefix='{}% data retained'.format(f * 100))

    if save:
        plt.savefig('roc_' + str(disease_onset) + format)


def class_conditional_uncertainty(y, uncertainty, disease_onset,
                                  save=False, format='.svg'):
    plt.figure(figsize=map(lambda x: x / 2.0, A4_WIDTH_SQUARE))
    plt.title('Disease onset: {}'.format(disease_onset))
    HEALTHY, DISEASED = 0, 1

    sns.distplot(uncertainty[y == HEALTHY], label='healthy')
    sns.distplot(uncertainty[y == DISEASED], label='diseased')
    plt.xlabel('model uncertainty')
    plt.ylabel('density')
    plt.legend(loc='best')
    plt.ylim(0, 80)

    if save:
        plt.savefig('class_cond_uncertainty_' + str(disease_onset) + format)


def main():
    y = load_labels()
    probs, probs_mc = load_predictions()

    disease_onset_levels = [1, 2, 3, 4]

    for dl in disease_onset_levels:
        y_bin, probs_bin, probs_mc_bin = detection_task(y, probs, probs_mc, dl)
        pred_mean, pred_std = posterior_statistics(probs_mc_bin)
        acc_rejection_figure(y_bin, pred_mean, pred_std, dl,
                             save=True, format='.png')
        roc_auc_rejection_figure(y_bin, pred_mean, pred_std, dl,
                                 save=True, format='.png')
        class_conditional_uncertainty(y_bin, pred_std, dl,
                                      save=True, format='.png')


if __name__ == '__main__':
    main()
