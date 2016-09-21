"""Analysis of how the rejection of test samples based on model uncertainty
   affects the performance of the rest of the data that remains automatically
   classified"""
from __future__ import print_function
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cPickle as pickle
import os
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score
import statsmodels.nonparametric.api as smnp

from util import roc_curve_plot

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

plt.ion()
sns.set_context('paper', font_scale=1.4)

A4_WIDTH_SQUARE = (8.27, 8.27)
TAG = {0: 'healthy', 1: 'diseased'}


CONFIG = {
    'KaggleDR':
        {'LABELS_FILE': 'data/kaggle_dr/retinopathy_solution.csv',
         'IMAGE_PATH': 'data/kaggle_dr/test_JF_512',
         'LEVEL': OrderedDict([(0, 'no DR'),
                               (1, 'mild DR'),
                               (2, 'moderate DR'),
                               (3, 'severe DR'),
                               (4, 'proliferative DR')]),
         'min_percentile': 10},
    'Messidor':
        {'LABELS_FILE': 'data/messidor/messidor.csv',
         'IMAGE_PATH': 'data/messidor/JF_512',
         'LEVEL': OrderedDict([(0, 'no DR'),
                               (1, 'mild non-proliferative DR'),
                               (2, 'severe non-proliferative DR'),
                               (3, 'most serious')]),
         'min_percentile': 50},
    'Messidor_R0vsR1':
        {'LABELS_FILE': 'data/messidor/messidor_R0vsR1.csv',
         'IMAGE_PATH': 'data/messidor/JF_512',
         'LEVEL': OrderedDict([(0, 'no DR'),
                               (1, 'mild non-proliferative DR')]),
         'min_percentile': 50}
}


def load_labels(labels_file):
    df_test = pd.read_csv(labels_file)
    y_test = df_test.level.values
    return y_test


def load_filenames(labels_file):
    df_test = pd.read_csv(labels_file)
    return df_test.image.values


def load_predictions(filename):
    """Load test predictions obtained with scripts/predict.py"""
    with open(filename, 'rb') as h:
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
    n_classes = probs.shape[1]
    if n_classes == 5:
        return probs[:, min_positive_level:].sum(axis=1)
    elif n_classes == 2:
        return np.squeeze(probs[:, 1:])
    else:
        print('Unknown number of classes: %d. Aborting.' % n_classes)


def detection_task(y, probs, probs_mc, disease_level):
    y_diseased = binary_labels(y, disease_level)
    probs_diseased = binary_probs(probs, disease_level)
    probs_mc_diseased = binary_probs(probs_mc, disease_level)
    return y_diseased, probs_diseased, probs_mc_diseased


def mode(data):
    """Compute a kernel density estimate and return the mode"""
    if len(np.unique(data)) == 1:
        return data[0]
    else:
        kde = smnp.KDEUnivariate(data.astype('double'))
        kde.fit(cut=0)
        grid, y = kde.support, kde.density
        return grid[y == y.max()][0]


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


def rel_freq(y, k):
    return (y == k).sum()/float(len(y))


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


def contralateral_agreement(y, config):
    """Get boolean array of contralateral label agreement

    Notes
    =====

    A very similar function is already there in datasets.py but here we want
    to work on indices and more importantly check for contralateral label
    agreement for a potentially binary label vector y for the corresponding
    disease detection problem.

    """

    if 'kaggle_dr' not in config['LABELS_FILE']:
        raise TypeError('Laterality not defined for %s'
                        % config['LABELS_FILE'])

    df = pd.read_csv(config['LABELS_FILE'])
    left = df.image.str.contains(r'\d+_left').values
    right = df.image.str.contains(r'\d+_right').values

    accepted_patients = (y[left] == y[right])
    accepted_images_left = df[left].image[accepted_patients]
    accepted_images_right = df[right].image[accepted_patients]
    accepted_images = pd.concat((accepted_images_left,
                                 accepted_images_right))
    return df.image.isin(accepted_images).values


def performance_over_uncertainty_tol(uncertainty, y, probs, measure, config):
    uncertainty_tol, frac_retain, accept_idx = \
        sample_rejection(uncertainty, config['min_percentile'])

    p = np.zeros_like(uncertainty_tol)
    p_rand = np.zeros_like(uncertainty_tol)
    p_strat = np.zeros_like(uncertainty_tol)

    for i, ut in enumerate(uncertainty_tol):
        accept = accept_idx[i]
        rand_sel = np.random.permutation(accept)
        strat_sel = stratified_mask(y, y[accept], shuffle=True)
        p[i] = measure(y[accept], probs[accept])
        p_rand[i] = measure(y[rand_sel], probs[rand_sel])
        p_strat[i] = measure(y[strat_sel], probs[strat_sel])

    return uncertainty_tol, frac_retain, p, p_rand, p_strat


def sample_rejection(uncertainty, min_percentile):
    uncertainty_tol = np.linspace(np.percentile(uncertainty, min_percentile),
                                  uncertainty.max(), 100)
    frac_retain = np.zeros_like(uncertainty_tol)
    n_samples = len(uncertainty)
    accept_indices = []
    for i, ut in enumerate(uncertainty_tol):
        accept = (uncertainty <= ut)
        accept_indices.append(accept)
        frac_retain[i] = accept.sum() / float(n_samples)

    return uncertainty_tol, frac_retain, accept_indices


def acc_rejection_figure(y, y_score, uncertainties, disease_onset, config,
                         save=False, format='.svg', fig=None):
    if fig is None:
        fig = plt.figure(figsize=(A4_WIDTH_SQUARE[0],
                                  A4_WIDTH_SQUARE[0] / 2.0))

    ax121 = plt.subplot(1, 2, 1)
    ax122 = plt.subplot(1, 2, 2)
    ax121.set_title('(a)')
    ax122.set_title('(b)')

    min_acc = 1.0
    for k, v in uncertainties.iteritems():
        v_tol, frac_retain, acc, acc_rand, acc_strat = \
            performance_over_uncertainty_tol(v, y, y_score, accuracy, config)
        ax121.plot(v_tol, acc, label=k)
        ax122.plot(frac_retain, acc, label=k)
        if min_acc > min(np.concatenate((acc, acc_rand, acc_strat))):
            min_acc = min(np.concatenate((acc, acc_rand, acc_strat)))

    ax121.set_ylim(min_acc, 1)
    ax122.set_ylim(min_acc, 1)
    ax121.set_xlabel('tolerated model uncertainty')
    ax121.set_ylabel('accuracy')
    ax121.legend(loc='best')

    ax122.plot(frac_retain, acc_rand, label='randomly rejected')
    ax122.plot(frac_retain, acc_strat, label='prior preserved')
    ax122.set_xlabel('fraction of retained data')
    ax122.legend(loc='best')

    if save:
        fig.savefig('acc_' + str(disease_onset) + format)


def level_rejection_figure(y_level, uncertainty, disease_onset, config,
                           save=False, format='.svg', fig=None):
    if fig is None:
        fig = plt.figure(figsize=(A4_WIDTH_SQUARE[0],
                                  A4_WIDTH_SQUARE[0] / 2.0))

    tol, frac_retain, accept_idx = sample_rejection(uncertainty,
                                                    config['min_percentile'])
    LEVEL = config['LEVEL']
    p = {level: np.array([rel_freq(y_level[~accept], level)
                          for accept in accept_idx])
         for level in LEVEL}
    cum = np.zeros_like(tol)

    with sns.axes_style('white'):

        ax121 = plt.subplot(1, 2, 1)
        ax122 = plt.subplot(1, 2, 2)
        ax121.set_title('(a)')
        ax122.set_title('(b)')

        colors = {level: sns.color_palette("Blues")[level] for level in LEVEL}
        for level in LEVEL:
            ax121.fill_between(tol, p[level] + cum, cum,
                               color=colors[level], label=LEVEL[level])
            ax122.fill_between(frac_retain, p[level] + cum, cum,
                               color=colors[level], label=LEVEL[level])
            if (level + 1) == disease_onset:
                ax121.plot(tol, p[level] + cum,
                           color='k', label='decision boundary')
                ax122.plot(frac_retain, p[level] + cum,
                           color='k', label='decision boundary')
            cum += p[level]

        ax121.set_xlim(min(tol), max(tol))
        ax122.set_xlim(min(frac_retain), max(frac_retain))
        ax121.set_ylim(0, 1)
        ax122.set_ylim(0, 1)

        ax121.set_xlabel('tolerated model uncertainty')
        ax121.set_ylabel('relative proportions within rejected dataset')
        ax121.legend(loc='lower center')
        ax122.set_xlabel('fraction of retained data')
        ax122.legend(loc='lower center')

    if save:
        fig.savefig('level_' + str(disease_onset) + format)


def label_disagreement_figure(y, uncertainty, disease_onset, config,
                              save=False, format='.svg', fig=None):
    try:
        disagreeing = ~contralateral_agreement(y, config)
    except TypeError:
        print('No data for label disagreement figure available.')
        return

    if fig is None:
        fig = plt.figure(figsize=(A4_WIDTH_SQUARE[0],
                                  A4_WIDTH_SQUARE[0] / 2.0))

    tol, frac_retain, accept_idx = sample_rejection(uncertainty,
                                                    config['min_percentile'])

    p_rejected = np.array([sum((~accept) & (disagreeing))/float(sum(~accept))
                           for accept in accept_idx])
    p_retained = np.array([sum((accept) & (disagreeing))/float(sum(accept))
                           for accept in accept_idx])

    with sns.axes_style('white'):

        ax121 = plt.subplot(1, 2, 1)
        ax122 = plt.subplot(1, 2, 2)
        ax121.set_title('(a)')
        ax122.set_title('(b)')

        ax121.fill_between(tol, p_rejected, 0, alpha=0.5,
                           color=sns.color_palette()[0], label='rejected')
        ax121.fill_between(tol, p_retained, 0, alpha=0.5,
                           color=sns.color_palette()[1], label='retained')
        ax122.fill_between(frac_retain, p_rejected, 0, alpha=0.5,
                           color=sns.color_palette()[0], label='rejected')
        ax122.fill_between(frac_retain, p_retained, 0, alpha=0.5,
                           color=sns.color_palette()[1], label='retained')

        ax121.set_xlim(min(tol), max(tol))
        ax122.set_xlim(min(frac_retain), max(frac_retain))
        ax121.set_ylim(0, 1)
        ax122.set_ylim(0, 1)

        ax121.set_xlabel('tolerated model uncertainty')
        ax122.set_xlabel('fraction of retained data')
        ax121.set_ylabel('fraction of data with patient level label noise')
        ax121.legend()
        ax122.legend()

    if save:
        fig.savefig('label_disagreement_' + str(disease_onset) + format)


def roc_auc_rejection_figure(y, y_score, uncertainties, disease_onset, config,
                             save=False, format='.svg', fig=None):
    if fig is None:
        fig = plt.figure(figsize=A4_WIDTH_SQUARE)

    colors = sns.color_palette()

    ax220 = plt.subplot2grid((2, 2), (0, 0))
    ax221 = plt.subplot2grid((2, 2), (0, 1))
    ax2223 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

    ax220.set_title('(a)')
    ax221.set_title('(b)')
    ax2223.set_title('(c)')

    for k, v in uncertainties.iteritems():
        v_tol, frac_retain, auc, auc_rand, auc_strat = \
            performance_over_uncertainty_tol(v, y, y_score,
                                             roc_auc_score,
                                             config)

        ax220.plot(v_tol, auc, label=k)
        ax221.plot(frac_retain, auc, label=k)

        ax2223
        fractions = [0.9, 0.8, 0.7]
        for i, f in enumerate(fractions):
            thr = v_tol[frac_retain >= f][0]
            roc_curve_plot(y[v <= thr],
                           y_score[v <= thr],
                           color=colors[i + 1],
                           legend_prefix='%d%% data retained, %s' % (f * 100,
                                                                     k))

    ax220.set_xlabel('tolerated model uncertainty')
    ax220.set_ylabel('roc_auc')
    ax220.legend(loc='best')

    ax221.plot(frac_retain, auc_rand, label='randomly rejected')
    ax221.plot(frac_retain, auc_strat, label='prior preserved')
    ax221.set_xlabel('fraction of retained data')
    ax221.legend(loc='best')

    ax2223
    roc_curve_plot(y, y_score, color=colors[0],
                   legend_prefix='without rejection',
                   plot_BDA=True)

    ax2223.set_aspect(1.0)

    if save:
        fig.savefig('roc_' + str(disease_onset) + format)


def error_conditional_uncertainty(y, y_score, uncertainty, disease_onset,
                                  label='pred_std', ax=None):
    """Plot conditional pdfs for correct and erroneous argmax predictions"""
    if ax is None:
        ax = plt.figure(figsize=A4_WIDTH_SQUARE).gca()

    y_pred = argmax_labels(y_score)
    corr = (y_pred == y)
    error = (y_pred != y)

    ax = sns.kdeplot(uncertainty[corr], ax=ax, shade=True, cut=0,
                     label=label + '[corr]')
    ax = sns.kdeplot(uncertainty[error], ax=ax, shade=True, cut=0,
                     label=label + '[error]')

    ax.set_xlabel('model uncertainty')
    ax.set_ylabel('density')
    ax.legend(loc='best')

    return ax


def fig1(y, y_score, images, uncertainty, probs_mc_diseased,
         disease_onset, y_level, label='$\sigma_{pred}$',
         save=False, format='.png', image_path=None, level=None):
    asc = np.argsort(uncertainty)
    certain = 0
    uncertain = len(y) - 1
    middle_certain = np.where(uncertainty[asc] > 0.14)[0][0]
    examples = [certain, middle_certain, uncertain]
    fig = plt.figure(figsize=A4_WIDTH_SQUARE)
    for idx, i in enumerate(examples):
        im = mpimg.imread(os.path.join(image_path, images[asc][i] + '.jpeg'))

        with sns.axes_style("white"):
            plt.subplot2grid((2, 2 * len(examples)), (0, 2 * idx))
            plt.imshow(im)
            plt.axis('off')
            title = ['(a)', '(b)', '(c)'][idx] + ' ' + TAG[y[asc][i]]
            level_info = ' (' + level[y_level[asc][i]] + ')'
            print(title, level_info)
            plt.title(title, loc='left')

        ax = plt.subplot2grid((2, 2 * len(examples)), (0, 2 * idx + 1))
        if uncertainty[asc][i] <= 0.000:
            color = sns.color_palette()[0]
            plt.bar(0.98, 1.0, width=0.02, alpha=0.5, color=color)
            plt.hlines(1.0, 0.98, 1.0, color=color, linewidth=2)
        else:
            sns.kdeplot(probs_mc_diseased[asc][i], shade=True)
        y_pos = ax.get_ylim()[1] / 2.0
        plt.annotate(['"certain":\n $\sigma_{pred}$ = %.2f'
                      % uncertainty[asc][i],
                      '"uncertain":\n $\sigma_{pred}$ = %.2f'
                      % uncertainty[asc][i],
                      '"uncertain":\n $\sigma_{pred}$ = %.2f'
                      % uncertainty[asc][i]][idx],
                     (0.33, 0.7 * y_pos))
        length = 0.5 * max(uncertainty[asc][i], 0.02)
        arrow_params = {'length_includes_head': True,
                        'width': 0.005 * y_pos,
                        'head_width': 0.05 * y_pos,
                        'head_length': 0.05}
        plt.arrow(0.5, y_pos, length, 0, **arrow_params)
        plt.arrow(0.5, y_pos, -length, 0, **arrow_params)
        plt.xlabel('p(diseased | image)')
        plt.ylabel('density')
        plt.xlim(0, 1)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect(1 / ax.get_ylim()[1])

    ax = plt.subplot2grid((2, 2 * len(examples)), (1, 0),
                          colspan=2 * len(examples))
    ax.set_title('(d)', loc='left')
    error_conditional_uncertainty(y, y_score, uncertainty, disease_onset,
                                  label=label, ax=ax)

    if save:
        fig.savefig('figure1_' + str(disease_onset) + format)


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

    config = CONFIG['KaggleDR']

    y = load_labels(config['LABELS_FILE'])
    images = load_filenames(config['LABELS_FILE'])
    probs, probs_mc = load_predictions('data/processed/'
        '100_mc_KaggleDR_test_BayesianJFnet17_onset2_b69aadd.pkl')

    disease_onset_levels = [2]
    for dl in disease_onset_levels:
        y_bin, probs_bin, probs_mc_bin = detection_task(y, probs, probs_mc, dl)
        pred_mean, pred_std = posterior_statistics(probs_mc_bin)
        uncertainties = {'$\sigma_{pred}$': pred_std}

        fig1(y_bin, pred_mean, images, pred_std, probs_mc_bin, dl, y,
             label='$\sigma_{pred}$', save=True, format='.png',
             image_path=config['IMAGE_PATH'],
             level=config['LEVEL'])

        acc_rejection_figure(y_bin, pred_mean, uncertainties, dl, config,
                             save=True, format='.png')

        roc_auc_rejection_figure(y_bin, pred_mean, uncertainties, dl, config,
                                 save=True, format='.svg')

        level_rejection_figure(y, pred_std, dl, config,
                               save=True, format='.svg')

        label_disagreement_figure(y_bin, pred_std, dl, config,
                                  save=True, format='.svg')


if __name__ == '__main__':
    main()
