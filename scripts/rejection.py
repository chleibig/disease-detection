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
from util import bootstrap

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

plt.ion()
sns.set_context('paper', font_scale=2)

A4_WIDTH_SQUARE = (8.27, 8.27)
A4_WIDTH = 8.27

TAG = {0: 'healthy', 1: 'diseased'}
ONSET_TAG = {1: 'mild DR', 2: 'moderate DR'}

DATA = {
    'KaggleDR_train':
        {'LABELS_FILE': 'data/kaggle_dr/trainLabels.csv',
         'IMAGE_PATH': 'data/kaggle_dr/train_JF_512',
         'LEVEL': OrderedDict([(0, 'no DR'),
                               (1, 'mild DR'),
                               (2, 'moderate DR'),
                               (3, 'severe DR'),
                               (4, 'proliferative DR')]),
         'min_percentile': 50,
         'n_bootstrap': 10000},
    'KaggleDR':
        {'LABELS_FILE': 'data/kaggle_dr/retinopathy_solution.csv',
         'IMAGE_PATH': 'data/kaggle_dr/test_JF_512',
         'LEVEL': OrderedDict([(0, 'no DR'),
                               (1, 'mild DR'),
                               (2, 'moderate DR'),
                               (3, 'severe DR'),
                               (4, 'proliferative DR')]),
         'min_percentile': 50,
         'n_bootstrap': 10000},
    'Messidor':
        {'LABELS_FILE': 'data/messidor/messidor.csv',
         'IMAGE_PATH': 'data/messidor/JF_512',
         'LEVEL': OrderedDict([(0, 'no DR'),
                               (1, 'mild non-proliferative DR'),
                               (2, 'severe non-proliferative DR'),
                               (3, 'most serious')]),
         'min_percentile': 50,
         'n_bootstrap': 10000}
}

CONFIG = {
    'BayesJF17_mildDR_Kaggle_train': dict(
        [('net', 'Bayesian JFnet'),
         ('dataset', 'Kaggle DR train'),
         ('predictions', 'data/processed/'
          '100_mc_KaggleDR_train_BayesJFnet17_392bea6.pkl'),
         ('disease_onset', 1)] +
        DATA['KaggleDR_train'].items()),

    'BayesJF17_mildDR_Kaggle': dict(
        [('net', 'Bayesian JFnet'),
         ('dataset', 'Kaggle DR'),
         ('predictions', 'data/processed/'
          '100_mc_KaggleDR_test_BayesJFnet17_392bea6.pkl'),
         ('disease_onset', 1)] +
        DATA['KaggleDR'].items()),

    'BayesJF17_moderateDR_Kaggle': dict(
        [('net', 'Bayesian JFnet'),
         ('dataset', 'Kaggle DR'),
         ('predictions', 'data/processed/'
          '100_mc_KaggleDR_test_BayesianJFnet17_onset2_b69aadd.pkl'),
         ('disease_onset', 2)] +
        DATA['KaggleDR'].items()),

    'JFnet_mildDR_Kaggle': dict(
        [('net', 'JFnet'),
         ('dataset', 'Kaggle DR'),
         ('predictions', 'data/processed/'
          'c9ade47_100_mc_KaggleDR_test_JFnet.pkl'),
         ('disease_onset', 1)] +
        DATA['KaggleDR'].items()),

    'JFnet_moderateDR_Kaggle': dict(
        [('net', 'JFnet'),
         ('dataset', 'Kaggle DR'),
         ('predictions', 'data/processed/'
          'c9ade47_100_mc_KaggleDR_test_JFnet.pkl'),
         ('disease_onset', 2)] +
        DATA['KaggleDR'].items()),

    'BayesJF17_mildDR_Messidor': dict(
        [('net', 'Bayesian JFnet'),
         ('dataset', 'Messidor'),
         ('predictions', 'data/processed/'
          '100_mc_Messidor_BayesJFnet17_392bea6.pkl'),
         ('disease_onset', 1)] +
        DATA['Messidor'].items()),

    'BayesJF17_moderateDR_Messidor': dict(
        [('net', 'Bayesian JFnet'),
         ('dataset', 'Messidor'),
         ('predictions', 'data/processed/'
          '100_mc_Messidor_BayesianJFnet17_onset2_b69aadd.pkl'),
         ('disease_onset', 2)] +
        DATA['Messidor'].items()),
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


def performance_over_uncertainty_tol(uncertainty, y, probs, measure,
                                     min_percentile, n_bootstrap):

    uncertainty_tol, frac_retain, accept_idx = \
        sample_rejection(uncertainty, min_percentile)

    p = np.zeros((len(uncertainty_tol),), dtype=[('value', 'float64'),
                                                 ('low', 'float64'),
                                                 ('high', 'float64')])
    p_rand = np.zeros((len(uncertainty_tol),), dtype=[('value', 'float64'),
                                                      ('low', 'float64'),
                                                      ('high', 'float64')])

    for i, ut in enumerate(uncertainty_tol):
        accept = accept_idx[i]
        rand_sel = np.random.permutation(accept)

        low, high = bootstrap([y[accept], probs[accept]], measure,
                              n_resamples=n_bootstrap, alpha=0.05)

        p['value'][i] = measure(y[accept], probs[accept])
        p['low'][i] = low.value
        p['high'][i] = high.value

        low, high = bootstrap([y[rand_sel], probs[rand_sel]], measure,
                              n_resamples=100, alpha=0.05)

        p_rand['value'][i] = measure(y[rand_sel], probs[rand_sel])
        p_rand['low'][i] = low.value
        p_rand['high'][i] = high.value

    return uncertainty_tol, frac_retain, p, p_rand


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


def acc_rejection_figure(y, y_score, uncertainties, config,
                         save=False, format='.svg', fig=None):
    if fig is None:
        fig = plt.figure(figsize=(A4_WIDTH_SQUARE[0],
                                  A4_WIDTH_SQUARE[0] / 2.0))

    colors = sns.color_palette()

    ax121 = plt.subplot(1, 2, 1)
    ax122 = plt.subplot(1, 2, 2)
    ax121.set_title('(a)')
    ax122.set_title('(b)')

    min_acc = 1.0
    for i, (k, v) in enumerate(uncertainties.iteritems()):
        v_tol, frac_retain, acc, acc_rand = \
            performance_over_uncertainty_tol(v, y, y_score, accuracy, 0.0,
                                             config['n_bootstrap'])
        ax121.plot(v_tol, acc['value'],
                   label=k, color=colors[i], linewidth=2)
        ax122.plot(frac_retain, acc['value'],
                   label=k, color=colors[i], linewidth=2)
        ax121.fill_between(v_tol, acc['value'], acc['low'],
                           color=colors[i], alpha=0.3)
        ax121.fill_between(v_tol, acc['high'], acc['value'],
                           color=colors[i], alpha=0.3)
        ax122.fill_between(frac_retain, acc['value'], acc['low'],
                           color=colors[i], alpha=0.3)
        ax122.fill_between(frac_retain, acc['high'], acc['value'],
                           color=colors[i], alpha=0.3)
        if min_acc > min(min(acc['low']), min(acc_rand['low'])):
            min_acc = min(min(acc['low']), min(acc_rand['low']))

    ax121.set_ylim(min_acc, 1)
    ax122.set_ylim(min_acc, 1)
    ax121.set_xlabel('tolerated model uncertainty')
    ax121.set_ylabel('accuracy')
    ax121.legend(loc='best')

    ax122.plot(frac_retain, acc_rand['value'], label='randomly rejected',
               color=colors[i+1], linewidth=2)
    ax122.fill_between(frac_retain, acc_rand['value'], acc_rand['low'],
                       color=colors[i+1], alpha=0.3)
    ax122.fill_between(frac_retain, acc_rand['high'], acc_rand['value'],
                       color=colors[i+1], alpha=0.3)
    ax122.set_xlabel('fraction of retained data')
    ax122.legend(loc='best')

    name = 'acc_' + config['net'] + '_' + str(config['disease_onset']) + \
           '_' + config['dataset']

    if save:
        fig.savefig(name + format)

    return {name: fig}


def level_rejection_figure(y_level, uncertainty, config,
                           save=False, format='.svg', fig=None):
    if fig is None:
        fig = plt.figure(figsize=(A4_WIDTH_SQUARE[0],
                                  A4_WIDTH_SQUARE[0] / 2.0))

    tol, frac_retain, accept_idx = sample_rejection(uncertainty, 0.0)
    LEVEL = config['LEVEL']
    p = {level: np.array([rel_freq(y_level[~accept], level)
                          for accept in accept_idx])
         for level in LEVEL}
    cum = np.zeros_like(tol)

    with sns.axes_style('white'):

        ax121 = plt.subplot(1, 2, 1)
        ax122 = plt.subplot(1, 2, 2)
        ax121.set_title('(a) Disease onset: %s'
                        % ONSET_TAG[config['disease_onset']])
        ax122.set_title('(b) Disease onset: %s'
                        % ONSET_TAG[config['disease_onset']])

        colors = {level: sns.color_palette("Blues")[level] for level in LEVEL}
        for level in LEVEL:
            ax121.fill_between(tol, p[level] + cum, cum,
                               color=colors[level],
                               label='%d: %s' % (level, LEVEL[level]))
            ax122.fill_between(frac_retain, p[level] + cum, cum,
                               color=colors[level],
                               label='%d: %s' % (level, LEVEL[level]))
            if (level + 1) == config['disease_onset']:
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
        ax121.set_ylabel('relative proportions within referred dataset')
        ax121.legend(loc='lower center')
        ax122.set_xlabel('fraction of retained data')
        ax122.legend(loc='lower center')

    name = 'level_' + config['net'] + '_' + str(config['disease_onset']) + \
           '_' + config['dataset']

    if save:
        fig.savefig(name + format)

    return {name: fig}


def label_disagreement_figure(y, uncertainty, config,
                              save=False, format='.svg', fig=None):
    try:
        disagreeing = ~contralateral_agreement(y, config)
    except TypeError:
        print('No data for label disagreement figure available.')
        return

    if fig is None:
        fig = plt.figure(figsize=(A4_WIDTH_SQUARE[0],
                                  A4_WIDTH_SQUARE[0] / 2.0))

    tol, frac_retain, accept_idx = sample_rejection(uncertainty, 0.1)

    p_referred = np.array([sum((~accept) & (disagreeing))/float(sum(~accept))
                           for accept in accept_idx])
    p_retained = np.array([sum((accept) & (disagreeing))/float(sum(accept))
                           for accept in accept_idx])

    with sns.axes_style('white'):

        ax121 = plt.subplot(1, 2, 1)
        ax122 = plt.subplot(1, 2, 2)
        ax121.set_title('(a)')
        ax122.set_title('(b)')

        ax121.fill_between(tol, p_referred, 0, alpha=0.5,
                           color=sns.color_palette()[0], label='referred')
        ax121.fill_between(tol, p_retained, 0, alpha=0.5,
                           color=sns.color_palette()[1], label='retained')
        ax122.fill_between(frac_retain, p_referred, 0, alpha=0.5,
                           color=sns.color_palette()[0], label='referred')
        ax122.fill_between(frac_retain, p_retained, 0, alpha=0.5,
                           color=sns.color_palette()[1], label='retained')

        ax121.set_xlim(min(tol), max(tol))
        ax122.set_xlim(min(frac_retain), max(frac_retain))
        ax121.set_ylim(0, 1)
        ax122.set_ylim(0, 1)

        ax121.set_xlabel('tolerated model uncertainty')
        ax122.set_xlabel('fraction of retained data')
        ax121.set_ylabel('fraction of data with patient level ambiguity')
        ax121.legend()
        ax122.legend()

    name = 'label_disagreement_' + config['net'] + '_' + \
           str(config['disease_onset']) + '_' + config['dataset']

    if save:
        fig.savefig(name + format)

    return {name: fig}


def roc_auc_rejection_figure(y, y_score, uncertainties, config,
                             save=False, format='.svg', fig=None):
    if fig is None:
        fig = plt.figure(figsize=(A4_WIDTH_SQUARE[0],
                                  A4_WIDTH_SQUARE[0] / 2.0))

    colors = sns.color_palette()

    ax121 = plt.subplot2grid((1, 2), (0, 0))
    ax122 = plt.subplot2grid((1, 2), (0, 1))

    ax121.set_title('(a) %s(disease onset: %s); %s'
                    % (config['net'], ONSET_TAG[config['disease_onset']],
                       config['dataset']))
    ax122.set_title('(b) %s(disease onset: %s); %s'
                    % (config['net'], ONSET_TAG[config['disease_onset']],
                       config['dataset']))

    for i, (k, v) in enumerate(uncertainties.iteritems()):
        v_tol, frac_retain, auc, auc_rand = \
            performance_over_uncertainty_tol(v, y, y_score,
                                             roc_auc_score,
                                             config['min_percentile'],
                                             config['n_bootstrap'])

        ax121.plot(frac_retain, auc['value'],
                   label=k, color=colors[i], linewidth=2)
        ax121.fill_between(frac_retain, auc['value'], auc['low'],
                           color=colors[i], alpha=0.3)
        ax121.fill_between(frac_retain, auc['high'], auc['value'],
                           color=colors[i], alpha=0.3)

        ax122
        fractions = [0.9, 0.8, 0.7]
        for j, f in enumerate(fractions):
            thr = v_tol[frac_retain >= f][0]
            roc_curve_plot(y[v <= thr],
                           y_score[v <= thr],
                           color=colors[j+1],
                           legend_prefix='%d%% data retained, %s' % (f * 100,
                                                                     k))

    ax121.plot(frac_retain, auc_rand['value'],
               label='randomly rejected', color=colors[i+1], linewidth=2)
    ax121.fill_between(frac_retain, auc_rand['value'], auc_rand['low'],
                       color=colors[i+1], alpha=0.3)
    ax121.fill_between(frac_retain, auc_rand['high'], auc_rand['value'],
                       color=colors[i+1], alpha=0.3)
    ax121.set_xlim(config['min_percentile']/100., 1.0)
    ax121.set_xlabel('fraction of retained data')
    ax121.set_ylabel('roc_auc')
    ax121.legend(loc='best')

    ax122
    roc_curve_plot(y, y_score, color=colors[0],
                   legend_prefix='without rejection',
                   plot_BDA=True)

    x0, x1 = ax121.get_xlim()
    y0, y1 = ax121.get_ylim()
    ax121.set_aspect((x1 - x0)/(y1 - y0))
    x0, x1 = ax122.get_xlim()
    y0, y1 = ax122.get_ylim()
    ax122.set_aspect((x1 - x0)/(y1 - y0))

    plt.tight_layout()

    name = 'roc_' + config['net'] + '_' + str(config['disease_onset']) + \
           '_' + config['dataset']

    if save:
        fig.savefig(name + format)

    return {name: fig}


def train_test_generalization():
    """Visualizes performance over uncertainty for both train and test data"""
    fig = plt.figure(figsize=(A4_WIDTH / 2.0,
                              A4_WIDTH / 2.0))
    ax = fig.gca()
    colors = sns.color_palette()

    configs = {'$\sigma_{pred} (train)$':
               CONFIG['BayesJF17_mildDR_Kaggle_train'],
               '$\sigma_{pred} (test)$':
               CONFIG['BayesJF17_mildDR_Kaggle']}

    for i, (k, config) in enumerate(configs.iteritems()):
        y = load_labels(config['LABELS_FILE'])
        probs, probs_mc = load_predictions(config['predictions'])
        y_bin, probs_bin, probs_mc_bin = detection_task(
            y, probs, probs_mc, config['disease_onset'])
        pred_mean, pred_std = posterior_statistics(probs_mc_bin)

        v_tol, _, auc, auc_rand = \
            performance_over_uncertainty_tol(pred_std, y_bin, pred_mean,
                                             roc_auc_score,
                                             config['min_percentile'],
                                             config['n_bootstrap'])
        ax.plot(v_tol, auc['value'],
                label=k, color=colors[i], linewidth=2)
        ax.fill_between(v_tol, auc['value'], auc['low'],
                        color=colors[i], alpha=0.3)
        ax.fill_between(v_tol, auc['high'], auc['value'],
                        color=colors[i], alpha=0.3)

    ax.set_xlabel('tolerated model uncertainty [$\sigma_{pred}$]')
    ax.set_ylabel('roc_auc')
    ax.legend(loc='best')

    name = 'train_test_' + config['net'] + '_' + \
           str(config['disease_onset']) + '_' + config['dataset']

    fig.savefig(name + '.pdf')

    return {name: fig}


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
         y_level, config, label='$\sigma_{pred}$', save=False, format='.png'):

    image_path = config['IMAGE_PATH']
    level = config['LEVEL']

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
                     (0.33, 0.6 * y_pos))
        length = 0.5 * max(uncertainty[asc][i], 0.02)
        arrow_params = {'length_includes_head': True,
                        'width': 0.005 * y_pos,
                        'head_width': 0.05 * y_pos,
                        'head_length': 0.05}
        plt.arrow(0.5, y_pos, length, 0, **arrow_params)
        plt.arrow(0.5, y_pos, -length, 0, **arrow_params)
        plt.xlabel('p(diseased | image)')
        plt.ylabel('density [a.u.]')
        plt.title('$\mu_{pred}$ = %.2f' % y_score[asc][i], loc='left')
        plt.xlim(0, 1)
        ax.get_yaxis().set_ticks([])
        ax.get_yaxis().set_ticklabels([])
        ax.set_aspect(1 / ax.get_ylim()[1])

    ax = plt.subplot2grid((2, 2 * len(examples)), (1, 1),
                          colspan=4)
    ax.set_title('(d)', loc='left')
    error_conditional_uncertainty(y, y_score, uncertainty,
                                  config['disease_onset'],
                                  label=label, ax=ax)

    name = 'fig1_' + config['net'] + '_' + str(config['disease_onset']) + \
           '_' + config['dataset']

    if save:
        fig.savefig(name + format)

    return {name: fig}


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


def resize_and_save(figures, size_inches, format='.pdf'):
    assert isinstance(figures, dict)
    for name, fig in figures.iteritems():
        fig.set_size_inches(size_inches)
        fig.savefig(name + format)


def main():

    figures = []

    config = CONFIG['BayesJF17_mildDR_Kaggle']

    y = load_labels(config['LABELS_FILE'])
    images = load_filenames(config['LABELS_FILE'])
    probs, probs_mc = load_predictions(config['predictions'])
    y_bin, probs_bin, probs_mc_bin = detection_task(
        y, probs, probs_mc, config['disease_onset'])
    pred_mean, pred_std = posterior_statistics(probs_mc_bin)
    uncertainties = {'$\sigma_{pred}$': pred_std}

    f = fig1(y_bin, pred_mean, images, pred_std, probs_mc_bin,
             y, config, label='$\sigma_{pred}$', save=True, format='.svg')
    figures.append(f)

    # figure 2
    f = acc_rejection_figure(y_bin, pred_mean, uncertainties, config,
                             save=True, format='.pdf')
    figures.append(f)

    # ROC figures for comparison of different architectures, tasks
    # and true generalization performance

    for name, config in CONFIG.iteritems():
        print('Working on %s...' % name)

        y = load_labels(config['LABELS_FILE'])
        probs, probs_mc = load_predictions(config['predictions'])
        y_bin, probs_bin, probs_mc_bin = detection_task(
            y, probs, probs_mc, config['disease_onset'])
        pred_mean, pred_std = posterior_statistics(probs_mc_bin)
        uncertainties = {'$\sigma_{pred}$': pred_std}

        f = roc_auc_rejection_figure(y_bin, pred_mean,
                                     uncertainties, config,
                                     save=False)
        figures.append(f)

        if config['dataset'] == 'Kaggle DR':
            f = level_rejection_figure(y, pred_std, config,
                                       save=False)
            figures.append(f)

            f = label_disagreement_figure(y_bin, pred_std, config,
                                          save=False)
            figures.append(f)

    f = train_test_generalization()
    figures.append(f)

    return figures


if __name__ == '__main__':
    figures = main()
