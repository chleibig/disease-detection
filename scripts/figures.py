"""Analysis of how the rejection of test samples based on model uncertainty
   affects the performance of the rest of the data that remains automatically
   classified"""
from __future__ import print_function
from collections import OrderedDict
import h5py
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
from util import balance_classes

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

plt.ion()
sns.set_context('paper', font_scale=1.5)
sns.set_style('whitegrid')

FIGURE_WIDTH = 8.27  # 8.27 inch corresponds to A4

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
    'BCNN_mildDR_Kaggle_train': dict(
        [('net', 'BCNN'),
         ('dataset', 'Kaggle train'),
         ('predictions', 'data/processed/'
          '100_mc_KaggleDR_train_BayesJFnet17_392bea6.pkl'),
         ('disease_onset', 1)] +
        DATA['KaggleDR_train'].items()),

    'BCNN_mildDR_Kaggle': dict(
        [('net', 'BCNN'),
         ('dataset', 'Kaggle'),
         ('predictions', 'data/processed/'
          '100_mc_KaggleDR_test_BayesJFnet17_392bea6.pkl'),
         ('predictions_gp', 'data/processed/'
          'GP/GPC_Results_MINIBATCH_KaggleDR_Onset1.mat'),
         ('disease_onset', 1)] +
        DATA['KaggleDR'].items()),

    'BCNN_moderateDR_Kaggle': dict(
        [('net', 'BCNN'),
         ('dataset', 'Kaggle'),
         ('predictions', 'data/processed/'
          '100_mc_KaggleDR_test_bcnn2_b69aadd.pkl'),
         ('predictions_gp', 'data/processed/'
          'GP/GPC_Results_MINIBATCH_KaggleDR_Onset2.mat'),
         ('disease_onset', 2)] +
        DATA['KaggleDR'].items()),

    'JFnet_mildDR_Kaggle': dict(
        [('net', 'JFnet'),
         ('dataset', 'Kaggle'),
         ('predictions', 'data/processed/'
          'c9ade47_100_mc_KaggleDR_test_JFnet.pkl'),
         ('disease_onset', 1)] +
        DATA['KaggleDR'].items()),

    'JFnet_moderateDR_Kaggle': dict(
        [('net', 'JFnet'),
         ('dataset', 'Kaggle'),
         ('predictions', 'data/processed/'
          'c9ade47_100_mc_KaggleDR_test_JFnet.pkl'),
         ('disease_onset', 2)] +
        DATA['KaggleDR'].items()),

    'BCNN_mildDR_Messidor': dict(
        [('net', 'BCNN'),
         ('dataset', 'Messidor'),
         ('predictions', 'data/processed/'
          '100_mc_Messidor_BayesJFnet17_392bea6.pkl'),
         ('predictions_gp', 'data/processed/'
          'GP/GPC_Results_MINIBATCH_Messidor_Onset1.mat'),
         ('disease_onset', 1)] +
        DATA['Messidor'].items()),

    'BCNN_moderateDR_Messidor': dict(
        [('net', 'BCNN'),
         ('dataset', 'Messidor'),
         ('predictions', 'data/processed/'
          '100_mc_Messidor_BayesianJFnet17_onset2_b69aadd.pkl'),
         ('predictions_gp', 'data/processed/'
          'GP/GPC_Results_MINIBATCH_Messidor_Onset2.mat'),
         ('disease_onset', 2)] +
        DATA['Messidor'].items())
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
    assert ((0.0 <= probs) & (probs <= 1.0 + 1e-6)).all()
    assert ((0.0 <= probs_mc) & (probs_mc <= 1.0 + 1e-6)).all()
    return probs, probs_mc


def load_predictions_gp(filename):
    """Load mat files from Murat Seckin Ayhan <msayhan@gmail.com>r"""
    with h5py.File(filename, 'r') as f:
        if 'KaggleDR' in filename:
            probs = f.get('probs_te')[0, :]
        elif 'Messidor' in filename:
            probs = np.exp(f.get('lp_MESS')[0, :])
        else:
            raise ValueError('Unsupported file: {}'.format(filename))
    return probs


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


def binary_entropy(p):
    assert p.ndim == 1
    return -(p * np.log2(p + 1e-6) + (1 - p) * np.log2((1 - p) + 1e-6))


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
    assert (0.0 <= predictive_std).all()
    return predictive_mean, predictive_std


def argmax_labels(probs):
    return (probs >= 0.5).astype(int)


def accuracy(y_true, probs):
    y_pred = argmax_labels(probs)
    assert len(y_true) == len(y_pred)
    return (y_true == y_pred).sum() / float(len(y_true))


def rel_freq(y, k):
    return (y == k).sum() / float(len(y))


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


def sample_rejection(uncertainty, min_percentile,
                     maximum=None):
    if maximum is None:
        maximum = uncertainty.max()
    uncertainty_tol = np.linspace(np.percentile(uncertainty, min_percentile),
                                  maximum, 100)
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
        fig = plt.figure(figsize=(FIGURE_WIDTH,
                                  FIGURE_WIDTH / 2.0))

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
    ax122.set_xlim(0.1, 1.0)
    ax121.set_xlabel('tolerated model uncertainty')
    ax121.set_ylabel('accuracy')
    ax121.legend(loc='best')

    ax122.plot(frac_retain, acc_rand['value'], label='random referral',
               color=colors[i+1], linewidth=2)
    ax122.fill_between(frac_retain, acc_rand['value'], acc_rand['low'],
                       color=colors[i+1], alpha=0.3)
    ax122.fill_between(frac_retain, acc_rand['high'], acc_rand['value'],
                       color=colors[i+1], alpha=0.3)
    ax122.set_xlabel('retained data')
    ax122.legend(loc='best')

    sns.despine(offset=10, trim=True)

    name = 'acc_' + config['net'] + '_' + str(config['disease_onset']) + \
           '_' + config['dataset']

    if save:
        fig.savefig(name + format)

    return {name: fig}


def level_subplot(y_level, uncertainty, config,
                  ax=None):
    tol, frac_retain, accept_idx = sample_rejection(uncertainty, 0)
    LEVEL = config['LEVEL']
    p = {level: np.array([rel_freq(y_level[~accept], level)
                          for accept in accept_idx])
         for level in LEVEL}
    cum = np.zeros_like(tol)

    with sns.axes_style('white'):
        ax.set_title('Disease onset: %s'
                     % ONSET_TAG[config['disease_onset']])

        colors = {level: sns.color_palette("Blues")[level] for level in LEVEL}
        for level in LEVEL:
            ax.fill_between(tol, p[level] + cum, cum,
                            color=colors[level],
                            label='%d: %s' % (level, LEVEL[level]))
            if (level + 1) == config['disease_onset']:
                ax.plot(tol, p[level] + cum,
                        color='k', label='healthy/diseased boundary')
            cum += p[level]

        ax.set_xlim(min(tol), max(tol))
        ax.set_ylim(0, 1)

        ax.set_xlabel('tolerated model uncertainty')
        ax.set_ylabel('relative proportions within referred dataset')
        ax.legend(loc='lower left')


def level_figure():
    keys = ['BCNN_mildDR_Kaggle',
            'BCNN_moderateDR_Kaggle']
    title_prefix = ['(a)', '(b)']
    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_WIDTH / 2.0))
    for i, k in enumerate(keys):
        config = CONFIG[k]
        y = load_labels(config['LABELS_FILE'])
        probs, probs_mc = load_predictions(config['predictions'])
        _, _, probs_mc_bin = detection_task(y, probs, probs_mc,
                                            config['disease_onset'])
        _, pred_std = posterior_statistics(probs_mc_bin)

        ax = fig.add_subplot(1, 2, i + 1)
        level_subplot(y, pred_std, config, ax=ax)
        ax.set_title(title_prefix[i] + ' ' + ax.get_title())
        if i == 1:
            ax.set_ylabel('')

    return {'level': fig}


def label_disagreement_subplot(y, uncertainty, config, ax=None):
    try:
        disagreeing = ~contralateral_agreement(y, config)
    except TypeError:
        print('No data for label disagreement figure available.')
        return

    tol, frac_retain, accept_idx = sample_rejection(uncertainty, 0.1)

    p_referred = np.array([sum((~accept) & (disagreeing)) / float(sum(~accept))
                           for accept in accept_idx])
    p_retained = np.array([sum((accept) & (disagreeing)) / float(sum(accept))
                           for accept in accept_idx])

    with sns.axes_style('white'):
        ax.fill_between(tol, p_referred, 0, alpha=0.5,
                        color=sns.color_palette()[0], label='referred')
        ax.fill_between(tol, p_retained, 0, alpha=0.5,
                        color=sns.color_palette()[1], label='retained')

        ax.set_xlim(min(tol), max(tol))
        ax.set_ylim(0, 1)

        ax.set_xlabel('tolerated model uncertainty')
        ax.set_ylabel('fraction of data with patient level ambiguity')
        ax.legend(loc='upper left')


def label_disagreement_figure():
    keys = ['BCNN_mildDR_Kaggle',
            'BCNN_moderateDR_Kaggle']
    titles = ['(a) Disease onset: mild DR',
              '(b) Disease onset: moderate DR']
    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_WIDTH / 2.0))

    for i, k in enumerate(keys):
        config = CONFIG[k]
        y = load_labels(config['LABELS_FILE'])
        probs, probs_mc = load_predictions(config['predictions'])
        y_bin, _, probs_mc_bin = detection_task(y, probs, probs_mc,
                                                config['disease_onset'])
        _, pred_std = posterior_statistics(probs_mc_bin)

        ax = fig.add_subplot(1, 2, i + 1)
        label_disagreement_subplot(y_bin, pred_std, config, ax=ax)
        ax.set_title(titles[i])
        if i == 1:
            ax.set_ylabel('')
        ax.set_ylim(0, 0.2)

    return {'label_disagreement': fig}


def roc_auc_subplot(y, y_score, uncertainties, config,
                    save=False, format='.svg',
                    ax121=None, ax122=None):
    if ax121 is None or ax122 is None:
        fig = plt.figure(figsize=(FIGURE_WIDTH,
                                  FIGURE_WIDTH / 2.0))
        ax121 = plt.subplot2grid((1, 2), (0, 0))
        ax122 = plt.subplot2grid((1, 2), (0, 1))

    colors = sns.color_palette()

    ax121.set_title('(a) %s(onset: %s); %s'
                    % (config['net'], ONSET_TAG[config['disease_onset']],
                       config['dataset']))
    ax122.set_title('(b) %s(onset: %s); %s'
                    % (config['net'], ONSET_TAG[config['disease_onset']],
                       config['dataset']))

    for i, (k, v) in enumerate(uncertainties.iteritems()):
        filename = 'data/processed/bt' + str(config['n_bootstrap']) + '_' + \
            k.replace(' ', '') + '_' + config['net'] + '_' + \
            str(config['disease_onset']) + '_' + config['dataset'] + '.npz'
        if os.path.exists(filename):
            data = np.load(filename)
            v_tol, frac_retain, auc, auc_rand = data['v_tol'], \
                data['frac_retain'], data['auc'], data['auc_rand']
        else:
            v_tol, frac_retain, auc, auc_rand = \
                performance_over_uncertainty_tol(v, y, y_score,
                                                 roc_auc_score,
                                                 config['min_percentile'],
                                                 config['n_bootstrap'])
            np.savez(filename, v_tol=v_tol, frac_retain=frac_retain,
                     auc=auc, auc_rand=auc_rand)

        ax121.plot(frac_retain, auc['value'],
                   label=k, color=colors[i], linewidth=2)
        ax121.fill_between(frac_retain, auc['value'], auc['low'],
                           color=colors[i], alpha=0.3)
        ax121.fill_between(frac_retain, auc['high'], auc['value'],
                           color=colors[i], alpha=0.3)

        if k == 'MC dropout':
            # plot exemplary roc curves
            ax122
            fractions = [0.9, 0.8, 0.7]
            for j, f in enumerate(fractions):
                thr = v_tol[frac_retain >= f][0]
                roc_curve_plot(y[v <= thr],
                               y_score[v <= thr],
                               color=colors[j+4],
                               legend_prefix='%d%% data retained, %s'
                               % (f * 100, k),
                               n_bootstrap=config['n_bootstrap'])
        else:
            # print confidence intervals for table 1
            fractions = [0.9, 0.8, 0.7]
            for j, f in enumerate(fractions):
                thr = v_tol[frac_retain >= f][0]
                low, high = bootstrap([y[v <= thr], y_score[v <= thr]],
                                      roc_auc_score,
                                      n_resamples=config['n_bootstrap'],
                                      alpha=0.05)
                roc_auc = roc_auc_score(y[v <= thr], y_score[v <= thr])
                msg = '%d%% data retained, %s' % (f * 100, k)
                msg += ' (auc:%0.3f; CI:%0.3f-%0.3f)' \
                    % (roc_auc, low.value, high.value)
                print(msg)

    ax121.plot(frac_retain, auc_rand['value'],
               label='random referral', color=colors[i+1], linewidth=2)
    ax121.fill_between(frac_retain, auc_rand['value'], auc_rand['low'],
                       color=colors[i+1], alpha=0.3)
    ax121.fill_between(frac_retain, auc_rand['high'], auc_rand['value'],
                       color=colors[i+1], alpha=0.3)
    ax121.set_xlim(config['min_percentile'] / 100., 1.0)
    ax121.set_xlabel('retained data')
    ax121.set_ylabel('auc')
    ax121.legend(loc='best')

    ax122
    roc_curve_plot(y, y_score, color=colors[-1],
                   legend_prefix='no referral',
                   recommendation=True,
                   n_bootstrap=config['n_bootstrap'])

    x0, x1 = ax121.get_xlim()
    y0, y1 = ax121.get_ylim()
    ax121.set_aspect((x1 - x0) / (y1 - y0))
    x0, x1 = ax122.get_xlim()
    y0, y1 = ax122.get_ylim()
    ax122.set_aspect((x1 - x0) / (y1 - y0))

    if save and fig is not None:
        name = 'roc_' + config['net'] + '_' + str(config['disease_onset']) + \
            '_' + config['dataset']
        fig.savefig(name + format)
        return {name: fig}


def roc_auc_figure():
    keys = ['BCNN_mildDR_Kaggle',
            'BCNN_moderateDR_Kaggle',
            'BCNN_mildDR_Messidor',
            'BCNN_moderateDR_Messidor']
    titles = ['(a)', '(b)', '(c)', '(d)']
    fig = plt.figure()
    for i, k in enumerate(keys):
        config = CONFIG[k]
        print('Working on %s...' % k)
        y = load_labels(config['LABELS_FILE'])
        probs, probs_mc = load_predictions(config['predictions'])
        y_bin, probs_bin, probs_mc_bin = detection_task(
            y, probs, probs_mc, config['disease_onset'])
        pred_mean, pred_std = posterior_statistics(probs_mc_bin)
        probs_gp = load_predictions_gp(config['predictions_gp'])
        uncertainties = {'MC dropout': pred_std,
                         'GP': binary_entropy(probs_gp),
                         'standard dropout': binary_entropy(probs_bin)}

        ax121 = plt.subplot(2, 4, 2 * i + 1)
        ax122 = plt.subplot(2, 4, 2 * i + 2)
        roc_auc_subplot(y_bin, pred_mean,
                        uncertainties, config,
                        save=False,
                        ax121=ax121,
                        ax122=ax122)
        ax121.set_title('')
        ax121.set_title(titles[i], loc='left')
        ax122.set_title('')
        ax122.set_xlim(0.0, 0.5)
        ax122.set_ylim(0.5, 1.0)

    sns.despine(offset=10)
    plt.tight_layout()

    return {'roc_auc': fig}


def train_test_generalization():
    """Visualizes performance over uncertainty for both train and test data"""
    fig = plt.figure(figsize=(FIGURE_WIDTH / 2.0,
                              FIGURE_WIDTH / 2.0))
    ax = fig.gca()
    colors = sns.color_palette()

    configs = {'$\sigma_{pred} (train)$':
               CONFIG['BCNN_mildDR_Kaggle_train'],
               '$\sigma_{pred} (test)$':
               CONFIG['BCNN_mildDR_Kaggle']}

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

    sns.despine(offset=10, trim=True)

    name = 'train_test_' + config['net'] + '_' + \
        str(config['disease_onset']) + '_' + config['dataset']

    fig.savefig(name + '.pdf')

    return {name: fig}


def error_conditional_uncertainty(y, y_score, uncertainty, disease_onset,
                                  label='pred_std', ax=None):
    """Plot conditional pdfs for correct and erroneous argmax predictions"""
    if ax is None:
        ax = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_WIDTH)).gca()

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
    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_WIDTH))

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
                     (0.2, 0.75 * y_pos))
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

    sns.despine(offset=10, trim=True)

    name = 'fig1_' + config['net'] + '_' + str(config['disease_onset']) + \
        '_' + config['dataset']

    if save:
        fig.savefig(name + format)

    return {name: fig}


def prediction_vs_uncertainty(y, uncertainty, prediction,
                              title='', n_levels=250, balance=False,
                              ax121=None, ax122=None):
    ylabel = uncertainty.keys()[0]
    uncertainty = uncertainty.values()[0]
    xlabel = prediction.keys()[0]
    prediction = prediction.values()[0]

    if balance:
        y, (uncertainty, prediction) = balance_classes(y, [uncertainty,
                                                           prediction])

    error = (y != (prediction >= 0.5))

    plt.suptitle(title)

    if ax121 is None:
        plt.subplot(1, 2, 1)

    ax121.set_title('(a) correct')
    sns.kdeplot(prediction[~error], uncertainty[~error],
                n_levels=n_levels, ax=ax121)
    ax121.set_ylabel(ylabel)
    ax121.set_xlabel(xlabel)
    ax121.set_xlim(0, 1.0)
    ax121.set_ylim(0, 0.25)

    if ax122 is None:
        plt.subplot(1, 2, 2)

    ax122.set_title('(b) error')
    sns.kdeplot(prediction[error], uncertainty[error],
                n_levels=n_levels, ax=ax122)
    ax122.set_ylabel(ylabel)
    ax122.set_xlabel(xlabel)
    ax122.set_xlim(0, 1.0)
    ax122.set_ylim(0, 0.25)

    sns.despine(offset=10, trim=True)


def bayes_vs_softmax():
    config = CONFIG['BCNN_moderateDR_Kaggle']
    y = load_labels(config['LABELS_FILE'])
    probs, probs_mc = load_predictions(config['predictions'])
    y_bin, probs_bin, probs_mc_bin = detection_task(y, probs, probs_mc,
                                                    config['disease_onset'])
    _, pred_std = posterior_statistics(probs_mc_bin)
    uncertainty = {'$\sigma_{pred}$': pred_std}
    prediction = {'p(diseased | image)': probs_bin}

    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_WIDTH / 2.0))
    prediction_vs_uncertainty(y_bin, uncertainty, prediction,
                              title='', n_levels=250)
    name = 'sigma_vs_soft_' + config['net'] + '_' + \
        str(config['disease_onset']) + '_' + config['dataset']
    return {name: fig}


def sigma_vs_mu():
    config = CONFIG['BCNN_moderateDR_Kaggle']
    y = load_labels(config['LABELS_FILE'])
    probs, probs_mc = load_predictions(config['predictions'])
    y_bin, probs_bin, probs_mc_bin = detection_task(y, probs, probs_mc,
                                                    config['disease_onset'])
    pred_mean, pred_std = posterior_statistics(probs_mc_bin)
    uncertainty = {'$\sigma_{pred}$': pred_std}
    prediction = {'$\mu_{pred}$': pred_mean}

    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_WIDTH / 2.0))
    ax131 = plt.subplot(131)
    ax132 = plt.subplot(132)
    prediction_vs_uncertainty(y_bin, uncertainty, prediction,
                              title='', n_levels=250, ax121=ax131, ax122=ax132)

    ax133 = plt.subplot(133)
    ax133.set_title('(c) Decision referral')
    colors = sns.color_palette()

    uncertainties = OrderedDict([('$\sigma_{pred}$', pred_std),
                                 ('$H(\mu_{pred})$',
                                  binary_entropy(pred_mean)),
                                 ('$H(p(diseased|image))$',
                                  binary_entropy(probs_bin))])
    for i, (k, v) in enumerate(uncertainties.iteritems()):
        v_tol, frac_retain, auc, auc_rand = \
            performance_over_uncertainty_tol(v, y_bin, pred_mean,
                                             roc_auc_score,
                                             config['min_percentile'],
                                             config['n_bootstrap'])
        ax133.plot(frac_retain, auc['value'],
                   label=k, color=colors[i], linewidth=2)
        ax133.fill_between(frac_retain, auc['value'], auc['low'],
                           color=colors[i], alpha=0.3)
        ax133.fill_between(frac_retain, auc['high'], auc['value'],
                           color=colors[i], alpha=0.3)

    ax133.plot(frac_retain, auc_rand['value'],
               label='random referral', color=colors[i + 1], linewidth=2)
    ax133.fill_between(frac_retain, auc_rand['value'], auc_rand['low'],
                       color=colors[i + 1], alpha=0.3)
    ax133.fill_between(frac_retain, auc_rand['high'], auc_rand['value'],
                       color=colors[i + 1], alpha=0.3)
    ax133.set_xlim(config['min_percentile'] / 100., 1.0)
    ax133.set_xlabel('retained data')
    ax133.set_ylabel('auc')
    ax133.legend(loc='best')

    for ax in [ax131, ax132, ax133]:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_aspect((x1 - x0) / (y1 - y0))

    sns.despine(offset=10, trim=True)

    return {'sigma_vs_mu': fig}


def gp_figure():

    def load(config_bcnn, config_gp):
        y = load_labels(config_bcnn['LABELS_FILE'])
        probs, probs_mc = load_predictions(config_bcnn['predictions'])
        y_bin, _, probs_mc_bin = detection_task(y, probs, probs_mc,
                                                config_bcnn['disease_onset'])
        pred_mean, _ = posterior_statistics(probs_mc_bin)

        probs_gp = load_predictions_gp(config_gp['predictions'])

        return y_bin, pred_mean, probs_gp

    def auc_plot(y_bin, pred_mean, probs_gp, ax=None, min_percentile=None,
                 n_bootstrap=None):
        traces = OrderedDict([('$H(\mu_{pred})$',
                               (pred_mean, binary_entropy(pred_mean))),
                              ('$H(p_{GP})$',
                               (probs_gp, binary_entropy(probs_gp)))])
        colors = sns.color_palette()
        for i, (k, (p, u)) in enumerate(traces.iteritems()):
            v_tol, frac_retain, auc, auc_rand = \
                performance_over_uncertainty_tol(u, y_bin, p,
                                                 roc_auc_score,
                                                 min_percentile,
                                                 n_bootstrap)
            ax.plot(frac_retain, auc['value'],
                    label=k, color=colors[i], linewidth=2)
            ax.fill_between(frac_retain, auc['value'], auc['low'],
                            color=colors[i], alpha=0.3)
            ax.fill_between(frac_retain, auc['high'], auc['value'],
                            color=colors[i], alpha=0.3)

        ax.plot(frac_retain, auc_rand['value'],
                label='random referral', color=colors[i + 1], linewidth=2)
        ax.fill_between(frac_retain, auc_rand['value'], auc_rand['low'],
                        color=colors[i + 1], alpha=0.3)
        ax.fill_between(frac_retain, auc_rand['high'], auc_rand['value'],
                        color=colors[i + 1], alpha=0.3)
        ax.set_xlim(min_percentile / 100., 1.0)
        ax.set_xlabel('retained data')
        ax.set_ylabel('auc')
        ax.legend(loc='best')

    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_WIDTH / 2.0))

    ax221 = plt.subplot(221)
    ax221.set_title('(a) Disease onset: mild DR')
    y, probs_bcnn, probs_gp = load(CONFIG['BCNN_mildDR_Kaggle'],
                                   CONFIG['GP_mildDR_Kaggle'])
    auc_plot(y, probs_bcnn, probs_gp, ax=ax221,
             min_percentile=DATA['KaggleDR']['min_percentile'],
             n_bootstrap=DATA['KaggleDR']['n_bootstrap'])

    ax223 = plt.subplot(223)
    ax223.set_title('(c) Disease onset: mild DR')
    sns.kdeplot(binary_entropy(probs_bcnn), binary_entropy(probs_gp),
                n_levels=250, ax=ax223)
    ax223.set_ylabel('GP uncertainty [$H(p_{GP})$]')
    ax223.set_xlabel('BCNN uncertainty [$H(\mu_{pred})$]')

    ax222 = plt.subplot(222)
    ax222.set_title('(b) Disease onset: moderate DR')
    y, probs_bcnn, probs_gp = load(CONFIG['BCNN_moderateDR_Kaggle'],
                                   CONFIG['GP_moderateDR_Kaggle'])
    auc_plot(y, probs_bcnn, probs_gp, ax=ax222,
             min_percentile=DATA['KaggleDR']['min_percentile'],
             n_bootstrap=DATA['KaggleDR']['n_bootstrap'])

    ax224 = plt.subplot(224)
    ax224.set_title('(d) Disease onset: moderate DR')
    sns.kdeplot(binary_entropy(probs_bcnn), binary_entropy(probs_gp),
                n_levels=250, ax=ax224)
    ax224.set_ylabel('GP uncertainty [$H(p_{GP})$]')
    ax224.set_xlabel('BCNN uncertainty [$H(\mu_{pred})$]')

    for ax in [ax221, ax222, ax223, ax224]:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_aspect((x1 - x0) / (y1 - y0))

    sns.despine(offset=10, trim=True)

    return {'gp_figure': fig}


def class_conditional_uncertainty(y, uncertainty, disease_onset,
                                  save=False, format='.svg'):
    plt.figure(figsize=(FIGURE_WIDTH / 2.0, FIGURE_WIDTH / 2.0))
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

    config = CONFIG['BCNN_mildDR_Kaggle']

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

    f = bayes_vs_softmax()
    figures.append(f)

    f = acc_rejection_figure(y_bin, pred_mean, uncertainties, config,
                             save=True, format='.pdf')
    figures.append(f)

    # ROC figure for comparison of different architectures, tasks
    # and true generalization performance

    f = roc_auc_figure()
    figures.append(f)

    f = level_figure()
    figures.append(f)

    f = label_disagreement_figure()
    figures.append(f)

    f = sigma_vs_mu()
    figures.append(f)

    f = gp_figure()
    figures.append(f)

    f = train_test_generalization()
    figures.append(f)

    return figures


if __name__ == '__main__':
    figures = main()
