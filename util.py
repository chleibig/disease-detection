from __future__ import division

from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import bokeh.plotting as bp
import bokeh.client as bc
from scipy import interpolate
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
import keras.callbacks
from keras import backend as K


def quadratic_weighted_kappa(labels_rater_1, labels_rater_2, num_classes):
    """
    Calculates the quadratic weighted kappa value,

    which is a measure of inter-rater agreement between two raters that provide
    discrete numeric ratings. Potential values range from -1 (representing
    complete disagreement) to 1 (representing complete agreement). A kappa
    value of 0 is expected if all agreement is due to chance.
    labels_rater_1 and labels_rater_2 each correspond to a list of integer
    ratings. These lists must have the same length. The ratings should be
    integers, and it is assumed that they contain the complete range of
    possible ratings.

    Parameters
    ----------
    labels_rater_1 : array-like, shape = (n_samples,)
        labels assigned by a human, values [0,num_classes]
    labels_rater_2 : array-like, shape = (n_samples,)
        labels assigned by the network
    num_classes : int
        number of classes

    Returns
    -------
    quadratic_weighted_kappa : scalar, values [-1,1]

    """

    w = np.zeros((num_classes, num_classes))
    e = np.zeros((num_classes, num_classes))
    ob = np.zeros((num_classes, num_classes))

    for i in np.arange(num_classes):
        for j in np.arange(num_classes):
            w[i][j] = ((i - j) / (num_classes - 1))**2
            e[i][j] = list(labels_rater_1).count(i) * \
                      list(labels_rater_2).count(j) / len(list(labels_rater_2))
            for ii, jj in zip(labels_rater_1, labels_rater_2):
                if ii == i and jj == j:
                    ob[i][j] += 1

    return 1 - sum(sum(np.multiply(w, ob))) / sum(sum(np.multiply(w, e)))


class Progplot(object):
    """Dynamically monitor training of neural network

    Usage
    =====

    prior to running code that uses the Progplot class, start a bokeh-server
    in a separate terminal running:

    bokeh serve

    The plot is then automatically shown in a new browser tab under
    (http://localhost:5006/?bokeh-session-id=disease-detection by default) and updated
    every epoch. For remote access, use
        ssh -L [bind_address:]port:host:hostport]

    """

    def __init__(self, n_x, x_axis_label, names, show=True,
                 title='Monitor neural network training'):
        """
        Parameters
        ----------
        n_x : int
            total number of expected samples in x-direction
        x_axis_label : string
        names : list of strings
            names of the quantities to be monitored
        show : bool (True by default)
            if True, browser window is automatically opened and dynamically
            updated
        title : str (default: 'Monitor neural network training')

        Notes
        -----

        The plot is served under:
            http://localhost:port/?bokeh-session-id=disease-detection

        """

        self.n_x = n_x
        self.seen_so_far = 0
        self.p = bp.figure(title=title,
                           x_axis_label=x_axis_label,
                           x_range=[0, n_x])
        self.y = {k: np.zeros(n_x) for k in names}
        self.x = np.arange(n_x)
        # Add one line for each tracked quantity:
        self.l = {}
        colors = sns.color_palette(n_colors=len(names)).as_hex()
        for i, k in enumerate(names):
            self.l[k] = self.p.line(x=[], y=[],
                                    line_color=colors[i], legend=k,
                                    line_width=2)
        self.curdoc = bp.curdoc()
        # This line is crucial as it adds the necessary on change callbacks:
        self.curdoc.add_root(self.p)
        self.session = bc.push_session(self.curdoc, session_id='disease-detection')
        if show:
            self.session.show()
        self.show = show

    def save(self, filename):
        self.output_file = bp.output_file(filename,
                                          title="Progression plot")
        bp.save(self.p, self.output_file)

    def show(self):
        self.session.show()

    def update(self, current, values=[]):
        """
        Parameters
        ----------
        current: int
            index of current step
        values: list of tuples (name, value_for_last_step)

        """

        for k, v in values:
            if k not in self.l.keys():
                raise KeyError('Name is not known to progplot instance.')

            self.y[k][current] = v
            self.l[k].data_source.data['x'] = self.x[:current]
            self.l[k].data_source.data['y'] = self.y[k][:current]

        self.seen_so_far = current

    def add(self, values=[]):
        """

        Parameters
        ----------
        values: list of tuples (name, value)

        """
        self.update(self.seen_so_far, values)
        self.seen_so_far += 1


class TrainingMonitor(keras.callbacks.Callback):
    """Monitor training and validation quantities

    By default, loss and val_loss are monitored, further quantities
    can be provided to the constructor

    To be used together with keras as documented under
       http://keras.io/callbacks/

    For further usage instructions see the documentation of the Progplot class

    """
    def __init__(self, n_epochs, batch_size, title='Disease detection monitor',
                 quantities=['loss', 'val_loss']):
        super(TrainingMonitor, self).__init__()
        self.progplot = Progplot(n_epochs,
                                 "epochs (batch_size " + str(batch_size) + ")",
                                 names=quantities,
                                 title=title)
        self.quantities = quantities

    def on_epoch_end(self, epoch, logs={}):
        self.progplot.add(values=[(q, logs.get(q)) for q in self.quantities])


class AdaptiveLearningRateScheduler(keras.callbacks.Callback):
    """Learning rate scheduler that decays learning rate by a step if
       validation loss stops improving.

       To be used together with keras as documented under
       http://keras.io/callbacks/
    """

    def __init__(self, initial_lr=0.1, decay=0.1, patience=20, verbose=0):
        super(AdaptiveLearningRateScheduler, self).__init__()
        assert type(initial_lr) == float, 'The learning rate should be float.'
        self.lr = initial_lr
        self.decay = decay
        self.patience = patience
        self.verbose = verbose
        self.best = np.Inf
        self.wait = 0

    def on_epoch_begin(self, epoch, logs={}):
        assert hasattr(self.model.optimizer, 'lr'), \
            'Optimizer must have a "lr" attribute.'

        current = logs.get('val_loss')
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
        else:
            if self.wait < self.patience:
                self.wait += 1
            else:
                self.lr *= self.decay
                K.set_value(self.model.optimizer.lr, self.lr)
                self.wait = 0
                if self.verbose > 0:
                    print('Epoch {}: lower learning rate to {}'
                          .format(epoch, self.lr))


class SelectiveSampler(object):
    """Selective sampling of informative instances

    Inspired by: Grinsven et al. (2016): "Fast convolutional neural network
    training using selective data sampling: Application to hemorrhage
    detection in color fundus images"

    """

    def __init__(self, M, y):
        """

        Parameters
        ==========

        M : int
            The number of samples to draw from each class.
        y : array_like, 1D, int
            class labels of all samples

        """
        # The following check throws a TypeError: 'int' object is not iterable?
        # assert set(len(np.lib.arraysetops.unique(y))) == {0, 1}, \
        #     'Labels have to be in {0, 1}.'
        self.M = M
        self.y = y
        self.Xpos = np.where(y == 1)[0]
        self.Xneg = np.where(y == 0)[0]

    def sample(self, probs_neg=None, shuffle=True):
        """Selective or random sampling with replacement

        Parameters
        ==========

        probs_neg : array_like, should correspond to the True entries of y == 0
            probabilities for all 'negative' (<-> 0) samples. These are used
            to assign selective sampling weights. If None, random sampling is
            performed.
        shuffle : bool (True by default)
            If True, indices are shuffled before they are returned

        Returns
        =======

        indices : array_like, int, of length 2*self.M with entries from the
                  interval [0, len(self.y)-1]

        """
        if probs_neg is not None:
            assert len(probs_neg) == self.Xneg.shape[0]
            # step 4
            # weights should be low for correct examples:
            weights = np.abs(probs_neg - 1)  # paper: - l_i = - 0?!
            sample_probs = weights / weights.sum()
            # step 5
            Xpos_t = self._random_sample('pos')
            Xneg_t = self._selective_sample(sample_probs)
        else:
            # We sample uniformly as predictions are not available (e.g. in
            # the first run)
            Xpos_t = self._random_sample('pos')
            Xneg_t = self._random_sample('neg')

        indices = np.concatenate((Xpos_t, Xneg_t))

        if shuffle:
            np.random.shuffle(indices)
        return indices

    def _random_sample(self, case='neg'):
        if case == 'pos':
            selection = np.random.randint(low=0,
                                          high=len(self.Xpos),
                                          size=self.M)
            return self.Xpos[selection]
        if case == 'neg':
            selection = np.random.randint(low=0,
                                          high=len(self.Xneg),
                                          size=self.M)
            return self.Xneg[selection]

    def _selective_sample(self, sample_probs):
        frequencies = np.random.multinomial(self.M, sample_probs, size=1)[0]
        selection = np.zeros((self.M,), dtype=np.int32)
        pos = 0
        for idx, freq in enumerate(frequencies):
            selection[pos:pos + freq] = idx
            pos += freq

        return self.Xneg[selection]


def roc_curve_plot(y_true, y_score, pos_label=1,
                   legend_prefix='', plot_BDA=False, n_bootstrap=100,
                   color=None):
    """Compute and plot receiver operating characteristic (ROC)

    Parameters
    ==========

    y_true : array, shape = [n_samples]
        True binary labels in range {0, 1} or {-1, 1}.  If labels are not
        binary, pos_label should be explicitly given.

    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive
        class or confidence values.

    pos_label : int
        Label considered as positive and others are considered negative.

    legend_prefix : string, by default empty
        plot legend: 'legend_prefix (auc=XX)''

    plot_BDA : boolean, False by default
        recommendation of British Diabetic Association

    """
    assert y_score.ndim == 1, 'y_score should be of shape (n_samples,)'
    assert len(y_true) == len(y_score), \
        'y_true and y_score must both be n_samples long'

    low, high = bootstrap([y_true, y_score], roc_auc_score,
                          n_resamples=n_bootstrap, alpha=0.05)

    fdr_low, tdr_low, _ = roc_curve(y_true[low.index],
                                    y_score[low.index],
                                    pos_label=pos_label)
    fdr_high, tdr_high, _ = roc_curve(y_true[high.index],
                                      y_score[high.index],
                                      pos_label=pos_label)
    interpolate_low = interpolate.interp1d(fdr_low, tdr_low, kind='nearest')
    interpolate_high = interpolate.interp1d(fdr_high, tdr_high, kind='nearest')

    fdr, tdr, _ = roc_curve(y_true, y_score,
                            pos_label=pos_label)
    roc_auc = roc_auc_score(y_true, y_score)

    plt.plot(fdr, tdr, color=color,
             label=legend_prefix + ' (auc:%0.3f; CI:%0.3f-%0.3f)'
             % (roc_auc, low.value, high.value))
    plt.fill_between(fdr, interpolate_high(fdr), tdr, color=color, alpha=0.3)
    plt.fill_between(fdr, tdr, interpolate_low(fdr), color=color, alpha=0.3)
    plt.plot([0, 1], [0, 1], 'k--')
    if plot_BDA:
        plt.scatter([0.05], [0.8], color='g', s=50,
                    label='recommendation British Diabetic Association')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('false diseased rate (1 - specificity)')
    plt.ylabel('true diseased rate (sensitivity)')
    plt.legend(loc="lower right")


def bootstrap(data, fun, n_resamples=10000, alpha=0.05):
    """Compute confidence interval for values of function fun

    Parameters
    ==========
    data: list of arguments to fun

    """
    assert isinstance(data, list)
    n_samples = len(data[0])
    idx = np.random.randint(0, n_samples, (n_resamples, n_samples))

    def select(data, sample):
        return [d[sample] for d in data]

    values = np.array([fun(*select(data, sample)) for sample in idx])

    idx = idx[np.argsort(values, axis=0)]
    values = np.sort(values, axis=0)

    stat = namedtuple('stat', ['value', 'index'])
    low = stat(value=values[int((alpha/2.0)*n_resamples)],
               index=idx[int((alpha/2.0)*n_resamples)])
    high = stat(value=values[int((1-alpha/2.0)*n_resamples)],
                index=idx[int((1-alpha/2.0)*n_resamples)])

    return low, high
