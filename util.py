from __future__ import division
import numpy as np
import bokeh.plotting as bp
import plotting
import seaborn as sns
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
            w[i][j] = ((i-j)/(num_classes - 1))**2
            e[i][j] = list(labels_rater_1).count(i) * \
                      list(labels_rater_2).count(j)/len(list(labels_rater_2))
            for ii, jj in zip(labels_rater_1, labels_rater_2):
                if ii == i and jj == j:
                    ob[i][j] += 1

    return 1 - sum(sum(np.multiply(w, ob)))/sum(sum(np.multiply(w, e)))


class Progplot(object):
    """Progression plot that monitors training of neural network"""

    def __init__(self, n_x, x_axis_label):
        """
        Parameters
        ----------
        n_x : int
            total number of expected samples in x-direction

        """

        self.n_x = n_x
        self.y = {}
        self.seen_so_far = 0
        self.output_file = bp.output_file("progplot.html",
                                          title="Progression plot")
        self.p = bp.figure(title="Monitor neural network training",
                           x_axis_label=x_axis_label)

    def save(self):
        """Add one line for each tracked quantity"""
        x = np.arange(self.n_x)
        n_lines = len(self.y.keys())
        colors = sns.color_palette(n_colors=n_lines).as_hex()
        for i, k in enumerate(sorted(self.y.keys())):
            self.p.line(x, self.y[k], line_color=colors[i], legend=k,
                        line_width=2)

        bp.save(self.p, self.output_file)

    def show(self):
        bp.show(self.p)

    def update(self, current, values=[]):
        """

        Parameters
        ----------
        current: int
            index of current step
        values: list of tuples (name, value_for_last_step)

        """

        for k, v in values:
            if k not in self.y.keys():
                self.y[k] = np.zeros(self.n_x)

            self.y[k][current] = v

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
    """Monitor loss and accuracy dynamically for training and validation
    data

    To be used together with keras as documented under
       http://keras.io/callbacks/

    """
    def __init__(self, history, show_accuracy=False):
        super(TrainingMonitor, self).__init__()
        self.loss_plot = plotting.LossPlot(1)
        self.history = history
        if show_accuracy:
            self.acc_plot = plotting.AccuracyPlot(2)

    def on_epoch_end(self, epoch, logs={}):
        train_loss = self.history.history['loss'][-1]
        val_loss = self.history.history['val_loss'][-1]
        self.loss_plot.plot(train_loss, val_loss, epoch)

        if hasattr(self, 'acc_plot'):
            train_acc = self.history.history['acc'][-1]
            val_acc = self.history.history['val_acc'][-1]
            self.acc_plot.plot(train_acc, val_acc, epoch)


class AdaptiveLearningRateScheduler(keras.callbacks.Callback):
    """Learning rate scheduler that decays learning rate by a step if
       validation loss stops improving.

       To be used together with keras as documented under
       http://keras.io/callbacks/
    """

    def __init__(self, initial_lr=0.1, decay=0.1, patience=20, verbose=0):
        super(AdaptiveLearningRateScheduler, self).__init__()
        assert type(initial_lr) == float, \
                    'The learning rate should be float.'
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