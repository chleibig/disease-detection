from __future__ import division
import numpy as np
import bokeh.plotting as bp
import bokeh.palettes


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
        if n_lines < 3:
            n_lines = 3
        if n_lines > 9:
            n_lines = 9

        colors = getattr(bokeh.palettes, "GnBu" + str(n_lines))
        for i, k in enumerate(sorted(self.y.keys())):
            self.p.line(x, self.y[k], line_color=colors[i % n_lines],
                        legend=k, line_width=2)

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
