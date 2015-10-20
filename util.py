from __future__ import division
import numpy as np


def kappa(labels_rater_1, labels_rater_2, num_classes):
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
    kappa : scalar, values [-1,1]

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
