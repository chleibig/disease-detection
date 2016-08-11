import numpy as np
from scripts import rejection


def test_stratified_mask():
    y = np.array([0, 0, 1, 0, 1, 1, 0, 1, 0, 0])
    y_prior = y[2:8]
    k_n = {k: (y_prior == k).sum() for k in range(2)}
    mask = rejection.stratified_mask(y, y_prior, shuffle=False)
    mask_shuffle = rejection.stratified_mask(y, y_prior, shuffle=True)
    assert not (mask == mask_shuffle).all(), \
        'It is ok if this assertion fails every now and then.'
    for k, n in k_n.iteritems():
        assert (y[mask] == k).sum() == n
        assert (y[mask_shuffle] == k).sum() == n
