import unittest
from numpy.random import randint
from util import quadratic_weighted_kappa
from ref_quadratic_weighted_kappa import quadratic_weighted_kappa as ref_quadratic_weighted_kappa
from numpy.testing import assert_almost_equal


class TestQuadraticWeightedKappa(unittest.TestCase):

    def test_quadratic_weighted_kappa(self):
        size = 10
        human_labels = randint(5, size=size)
        predicted = randint(5, size=size)

        my_kappa = quadratic_weighted_kappa(human_labels, predicted, 5)
        test = ref_quadratic_weighted_kappa(human_labels, predicted)

        assert_almost_equal(my_kappa, test, 7)


if __name__ == '__main__':
    unittest.main()
