from __future__ import division
from util import *


class PointProcess(object):
    """
    Generic class for point processes; methods to be overwritten by
        each specific class.
    """
    def __init__(self, dim, bounds):
        """
        Initializes a point process.

        Args:
            dim: int, dimension of domain.
            bounds: list of length dim, bounding box for each dimension.
        """
        self.dim = dim  # Dimension

        # Bounds
        assert_eq(len(bounds), dim)
        assert all(bound[0] < bound[1] for bound in bounds), bounds
        self.bounds = deepcopy(bounds)  # Rectangular boundaries

        return

    def intensity(self, x):
        """
        Intensity function at location x.

        Args:
            x: array(dim), a point location.
        """
        return

    def papangelou(self, u, X):
        """
        Papangelou conditional intensity function at location u given points X.

        Args:
            u: array(dim), a new point location.
            X: array((..., dim)), existing points in a sample.
        """
        return

    def sample(self, num_samples):
        """
        Draw samples.

        Args:
            num_samples: int, number of point process realizations.
        """
        return

    def check_valid_samples(self, samples):
        """
        Check valid samples.

        Args:
            samples: list of 2D-arrays.
        """
        low, high = zip(*self.bounds)

        for X in samples:
            if X.size == 0:  # Empty set
                continue
            X_min = np.min(X, axis=0)
            X_max = np.max(X, axis=0)
            assert np.all(low <= X_min) and np.all(X_max <= high), X

        return True
