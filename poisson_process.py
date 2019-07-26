from __future__ import division
from util import *
from point_process import PointProcess


class PoissonProcess(PointProcess):
    """
    Poisson point process.
    """
    def __init__(self, dim, bounds, intensity, homogeneous=False,
                 max_intensity=None):
        """
        Initializes a Poisson point process.

        Args:
            dim: int, dimension of domain.
            bounds: list of length dim, bounding box for each dimension.
            intensity: float/int if homogeneous == True else callable,
                intensity/rate (function) of the Poisson process.
            homogeneous: boolean, whether the Poisson process is homogeneous.
            max_intensity: float, maximum value of the intensity function on
                the domain; must be specified if homogeneous == False.
        """
        super(PoissonProcess, self).__init__(dim, bounds)

        self.homogeneous = homogeneous
        if self.homogeneous:  # Homogeneous process
            assert isinstance(intensity, float) or isinstance(intensity, int)
            self.intensity = lambda x: np.array(intensity)  # TODO: Check ...
            self.max_intensity = intensity
        else:  # Inhomogeneous
            assert callable(intensity)
            self.intensity = intensity
            assert max_intensity is not None
            self.max_intensity = max_intensity

        return

    def papangelou(self, u, X):
        """
        Papangelou conditional intensity function at location u given points X.

        Args:
            u: array(dim), a new point location.
            X: array((..., dim)), existing points in a sample.
        """
        # NOTE: Assumes u is within the domain (self.bounds).
        return self.intensity(u)  # Does not depend on existing points X

    def sample(self, num_samples):
        """
        Draw samples.

        Returns:
            samples: list of 2D-arrays of shape (num_pts, dim).
        """
        samples = list()  # list of sets for point clouds

        # Draw number of points in each sample
        low, high = zip(*self.bounds)
        area = np.prod([high[i] - low[i] for i in xrange(self.dim)])
        max_intensity = self.max_intensity + 1e-10  # Prevent probs > 1
        mean_num = max_intensity * area  # \Lambda
        nums = rand.poisson(lam=mean_num, size=num_samples)

        for i in xrange(num_samples):
            # Sample points from homogeneous PP with max intensity
            X = rand.uniform(low=low, high=high, size=(nums[i], self.dim))

            if not self.homogeneous:  # Thinning
                probs = [self.intensity(x) / max_intensity for x in X]
                probs = np.asarray(probs).ravel()

                # Indices of points to retain
                inds = rand.binomial(n=1, p=probs).astype(bool)
                X = X[inds, :]

            samples.append(X)

        # Check valid samples
        self.check_valid_samples(samples)

        return samples
