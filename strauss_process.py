from __future__ import division
from util import *
from point_process import *
from poisson_process import PoissonProcess


class StraussProcess(PointProcess):
    """
    Strauss process.
    """
    def __init__(self, dim, bounds, beta, gamma, r):
        """
        Args:
            bounds: list of len dim, bounding box for each dimension.
            beta, gamma, r: the conditional intensity takes the form
                c(u, X) = beta * gamma ^ t(u, X)
                where t(u, X) is the number of points in X that lie within a
                distance r of the location u.
        """
        super(StraussProcess, self).__init__(dim, bounds)

        # Model parameters
        assert beta > 0 and 0 <= gamma <= 1 and r >= 0
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.r = float(r)

        return

    def papangelou(self, u, X):
        """
        Papangelou conditional intensity function.

        Args:
            u: array(dim), a new point location.
            X: array((..., dim)), existing points in a sample.
        """
        t = np.sum(norm(X-u, axis=1) <= self.r)  # Number of adjacent points
        c = self.beta * np.power(self.gamma, t)  # Conditional intensity
        return c

    def sample(self, num_samples, oversample_factor=1000):
        """
        Draw samples using the R spatstat library.

        Args:
            factor: number of proposals / num_samples.
        """
        if self.dim == 1:
            # Accept-reject from Poisson process
            samples = list()
            while len(samples) < num_samples:
                # Proposal samples from Poisson process
                pp = PoissonProcess(dim=self.dim, bounds=self.bounds,
                                    intensity=self.beta, homogeneous=True)
                # Number of proposal samples
                m = oversample_factor * (num_samples - len(samples))
                proposals = pp.sample(m)

                # Compute aacceptance probs
                probs = np.zeros(m)
                for i, X in enumerate(proposals):
                    s = np.sum(pdist(X) <= self.r)
                    # Number of distinct unordered pairs of points that are
                    # closer than r units apart
                    probs[i] = np.power(self.gamma, s)   # Acceptance prob
                # probs = [np.power(self.gamma, np.sum(pdist(X) <= self.r))
                #          for X in proposals]

                inds = rand.binomial(n=1, p=probs).astype(bool)  # Accepted indices
                new_samples = list(np.array(proposals)[inds])
                samples.extend(new_samples)

                print "Strauss: rejection sampling acceptance rate = %.3e" % \
                    np.mean(probs)  # (len(samples) / len(proposals))
                print "Currently sampled %d samples...\n" % len(samples)

            # if len(samples) < num_samples:
            #     raise ValueError("Need more proposal samples!\n")

            return samples[:num_samples]

        elif self.dim == 2:
            raise NotImplementedError

        return
