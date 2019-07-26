from __future__ import division
from util import *
from point_process import PointProcess


class HawkesProcess(PointProcess):
    """
    Hawkes process (one-dimensional).
    """
    def __init__(self, dim, bounds, gamma, beta, tau):
        """
        Initializes a one-dimensional Hawkes process.

        Args:
            dim: int, dimension of domain.
            bounds: list of length dim, bounding box for each dimension.
            gamma, beta, tau: floats, parameters in triggering function.
        """
        super(HawkesProcess, self).__init__(dim, bounds)

        if dim != 1 or bounds[0][0] != 0:
            raise NotImplementedError(
                "Only 1-D Hawkes processes, with time starting at 0!\n")

        assert beta*tau < 1, "Stationarity condition violated!\n"

        # Model parameters
        self.bounds = bounds
        self.T = self.bounds[0][1]  # End time
        self.gamma = gamma
        self.beta = beta
        self.tau = tau

        return

    def intensity(self, t, X):
        """
        Conditional intensity function given event history X up to time t.

        Args:
            t: array(dim), a new point location.
            X: array((..., dim)), existing points in a sample.
        """
        X = np.asarray(X)

        recip = np.exp(-(t - X[X < t]) / self.tau)  # Reciprocation

        return self.gamma + np.sum(self.beta * recip, axis=0)

    def trigger(self, dt):
        """
        Computes triggering function.

        Args:
            dt: float or array, delta time.
        """
        return self.beta * np.exp(-dt / self.tau)

    def papangelou(self, t, X):
        """
        Papangelou conditional intensity function at time t given points X.
        This is different from the conditional intensity (self.intensity).

        Args:
            t: array(dim), a new time point.
            X: array((..., dim)), existing time-points in a sample.
        """
        assert X.shape[1] == 1

        times = X.ravel()   # array
        denom = [self.intensity(u, X) for u in times[times > t]]  # Check
        g_t = self.trigger(times[times > t] - t)
        log_frac = np.sum(np.log(denom + g_t) - np.log(denom))

        # Integrated intensity term
        log_Lterm = -self.beta*self.tau * (1.-np.exp(-(self.T-t)/self.tau))

        # Intensity term
        log_lterm = np.log(self.intensity(t, X))

        return np.exp(log_Lterm + log_lterm + log_frac)

    def simulate(self):
        """
        Simulate a self-exciting Hawkes process using Ogata's thinning algorithm.

        Returns:
            X: array((..., dim)) of simulated event times.
        """
        if self.dim != 1 or self.bounds[0][0] != 0:
            raise NotImplementedError

        # Use list for in-place append, but return X as array
        X = list()  # Simulated event times

        num = 0  # Number of simulated events
        rate = self.gamma  # Maximum intensity

        # First event
        s = -np.log(rand.uniform()) / rate
        if s > self.T:
            return np.array(X).reshape((num, self.dim))
        X.append(s)
        num += 1
        rate = self.intensity(s, X) + self.beta  # Left-continuous

        # Subsequent events
        while s < self.T:
            s += -np.log(rand.uniform()) / rate
            if s >= self.T:
                break
            # Rejection test
            new_rate = self.intensity(s, X)
            if rand.uniform() <= new_rate / rate:
                X.append(s)
                num += 1
                rate = new_rate + self.beta  # Left-continuous
            else:
                rate = new_rate

        assert_len(X, num)

        return np.asarray(X).reshape((num, self.dim))

    def sample(self, num_samples):
        """
        Draw samples from a Hawkes process. Wrapper for simulate().

        Args:
            num_samples: int, number of point process realizations.
        """
        return [self.simulate() for _ in xrange(num_samples)]
