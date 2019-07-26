"""
Kernel functions for point configurations.

The functions below are not computing a kernel matrix; rather, each  computes
  a single evaluation of the kernel function for two point-sets.

Only the MMD-based kernel was used in the paper.
"""
from __future__ import division
from scipy.stats import wasserstein_distance
from util import *
import ot  # Optimal transport


def euclidean_kernel(X, Y, l=1.):
    """
    Kernel based on squared Euclidean distances.

    Args:
        X, Y: 2D-arrays, each containing a collection of points.
        l: float, length-scale.
    """
    if X.size == 0 and Y.size == 0:
        return 1.
    elif X.size == 0 or Y.size == 0:
        return 0.

    dists = cdist(X, Y, metric="sqeuclidean")
    return np.exp(-np.mean(dists) / l)


def emd_kernel(X, Y, l=1.):
    """
    Kernel based on the earth-mover's distance.
    This is not a positive-definite beyond one dimension!

    Args:
        X, Y: 2D-arrays, each containing a collection of points.
        l: float, length-scale.
    """
    if X.size == 0 and Y.size == 0:
        return 1.
    elif X.size == 0 or Y.size == 0:
        return 0.

    M = ot.dist(X, Y)  # Equivalent to cdist with metric sqeuclidean
    emd = ot.emd2([], [], M)  # Square of EMD; assumes equal weights

    return np.exp(-np.sqrt(emd) / l)


def mmd_kernel(X, Y, rbf_h=1., l=1., use_ustat=False):
    """
    MMD-based kernel ('M-kernel') defined in Eq.(15) of the paper.

    Uses a Gaussian RBF kernel on the ground space.
    """
    if X.size == 0 and Y.size == 0:
        return 1.
    elif X.size == 0 or Y.size == 0:
        return 0.

    # Compute ground kernel
    dists = squareform(pdist(np.vstack([X, Y]), metric="sqeuclidean"))

    K = np.exp(-dists / rbf_h)

    m, n = len(X), len(Y)  # Number of samples
    assert_shape(K, (m+n, m+n))
    kxx = K[:m, :m]
    kyy = K[m:, m:]
    kxy = K[:m, m:]
    assert is_symmetric(kxx)
    assert is_symmetric(kyy)

    if use_ustat:  # U-statistic estimate
        term_xx = (kxx.sum() - np.diag(kxx).sum()) / (m*(m-1))
        term_yy = (kyy.sum() - np.diag(kyy).sum()) / (n*(n-1))
        term_xy = kxy.sum() / (m*n)
    else:  # V-statistic estimate
        term_xx = kxx.sum() / (m**2)
        term_yy = kyy.sum() / (n**2)
        term_xy = kxy.sum() / (m*n)

    mmd = term_xx + term_yy - 2*term_xy

    return np.exp(-mmd / l)
