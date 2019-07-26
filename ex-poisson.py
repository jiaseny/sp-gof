"""
Apply KDSD and MMD tests to the Poisson process.
Samples are drawn via thinning.
"""
from __future__ import division
from util import *
from kernels import *
from poisson_process import PoissonProcess
from ksd import KSD
from mmd import MMD


if __name__ == "__main__":
    dim, n, kernel_type, gamma, eps, seed, res_dir = sys.argv[1:]

    dim = int(dim)  # Dimension
    n = int(n)  # Sample size
    gamma = float(gamma)  # Base rate
    eps = float(eps)  # Fluctuation magnitude
    seed = int(seed)  # Random seed

    assert_le(eps, gamma)  # Ensure intensity is non-negative

    bounds = [(0, 1)]*dim  # Domain
    tau = 2*np.pi  # Fluctuation frequency of intensity function

    print ("dim = %d\nn = %d\nkernel_type = %s\nbounds = %r\n" +
           "gamma = %s\neps = %s\nseed = %s\nres_dir=%s\n") % \
        (dim, n, kernel_type, bounds, gamma, eps, seed, res_dir)

    rand.seed(seed)

    # --------------------------- Draw samples --------------------------- #

    # Null model
    model_p = PoissonProcess(dim=dim, bounds=bounds,
                             intensity=gamma, homogeneous=True)
    samples_p = model_p.sample(num_samples=n)

    # Set q to perturbed dist or true p
    true_dist = rand.binomial(n=1, p=.5)  # 0 for p, 1 for q
    print "Ground truth: %s" % ("q != p" if true_dist else "q == p")

    if true_dist:
        def intensity_q(x):
            return gamma + eps * np.sin(tau * np.sum(x))
        max_intensity_q = gamma + eps
    else:
        intensity_q = gamma
        max_intensity_q = gamma

    # Alternative model (draw samples from)
    model_q = PoissonProcess(dim=dim, bounds=bounds, intensity=intensity_q,
                             max_intensity=max_intensity_q,
                             homogeneous=not true_dist)
    samples_q = model_q.sample(num_samples=n)

    # --------------- Set kernel function for KSD and MMD --------------- #

    rbf_h = None

    if kernel_type == 'emd':
        kernel_fun = emd_kernel
        int_method = 'fixed_quad'

    elif kernel_type == 'euclid':
        kernel_fun = euclidean_kernel
        int_method = 'fixed_quad'

    elif kernel_type == 'mmd':
        # Compute RBF bandwdith using median heuristic
        dists = pdist(np.concatenate(samples_q), metric="sqeuclidean")
        rbf_h = np.median(dists)  # dists[dists > 0]
        del dists  # Free memory

        def kernel_fun(X, Y):
            return mmd_kernel(X, Y, rbf_h=rbf_h)

        int_method = 'trapz'
        ksd_method = 'direct'

    else:
        raise ValueError("kernel_type %s not recognized!" % kernel_type)

    # ------------------------- Perform KSD test ------------------------- #

    print "Performing KSD test ..."

    ksd = KSD(dim=model_p.dim, bounds=model_p.bounds,
              papangelou_fun=model_p.papangelou,
              kernel_type=kernel_type,
              int_method=int_method,
              rbf_h=rbf_h,
              mp_npts=400 if dim == 1 else 20,
              mc_npts=10**4,
              disp=False)

    kappa = ksd.compute_kappa(samples_q, method=ksd_method)

    ksd_stat = ksd.test_statistic(kappa)
    ksd_thres, ksd_boot = ksd.bootstrap(kappa)
    ksd_pval = ksd.p_value(ksd_stat, ksd_boot)
    ksd_pred = 1 * (ksd_stat > ksd_thres)  # 0 for p, 1 for q

    # ------------------------- Perform MMD test ------------------------- #

    print "Performing MMD test ..."

    mmd = MMD(kernel_fun=kernel_fun)
    mmd_stat, mmd_thres, mmd_pval, _ = mmd.perform_test(samples_p, samples_q)
    mmd_pred = 1 * (mmd_stat > mmd_thres)  # 0 for p, 1 for q

    # ------------------------- Save results ------------------------- #

    res = {'dim': dim, 'n': n, 'kernel_type': kernel_type,
           'gamma': gamma, 'eps': eps, 'tau': tau, 'rbf_h': rbf_h,
           'true_dist': true_dist, 'int_method': int_method,
           'ksd_stat': ksd_stat, 'ksd_thres': ksd_thres,
           'ksd_pval': ksd_pval, 'ksd_pred': ksd_pred,
           'mmd_stat': mmd_stat, 'mmd_thres': mmd_thres,
           'mmd_pval': mmd_pval, 'mmd_pred': mmd_pred}

    pckl_write(res, res_dir + "poisson-d%d-n%d-%s-gamma%s-eps%s-seed%d.res" %
               (dim, n, kernel_type, gamma, eps, seed))

    print 'Finished!'
