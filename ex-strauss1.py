"""
Apply KDSD and MMD tests to the 1-D Strauss process.
Samples are drawn via rejection sampling.
"""
from __future__ import division
from util import *
from kernels import *
from strauss_process import StraussProcess
from ksd import KSD
from mmd import MMD


if __name__ == "__main__":
    n, kernel_type, l, beta0, gamma0, r0, beta, gamma, r, seed, res_dir = sys.argv[1:]

    n = int(n)  # Sample size
    l = float(l)  # Length of square domain
    # Null model parameters
    beta0 = float(beta0)
    gamma0 = float(gamma0)
    r0 = float(r0)
    # Alternative model parameters
    beta = float(beta)
    gamma = float(gamma)
    r = float(r)
    seed = int(seed)  # Random seed

    dim = 1  # Default
    bounds = [(0, l)]*dim  # Domain

    print ("n = %d\nkernel_type = %s\nbounds = %r\n" +
           "beta = %s\ngamma = %s\nr = %s\nseed = %s\nres_dir=%s\n") % \
        (n, kernel_type, bounds, beta, gamma, r, seed, res_dir)

    rand.seed(seed)

    print "Null model params:"
    print "beta0 = %.3f\ngamma0 = %.3f\nr0 = %.3f\n" % (beta0, gamma0, r0)

    print "Model q params:"
    print "beta = %.3f\ngamma = %.3f\nr = %.3f\n" % (beta, gamma, r)

    # --------------------------- Draw samples --------------------------- #

    # Null model
    model_p = StraussProcess(dim=dim, bounds=bounds,
                             beta=beta0, gamma=gamma0, r=r0)
    samples_p = model_p.sample(num_samples=n)

    # Set q to perturbed dist or true p
    true_dist = rand.binomial(n=1, p=.5)  # 0 for p, 1 for q
    print "Ground truth: %s" % ("q != p" if true_dist else "q == p")

    # Parameters for alternative model
    (beta_q, gamma_q, r_q) = (beta, gamma, r) if true_dist else \
        (beta0, gamma0, r0)

    # Alternative model (draw samples from)
    model_q = StraussProcess(dim=dim, bounds=bounds,
                             beta=beta_q, gamma=gamma_q, r=r_q)
    samples_q = model_q.sample(num_samples=n)

    # --------------- Set kernel function for KSD and MMD --------------- #

    # Make sure KSD and MMD are using the same kernel_fun and bandwdith

    if kernel_type == 'emd':
        kernel_fun = emd_kernel
        int_method = 'fixed_quad'
        ksd_method = 'indirect'

    elif kernel_type == 'euclid':
        kernel_fun = euclidean_kernel
        int_method = 'fixed_quad'
        ksd_method = 'indirect'

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

    # Use null model
    ksd = KSD(dim=model_p.dim, bounds=model_p.bounds,
              papangelou_fun=model_p.papangelou,
              kernel_type=kernel_type,
              int_method=int_method,
              rbf_h=rbf_h,
              mp_npts=400,
              mc_npts=10**5,
              disp=False)

    kappa = ksd.compute_kappa(samples_q, method=ksd_method)
    ksd_stat = ksd.test_statistic(kappa)
    ksd_thres, ksd_boot = ksd.bootstrap(kappa)
    ksd_pval = ksd.p_value(ksd_stat, ksd_boot)
    ksd_pred = 1 * (ksd_stat > ksd_thres)  # 0 for p, 1 for q

    # ------------------------- Perform MMD test ------------------------- #

    mmd = MMD(kernel_fun=kernel_fun)
    mmd_stat, mmd_thres, mmd_pval, _ = mmd.perform_test(samples_p, samples_q)
    mmd_pred = 1 * (mmd_stat > mmd_thres)  # 0 for p, 1 for q

    # ------------------------- Save results ------------------------- #

    res = {'dim': dim, 'n': n, 'kernel_type': kernel_type,
           'beta0': beta0, 'gamma0': gamma0, 'r0': r0,
           'beta': beta, 'gamma': gamma, 'r': r, 'rbf_h': rbf_h,
           'true_dist': true_dist, 'int_method': int_method,
           'ksd_stat': ksd_stat, 'ksd_thres': ksd_thres,
           'ksd_pval': ksd_pval, 'ksd_pred': ksd_pred,
           'mmd_stat': mmd_stat, 'mmd_thres': mmd_thres,
           'mmd_pval': mmd_pval, 'mmd_pred': mmd_pred}

    pckl_write(res, res_dir + "strauss-n%d-%s-l%.1f-beta%.3f-gamma%.3f-r%.3f-seed%d.res" %
               (n, kernel_type, l, beta, gamma, r, seed))

    print 'Finished!'
