"""
Apply KDSD and MMD tests to the Hawkes process.
Samples are drawn via Ogata's thinning method.
"""
from __future__ import division
from util import *
from kernels import *
from hawkes_process import HawkesProcess
from ksd import KSD
from mmd import MMD


if __name__ == "__main__":
    dim, n, l, kernel_type, gamma0, beta0, tau0, gamma, beta, tau, seed, res_dir = sys.argv[1:]

    dim = int(dim)  # Dimension
    n = int(n)  # Sample size
    l = float(l)  # Domain length
    # Null model parameters
    gamma0 = float(gamma0)
    beta0 = float(beta0)
    tau0 = float(tau0)
    # Alternative model parameters
    gamma = float(gamma)
    beta = float(beta)
    tau = float(tau)
    seed = int(seed)  # Random seed

    bounds = [(0, l)]*dim  # Domain

    print ("dim = %d\nn = %d\nl = %s\nkernel_type = %s\nbounds = %r\n" +
           "gamma0 = %s\nbeta0 = %s\ntau0 = %s\n" +
           "gamma = %s\nbeta = %s\ntau = %s\nseed = %s\nres_dir=%s\n") % \
        (dim, n, l, kernel_type, bounds, gamma0, beta0, tau0, gamma, beta, tau,
         seed, res_dir)

    # Stationarity condition
    assert dim == 1, dim
    assert beta0 * tau0 < 1
    assert beta * tau < 1

    rand.seed(seed)

    # --------------------------- Draw samples --------------------------- #

    # Null model
    model_p = HawkesProcess(dim=dim, bounds=bounds,
                            gamma=gamma0, beta=beta0, tau=tau0)
    samples_p = model_p.sample(num_samples=n)

    # Set q to perturbed dist or true p
    true_dist = rand.binomial(n=1, p=.5)  # 0 for p, 1 for q
    print "Ground truth: %s" % ("q != p" if true_dist else "q == p")

    # Parameters for alternative model
    gamma_q, beta_q, tau_q = (gamma, beta, tau) if true_dist else \
        (gamma0, beta0, tau0)

    # Alternative model (draw samples from)
    model_q = HawkesProcess(dim=dim, bounds=bounds,
                            gamma=gamma_q, beta=beta_q, tau=tau_q)
    samples_q = model_q.sample(num_samples=n)

    # --------------- Set kernel function for KSD and MMD --------------- #

    rbf_h = None

    # TODO! Make sure KSD and MMD are using the same kernel_fun and bandwdith
    if kernel_type == 'emd':
        kernel_fun = emd_kernel
        # int_method = 'fixed_quad'
        int_method = 'trapz'
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

    ksd = KSD(dim=model_p.dim, bounds=model_p.bounds,
              papangelou_fun=model_p.papangelou,
              kernel_type=kernel_type,
              int_method=int_method,
              rbf_h=rbf_h,
              mp_npts=400,
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
           'gamma0': gamma0, 'beta0': beta0, 'tau0': tau0,
           'gamma': gamma, 'beta': beta, 'tau': tau, 'rbf_h': rbf_h,
           'true_dist': true_dist, 'int_method': int_method,
           'ksd_stat': ksd_stat, 'ksd_thres': ksd_thres,
           'ksd_pval': ksd_pval, 'ksd_pred': ksd_pred,
           'mmd_stat': mmd_stat, 'mmd_thres': mmd_thres,
           'mmd_pval': mmd_pval, 'mmd_pred': mmd_pred}

    pckl_write(res, res_dir + "hawkes-d%d-n%d-l%s-%s-gamma%.3f-beta%.3f-tau%.3f-seed%d.res" %
               (dim, n, l, kernel_type, gamma, beta, tau, seed))

    print 'Finished!'
