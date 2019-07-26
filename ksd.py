from __future__ import division
from util import *


class KSD(object):
    """
    KSD goodness-of-fit test for point processes.
    """
    def __init__(self, dim, bounds, papangelou_fun, kernel_type, rbf_h=None,
                 int_method='quad', epsabs=1e-3, epsrel=1e-3,
                 tz_npts=101, mp_npts=100, mc_npts=10**4, disp=False):
        """
        Initialize a KSD goodness-of-fit test.

        Args:
            dim: int, dimension of domain (ground space).
            bounds: list of length dim, bounding box for each dimension.
            papangelou_fun: callable, Papangelou conditional intensity function.
            kernel_type: str, one of 'emd', 'euclid', or 'mmd' (see kernels.py)
                for more details.
            rbf_h: float, required if kernel_type == 'mmd': bandwidth for
                Gaussian RBF kernel used as the ground kernel in the M-kernel.
            int_method: float, numerical quadrature method.
            epsabs, epsrel: floats, absolute and relative error controls for
                scipy.integrate.nquad.
            tz_npts, mp_npts, mc_npts: ints, number of quadrature points for
                trapezoidal rule, mid-point rule, or Monte Carlo integration.
            disp: boolean, display intermediate outputs for debugging.
        """
        self.dim = dim
        self.bounds = bounds

        assert callable(papangelou_fun)
        self.papangelou = papangelou_fun

        # Set kernel function
        self.kernel_type = kernel_type
        if self.kernel_type == 'emd':
            self.kernel = emd_kernel

        elif self.kernel_type == 'euclid':
            self.kernel = euclidean_kernel

        elif self.kernel_type == 'mmd':
            self.kernel = mmd_kernel
            self.rbf_h = rbf_h
            assert self.rbf_h is not None

        else:
            raise ValueError("kernel_type %s not recognized!" % kernel_type)

        # Control numerical integration error
        self.int_method = int_method
        self.epsabs = epsabs
        self.epsrel = epsrel
        self.disp = disp

        self.tz_npts = int(tz_npts)  # Trapezoidal rule
        self.mp_npts = int(mp_npts)  # Mid-point rule
        self.mc_npts = int(mc_npts)  # Monte Carlo

        return

    def stein(self, func, X):
        """
        Evaluate the Stein operator.
        """
        assert callable(func)
        assert isinstance(X, np.ndarray)

        fX = func(X)

        term1 = self.integrate(
            lambda x: (func(np.vstack((X, x))) - fX) * self.papangelou(x, X))

        term2 = sum(func(np.delete(X, i, axis=0)) - fX
                    for i in xrange(X.shape[0]))

        return term1 + term2

    def integrate(self, func, rtol=1e-3):
        """
        Perform numerical integration of func over self.domain.

        Args:
            func: callable, function taking
            rtol: float, relative tolerance.
        """
        assert callable(func)
        mc_npts = self.mc_npts
        tz_npts = self.tz_npts

        if self.int_method == "quad":
            # Generic numerical quadrature; may be quite slow.
            if self.dim == 1:
                def f(x0): return func(np.asarray([x0]))
            elif self.dim == 2:
                def f(x0, x1):
                    return func([x0, x1])
            else:
                raise NotImplementedError("Only supports dim = 1 or 2!")

            res, err = nquad(f, ranges=self.bounds,
                             opts={'epsabs': self.epsabs,
                                   'epsrel': self.epsrel})  # 'limit'

        elif self.int_method == "fixed_quad":
            # Fixed-order Gaussian quadrature
            # Fast, but works well only for smooth integrands.
            if self.dim == 1:
                res, err = fixed_quad(lambda x: map(func, x),
                                      self.bounds[0][0], self.bounds[0][1])

            elif self.dim == 2:
                (lo0, hi0), (lo1, hi1) = self.bounds  # Lower-/upper- bounds

                res, err = fixed_quad(
                    lambda x1: map(
                        lambda x10: fixed_quad(
                            lambda x0: map(
                                lambda x00: func(np.atleast_2d([x00, x10])),
                                x0),
                            lo0, hi0)[0],
                        x1),
                    lo1, hi1)
            else:
                raise NotImplementedError("Only supports dim = 1 or 2!")

        elif self.int_method == "monte_carlo":
            # Monte Carlo integration
            lows, highs = zip(*self.bounds)
            U = rand.uniform(low=lows, high=highs, size=(mc_npts, self.dim))
            y = map(func, U)
            res = np.mean(y)
            err = np.std(y) / np.sqrt(mc_npts)

        elif self.int_method == "trapz":
            # Trapezoidal rule
            if self.dim == 1:
                res, err = int_trapz(func, self.bounds[0], tz_npts), None

            elif self.dim == 2:
                res, err = int_trapz(
                    lambda y: int_trapz(
                        lambda x: func(x, y), self.bounds[0], tz_npts),
                    self.bounds[1], tz_npts), None
            else:
                raise NotImplementedError("Only supports dim = 1 or 2!")

        else:
            raise ValueError("Integration method not recognized!")

        if self.disp and err is not None and err > 1e-6:  # Numerical error
            print "Warning: numerical error in KSD.integrate() is %.2e!" % err

        return res

    def integrate2(self, func, mc_npts=1e4, tz_npts=101):
        """
        Calls scipy.integrate function to perform numerical integration
            of a bivariate function over self.domain.
        """
        assert callable(func)
        mc_npts = int(mc_npts)

        # Rewrite func which takes two dim-dimensional arrays as input to have
        #   separate arguments
        if self.dim == 1:
            def f(x0, y0): return func([x0], [y0])
        elif self.dim == 2:
            def f(x0, x1, y0, y1): return func([x0, x1], [y0, y1])
        else:
            raise NotImplementedError("Only supports dim = 1 or 2!")

        if self.int_method == "quad":
            # Generic numerical quadrature; may be quite slow.
            res, err = nquad(f, ranges=self.bounds + self.bounds,
                             opts={'epsabs': self.epsabs,
                                   'epsrel': self.epsrel})

        elif self.int_method == "fixed_quad":  # Gaussian quadrature
            raise NotImplementedError

        elif self.int_method == 'trapz':
            # Trapezoidal rule
            assert self.dim == 1

            res, err = int_trapz(
                    lambda y: int_trapz(
                        lambda x: func(x, y), self.bounds[0], tz_npts),
                    self.bounds[0], tz_npts), None

        elif self.int_method == "monte_carlo":
            # Monte Carlo integration
            lows, highs = zip(*self.bounds)
            U = rand.uniform(low=lows, high=highs, size=(mc_npts, self.dim))
            V = rand.uniform(low=lows, high=highs, size=(mc_npts, self.dim))
            y = map(func, U, V)
            res = np.mean(y)
            err = np.std(y) / np.sqrt(mc_npts)

        else:
            raise ValueError("Integration method not recognized!")

        if self.disp and err > 1e-6:  # Numerical error
            print "Warning: numerical error in KSD.integrate2() is %.2e!" % err

        return res

    def kappa(self, X, Y, method="direct"):
        """
        Evaluates the kappa kernel function for inputs X and Y.

        If using the MMD-based kernel ('M-kernel'), directly call
           self.kappa_mmd() for efficient computations via caching.

        Args:
            X: array((m, dim)), a collection of points.
            Y: array((n, dim)), a collection of points.
        """
        if method == "indirect":
            # Indirect method
            return self.stein(lambda X_: self.stein(
                              lambda Y_: self.kernel(X_, Y_), Y), X)

        elif self.kernel_type == 'mmd':  # Direct
            return self.kappa_mmd(X, Y)

        # Direct method
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        n = X.shape[0]
        m = Y.shape[0]

        k = self.kernel(X, Y)
        # \sum_{x\in X} K(X\x, Y)
        k_X = sum(self.kernel(np.delete(X, i, axis=0), Y) for i in xrange(n))
        # \sum_{y\in Y} K(X, Y\y)
        k_Y = sum(self.kernel(X, np.delete(Y, j, axis=0)) for j in xrange(m))
        # \sum_{x,y\in Y} K(X\x, Y\y)
        k_X_Y = sum(self.kernel(np.delete(X, i, axis=0),
                                np.delete(Y, j, axis=0))
                    for i in xrange(n) for j in xrange(m))

        def integrand_uv(u, v):
            # Double integral over u and v
            # K(X + u, Y + v)
            k_uv = self.kernel(np.vstack((X, u)), np.vstack((Y, v)))
            # K(X, Y + v)
            k_v = self.kernel(X, np.vstack((Y, v)))
            # K(X + u, Y)
            k_u = self.kernel(np.vstack((X, u)), Y)
            c_u = self.papangelou(u, X)
            c_v = self.papangelou(v, Y)

            return (k_uv - k_v - k_u + k) * c_u * c_v

        def integrand_v(v):
            # \sum_{x\in X} k(X\x, Y + v)
            k_X_v = sum(self.kernel(np.delete(X, i, axis=0), np.vstack((Y, v)))
                        for i in xrange(n))
            # K(X, Y + v)
            k_v = self.kernel(X, np.vstack((Y, v)))
            c_v = self.papangelou(v, Y)

            return ((k_X_v - k_X) - n*(k_v - k)) * c_v

        def integrand_u(u):
            # \sum_{y\in Y} k(X + u, Y\v)
            k_Y_u = sum(self.kernel(np.vstack((X, u)), np.delete(Y, j, axis=0))
                        for j in xrange(m))
            # K(X + u, Y)
            k_u = self.kernel(np.vstack((X, u)), Y)
            c_u = self.papangelou(u, X)

            return ((k_Y_u - k_Y) - m*(k_u - k)) * c_u

        term1 = self.integrate2(integrand_uv)
        term2 = self.integrate(integrand_v)
        term3 = self.integrate(integrand_u)
        term4 = k_X_Y - n*k_Y - m*k_X + m*n*k

        return term1 + term2 + term3 + term4

    def kappa_mmd(self, X, Y, debug=False):
        """
        Efficient computation of the KSD kernel matrix when MMD-kernel is used.
        """
        def rbf_kernel(X, Y=None):
            """
            Compute RBF-kernel matrix for point-sets

            Returns:
                array(X.shape[0], Y.shape[0])
            """
            X = np.atleast_2d(X)
            if Y is None:
                dists = squareform(pdist(X, metric="sqeuclidean"))
            else:
                Y = np.atleast_2d(Y)
                dists = cdist(X, Y, metric="sqeuclidean")

            return np.exp(-dists / self.rbf_h)

        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        n = X.shape[0]
        m = Y.shape[0]

        K = rbf_kernel(np.vstack([X, Y]))
        Kxx = K[:n, :n]  # array(n, n)
        Kyy = K[n:, n:]  # array(m, m)
        Kxy = K[:n, n:]  # array(n, m)
        sxx = Kxx.sum()  # float, sum of all entries in Kxx
        syy = Kyy.sum()  # float, sum of all entries in Kyy
        sxy = Kxy.sum()  # float, sum of all entries in Kxy
        Kxx_sum = Kxx.sum(axis=0)
        Kyy_sum = Kyy.sum(axis=1)
        Kxy_xsum = Kxy.sum(axis=0)  # Sum over x
        Kxy_ysum = Kxy.sum(axis=1)  # Sum over y

        if debug:  # Checks
            assert is_psd(K)
            assert_shape(Kxx, (n, n))
            assert is_psd(Kxx)
            assert_shape(Kyy, (m, m))
            assert is_psd(Kyy)
            assert_shape(Kxy, (n, m))

        if self.int_method == "trapz":
            mp_npts = self.mp_npts
            npts = mp_npts**self.dim  # Actual total number of quadrature pts

            if self.dim == 1:
                lo, hi = self.bounds[0]
                # # Trapezoidal rule
                # U = np.linspace(lo, hi, tz_npts).reshape(tz_npts, 1)
                # V = np.linspace(lo, hi, tz_npts).reshape(tz_npts, 1)
                # Mid-point rule
                h = (hi - lo) / mp_npts
                U = V = np.linspace(lo+h/2, hi-h/2, mp_npts).reshape(mp_npts, 1)

            elif self.dim == 2:
                # Mid-point rule
                (lo0, hi0), (lo1, hi1) = self.bounds
                h0, h1 = (hi0 - lo0) / mp_npts, (hi1 - lo1) / mp_npts  # Width
                uu = np.linspace(lo0 + h0/2, hi0 - h0/2, mp_npts)
                vv = np.linspace(lo1 + h1/2, hi1 - h1/2, mp_npts)
                U = V = np.array(np.meshgrid(uu, vv)).T.reshape(-1, 2)
            else:
                raise NotImplementedError("Only supports dim = 1 or 2!")

            assert_shape(U, (npts, self.dim))
            assert_shape(V, (npts, self.dim))

        elif self.int_method == "monte_carlo":
            npts = mc_npts = self.mc_npts

            low, high = zip(*self.bounds)
            U = rand.uniform(low=low, high=high, size=(mc_npts, self.dim))
            V = rand.uniform(low=low, high=high, size=(mc_npts, self.dim))

        else:
            raise ValueError("Integration method not recognized!")

        # Assumes U and V are equal
        Kux = Kvx = rbf_kernel(U, X)
        Kvy = Kuy = rbf_kernel(V, Y)
        Kuu = Kvv = Kuv = rbf_kernel(U, V)
        Kux_sum = Kvx_sum = Kux.sum(axis=1)
        Kvy_sum = Kuy_sum = Kvy.sum(axis=1)

        if debug:
            if self.dim == 1:
                assert is_psd(Kuu)
            else:  # Too expensive...
                assert is_symmetric(Kuu)

        def _mmd(X, Y):  # For testing
            """
            Computes the MMD between point-sets X and Y.
            """
            X = np.atleast_2d(X)
            Y = np.atleast_2d(Y)
            n = X.shape[0]
            m = Y.shape[0]

            K = rbf_kernel(np.vstack([X, Y]))
            Kxx = K[:n, :n]  # array(n, n)
            Kyy = K[n:, n:]  # array(m, m)
            Kxy = K[:n, n:]  # array(n, m)
            sxx = Kxx.sum()  # float, sum of all entries in Kxx
            syy = Kyy.sum()  # float, sum of all entries in Kyy
            sxy = Kxy.sum()  # float, sum of all entries in Kxy

            mmd = sxx/(n*n) + syy/(m*m) - 2*sxy/(n*m)  # K(X, Y)

            return mmd

        # Notation: i, j are indices into X, Y; u, v are indices into U, V
        # Cached all kernel computations and summations;
        #   each integrand evaluation should take O(1) time.

        # K(X, Y)
        if n > 0 and m > 0:
            k = np.exp(-(sxx/(n*n) + syy/(m*m) - 2*sxy/(n*m)))
        else:
            k = 1 * (n == 0 and m == 0)

        def ku(u):  # K(X + u, Y)
            if m == 0:
                return 0

            suu = Kuu[u, u]  # rbf_kernel(u, u).sum()  # float, sum of float
            sux = Kux_sum[u]  # rbf_kernel(u, X).sum()  # float, sum of array(n)
            suy = Kuy_sum[u]  # rbf_kernel(u, Y).sum()  # float, sum of array(m)
            txx = sxx + 2*sux + suu
            txy = sxy + suy

            mmd = txx/((n+1)*(n+1)) + syy/(m*m) - 2*txy/((n+1)*m)

            if debug:
                assert_close(mmd, _mmd(np.vstack((X, U[u])), Y))

            return np.exp(-mmd)

        def kv(v):  # K(X, Y + v)
            if n == 0:
                return 0

            svv = Kvv[v, v]  # rbf_kernel(v, v).sum()  # float, sum of float
            svx = Kvx_sum[v]  # rbf_kernel(v, X).sum()  # float, sum of array(n)
            svy = Kvy_sum[v]  # rbf_kernel(v, Y).sum()  # float, sum of array(m)
            tyy = syy + 2*svy + svv
            txy = sxy + svx

            mmd = sxx/(n*n) + tyy/((m+1)*(m+1)) - 2*txy/(n*(m+1))

            if debug:
                assert_close(mmd, _mmd(X, np.vstack((Y, V[v]))))

            return np.exp(-mmd)

        def kuv(u, v):  # K(X + u, Y + v)
            suu = Kuu[u, u]  # rbf_kernel(u, u).sum()  # float, sum of float
            sux = Kux_sum[u]  # rbf_kernel(u, X).sum()  # float, sum of array(n)
            suy = Kuy_sum[u]  # rbf_kernel(u, Y).sum()  # float, sum of array(m)
            svv = Kvv[v, v]  # rbf_kernel(v, v).sum()  # float, sum of float
            svx = Kvx_sum[v]  # rbf_kernel(v, X).sum()  # float, sum of array(n)
            svy = Kvy_sum[v]  # rbf_kernel(v, Y).sum()  # float, sum of array(m)
            suv = Kuv[u, v]  # rbf_kernel(u, v).sum()  # float, sum of float
            txx = sxx + 2*sux + suu
            tyy = syy + 2*svy + svv
            txy = sxy + svx + suy + suv

            mmd = txx/((n+1)*(n+1)) + tyy/((m+1)*(m+1)) - 2*txy/((m+1)*(n+1))

            if debug:
                assert_close(mmd, _mmd(np.vstack((X, U[u])),
                             np.vstack((Y, V[v]))))

            return np.exp(-mmd)

        def k_x(i):  # K(X \ X[i], Y)
            if n == 1 or m == 0:
                return 1 * (n == 1 and m == 0)

            sii = Kxx[i, i]
            six = Kxx_sum[i]  # Kxx[i, :].sum()
            siy = Kxy_ysum[i]  # Kxy[i, :].sum()
            txx = sxx - 2*six + sii
            txy = sxy - siy

            mmd = txx/((n-1)*(n-1)) + syy/(m*m) - 2*txy/((n-1)*m)

            if debug:
                assert_close(mmd, _mmd(np.delete(X, i, axis=0), Y))

            return np.exp(-mmd)

        def k_y(j):  # K(X, Y \ Y[j])
            if n == 0 or m == 1:
                return 1 * (n == 0 and m == 1)

            sjj = Kyy[j, j]
            sjy = Kyy_sum[j]  # Kyy[:, j].sum()
            sxj = Kxy_xsum[j]  # Kxy[:, j].sum()
            tyy = syy - 2*sjy + sjj
            txy = sxy - sxj

            mmd = sxx/(n*n) + tyy/((m-1)*(m-1)) - 2*txy/(n*(m-1))

            if debug:
                assert_close(mmd, _mmd(X, np.delete(Y, j, axis=0)))

            return np.exp(-mmd)

        def k_x_y(i, j):  # K(X \ X[i], Y \ Y[j])
            if n == 1 or m == 1:
                return 1 * (n == 1 and m == 1)

            sii = Kxx[i, i]
            six = Kxx_sum[i]  # Kxx[i, :].sum()
            siy = Kxy_ysum[i]  # Kxy[i, :].sum()
            sjj = Kyy[j, j]
            sjy = Kyy_sum[j]  # Kyy[:, j].sum()
            sxj = Kxy_xsum[j]  # Kxy[:, j].sum()
            sij = Kxy[i, j]
            txx = sxx - 2*six + sii
            tyy = syy - 2*sjy + sjj
            txy = sxy - sxj - siy + sij

            mmd = txx/((n-1)*(n-1)) + tyy/((m-1)*(m-1)) - 2*txy/((n-1)*(m-1))

            if debug:
                assert_close(mmd, _mmd(np.delete(X, i, axis=0),
                                       np.delete(Y, j, axis=0)))

            return np.exp(-mmd)

        def k_xv(i, v):  # K(X \ X[i], Y + v)
            if n == 1:
                return 0

            sii = Kxx[i, i]
            six = Kxx_sum[i]  # Kxx[i, :].sum()
            siy = Kxy_ysum[i]  # Kxy[i, :].sum()
            txx = sxx - 2*six + sii

            svv = Kvv[v, v]  # rbf_kernel(v, v).sum()  # float, sum of float
            svx = Kvx_sum[v]  # rbf_kernel(v, X).sum()  # float, sum of array(n)
            svy = Kvy_sum[v]  # rbf_kernel(v, Y).sum()  # float, sum of array(m)
            tyy = syy + 2*svy + svv

            siv = Kvx[v, i]  # rbf_kernel(v, X[i]).sum()  # float, sum of float
            txy = sxy - siy + svx - siv

            mmd = txx/((n-1)*(n-1)) + tyy/((m+1)*(m+1)) - 2*txy/((n-1)*(m+1))

            if debug:
                assert_close(mmd, _mmd(np.delete(X, i, axis=0),
                                       np.vstack((Y, V[v]))))

            return np.exp(-mmd)

        def ku_y(u, j):  # K(X + u, Y \ Y[j])
            if m == 1:
                return 0

            suu = Kuu[u, u]  # rbf_kernel(u, u).sum()  # float, sum of float
            sux = Kux_sum[u]  # rbf_kernel(u, X).sum()  # float, sum of array(n)
            suy = Kuy_sum[u]  # rbf_kernel(u, Y).sum()  # float, sum of array(m)
            txx = sxx + 2*sux + suu

            sjj = Kyy[j, j]
            sjy = Kyy_sum[j]  # Kyy[:, j].sum()
            sxj = Kxy_xsum[j]  # Kxy[:, j].sum()
            tyy = syy - 2*sjy + sjj

            suj = Kuy[u, j]  # rbf_kernel(u, Y[j]).sum()  # float, sum of float
            txy = sxy - sxj + suy - suj

            mmd = txx/((n+1)*(n+1)) + tyy/((m-1)*(m-1)) - 2*txy/((n+1)*(m-1))

            if debug:
                assert_close(mmd, _mmd(np.vstack((X, U[u])),
                                       np.delete(Y, j, axis=0)))

            return np.exp(-mmd)

        # \sum_{x\in X} K(X\x, Y)
        k_X = sum(k_x(i) for i in xrange(n))
        # \sum_{y\in Y} K(X, Y\y)
        k_Y = sum(k_y(j) for j in xrange(m))
        # \sum_{x,y\in Y} K(X\x, Y\y)
        k_X_Y = sum(k_x_y(i, j) for i in xrange(n) for j in xrange(m))

        # Cache Papangelou computations
        cu = [self.papangelou(U[u], X) for u in xrange(npts)]
        cv = [self.papangelou(V[v], Y) for v in xrange(npts)]

        # Note: using u, v as indices into U, V

        def integrand_uv(u, v):  # Integrand over (u, v)
            return (kuv(u, v) - kv(v) - ku(u) + k) * cu[u] * cv[v]

        def integrand_v(v):  # Integrand over v
            k_Xv = sum(k_xv(i, v) for i in xrange(n))  # \sum_x k(X\x, Y + v)
            return ((k_Xv - k_X) - n*(kv(v) - k)) * cv[v]

        def integrand_u(u):  # Integrand over u
            ku_Y = sum(ku_y(u, j) for j in xrange(m))  # \sum_y k(X + u, Y\y)
            return ((ku_Y - k_Y) - m*(ku(u) - k)) * cu[u]

        if self.int_method == 'trapz':
            if self.dim == 1:
                term1 = h**2. * sum(integrand_uv(u, v) for u in xrange(npts)
                                    for v in xrange(npts))
                term2 = h * sum(integrand_v(v) for v in xrange(npts))
                term3 = h * sum(integrand_u(u) for u in xrange(npts))

            elif self.dim == 2:
                term1 = h0**2 * h1**2 * sum(integrand_uv(u, v)
                                            for u in xrange(npts)
                                            for v in xrange(npts))
                term2 = h0 * h1 * sum(integrand_v(v) for v in xrange(npts))
                term3 = h0 * h1 * sum(integrand_u(u) for u in xrange(npts))

        elif self.int_method == 'monte_carlo':
            term1 = np.mean([integrand_uv(_, _) for _ in xrange(npts)])
            term2 = np.mean([integrand_v(_) for _ in xrange(npts)])
            term3 = np.mean([integrand_u(_) for _ in xrange(npts)])

        else:
            raise NotImplementedError("Only supports dim = 1 or 2!")

        # term1 = self.integrate2(integrand_uv)
        # term2 = self.integrate(integrand_v)
        # term3 = self.integrate(integrand_u)
        term4 = k_X_Y - n*k_Y - m*k_X + m*n*k

        return term1 + term2 + term3 + term4

    def compute_kappa(self, samples, method='direct'):
        """
        Compute the KSD kernel matrix kappa_mu.
        """
        n = len(samples)  # Sample size
        kappa = np.zeros((n, n))

        if self.kernel_type == 'mmd':
            print "Compute kappa using rbf_h = %.3f." % self.rbf_h

        for i in xrange(n):
            print "Computing kappa[%d, :] ..." % i

            for j in xrange(i, n):
                # print "Computing kappa[%d, %d] ..." % (i, j)
                X, Y = samples[i], samples[j]
                kappa[i, j] = kappa[j, i] = self.kappa(X, Y, method=method)

        assert not np.any(np.isnan(kappa)), kappa
        # assert is_psd(kappa)
        if not is_psd(kappa):
            print "Error: KSD kappa matrix is not psd!\n"

        return kappa

    def test_statistic(self, kappa, use_ustat=True):
        """
        Compute the U- or V- test statistic.

        Args:
            kappa: array(n, n), pre-computed kappa kernel matrix.
            use_ustat: boolean, whether to use U- or V- statistics.

        Returns:
            test_stat: float, computed test statistic.
        """
        assert_eq(kappa.shape[0], kappa.shape[1])
        n = kappa.shape[0]  # Sample size

        if use_ustat:  # U-stat
            if n == 1:
                raise ValueError("Sample size is 1!")
            diagonal = np.diag(np.diag(kappa))  # (n, n)
            test_stat = np.sum(kappa - diagonal) / (n*(n-1))

        else:  # V-stat
            test_stat = np.sum(kappa) / (n**2)

        return test_stat

    def bootstrap(self, kappa, quantile=.95, n_boot=10000):
        """
        Generalized bootstrap for U-statistics.

        Args:
            kappa: array(n, n), pre-computed kappa kernel matrix.
            n_boot: int, number of bootstrap samples to use.

        Returns:
            boot_thres: float, critical value of the test.
            boot_samples: array of length n_boot, bootstrap statistics.
        """
        n = kappa.shape[0]  # Sample size
        kappa = kappa - np.diag(np.diag(kappa))  # Remove diagonal

        # Bootstrap samples for KSD estimates
        boot_samples = np.zeros(n_boot)

        for j in xrange(n_boot):
            wvec = (rand.multinomial(n=n, pvals=np.ones(n)/n) - 1.) / n
            boot_samples[j] = wvec.dot(kappa).dot(wvec)

        # Compute quantile of bootstrap sampling distribution
        boot_thres = np.percentile(boot_samples, 100.*quantile)

        return boot_thres, boot_samples

    def p_value(self, test_stat, boot_samples):
        """
        Computes the p-value of the goodness-of-fit test.

        Args:
            test_stat: float, value of the test statistic.
            boot_samples: array, bootstrap statistics.
        """
        assert isinstance(test_stat, float)
        assert isinstance(boot_samples, np.ndarray)

        return np.mean(boot_samples >= test_stat)
