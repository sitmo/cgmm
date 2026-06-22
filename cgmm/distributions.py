# cgmm/distributions.py
# Layer 1 — single multivariate distributions + closed-form conditioning math.
#
# This module defines the structural `Distribution` protocol that the conditioner
# and mixture value objects program against, plus the first concrete member,
# `MultivariateNormal`. Heavy-tailed families (Student-t, GH) will implement the
# same protocol so that nothing downstream branches on family.
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import cholesky, solve_triangular
from scipy.special import gammaln, kve

Array = np.ndarray


def _log_kv(v: float, z) -> Array:
    """log K_v(z), the modified Bessel function of the second kind, computed
    stably via the exponentially-scaled ``kve(v, z) = K_v(z) e^z`` (so
    log K_v(z) = log kve(v, z) - z). Valid for z > 0."""
    z = np.asarray(z, dtype=float)
    return np.log(kve(v, z)) - z


def _as_generator(random_state):
    """Normalize a random_state into something with ``standard_normal``/``choice``.

    Accepts an int seed, ``None``, a ``numpy.random.Generator`` or a legacy
    ``numpy.random.RandomState`` (the latter two are returned unchanged so callers
    can thread a single stream through nested draws).
    """
    if isinstance(random_state, (np.random.Generator, np.random.RandomState)):
        return random_state
    return np.random.default_rng(random_state)


@runtime_checkable
class Distribution(Protocol):
    """Structural protocol for a single multivariate distribution.

    ``cov()`` is the actual covariance (for Student-t / GH this differs from the
    scale matrix and exists only under finite-moment conditions); families expose
    the scale matrix separately via a ``scale_`` attribute.
    """

    dim: int

    def logpdf(self, x: ArrayLike) -> Array: ...
    def rvs(self, size: int = 1, random_state=None) -> Array: ...
    def mean(self) -> Array: ...
    def cov(self) -> Array: ...
    def marginal(self, idx: Sequence[int]) -> "Distribution": ...
    def condition(
        self, cond_idx: Sequence[int], x_cond: ArrayLike
    ) -> "Distribution": ...


class MultivariateNormal:
    """A single multivariate normal N(mean, cov).

    For the Gaussian family the scale matrix equals the covariance, so
    ``scale_`` is simply an alias of ``cov()``. It is exposed separately because
    the Student-t / GH members will need ``cov()`` and ``scale_`` to differ while
    keeping an identical interface.

    Parameters
    ----------
    mean : array-like of shape (d,)
    cov : array-like of shape (d, d)
        Symmetric positive-definite covariance / scale matrix.
    """

    def __init__(self, mean: ArrayLike, cov: ArrayLike):
        self._mean = np.asarray(mean, dtype=float).ravel()
        self._cov = np.asarray(cov, dtype=float)
        d = self._mean.size
        if self._cov.shape != (d, d):
            raise ValueError(
                f"cov must have shape ({d}, {d}) to match mean, got {self._cov.shape}."
            )
        self.dim = int(d)
        self.scale_ = self._cov  # Gaussian: scale == covariance

    # ----- moments -----
    def mean(self) -> Array:
        return self._mean.copy()

    def cov(self) -> Array:
        return self._cov.copy()

    # ----- density -----
    def logpdf(self, x: ArrayLike) -> Array:
        """log N(x | mean, cov). Accepts (d,) or (n, d); always returns (n,)."""
        X = np.atleast_2d(np.asarray(x, dtype=float))
        if X.shape[1] != self.dim:
            raise ValueError(f"x has {X.shape[1]} columns, expected {self.dim}.")
        L = cholesky(self._cov, lower=True, check_finite=False)
        diff = (X - self._mean).T  # (d, n)
        u = solve_triangular(L, diff, lower=True, check_finite=False)  # (d, n)
        quad = np.sum(u * u, axis=0)  # (n,)
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        return -0.5 * (self.dim * np.log(2.0 * np.pi) + logdet + quad)

    # ----- sampling -----
    def rvs(self, size: int = 1, random_state=None) -> Array:
        """Draw ``size`` samples; returns shape (size, d)."""
        rng = _as_generator(random_state)
        z = rng.standard_normal((int(size), self.dim))
        L = cholesky(self._cov, lower=True, check_finite=False)
        return self._mean + z @ L.T

    # ----- marginalization / conditioning (same family) -----
    def marginal(self, idx: Sequence[int]) -> "MultivariateNormal":
        idx = np.asarray(idx, dtype=int)
        return MultivariateNormal(self._mean[idx], self._cov[np.ix_(idx, idx)])

    def condition(
        self, cond_idx: Sequence[int], x_cond: ArrayLike
    ) -> "MultivariateNormal":
        """Condition on the block ``cond_idx`` taking values ``x_cond``.

        Returns the (Gaussian) distribution of the complementary block, using the
        standard affine mean and Schur-complement covariance:

            mu_{t|c} = mu_t + S_tc S_cc^{-1} (x_cond - mu_c)
            S_{t|c}  = S_tt - S_tc S_cc^{-1} S_ct
        """
        cond_idx = np.asarray(cond_idx, dtype=int)
        x_cond = np.asarray(x_cond, dtype=float).ravel()
        mask = np.ones(self.dim, dtype=bool)
        mask[cond_idx] = False
        tgt = np.nonzero(mask)[0]

        mu_c = self._mean[cond_idx]
        mu_t = self._mean[tgt]
        S_cc = self._cov[np.ix_(cond_idx, cond_idx)]
        S_tc = self._cov[np.ix_(tgt, cond_idx)]
        S_tt = self._cov[np.ix_(tgt, tgt)]

        # A = S_tc S_cc^{-1}  via solving S_cc A^T = S_tc^T (S_cc symmetric)
        A = np.linalg.solve(S_cc, S_tc.T).T  # (len(tgt), len(cond_idx))
        mu_cond = mu_t + A @ (x_cond - mu_c)
        S_cond = S_tt - A @ S_tc.T  # S_tc.T == S_ct
        # symmetrize against round-off
        S_cond = 0.5 * (S_cond + S_cond.T)
        return MultivariateNormal(mu_cond, S_cond)


class MultivariateStudentT:
    r"""A single multivariate Student-t distribution t_nu(loc, scale).

    Parameterization (matching ``scipy.stats.multivariate_t``)
    ----------------------------------------------------------
    Location ``loc`` (the mean when nu > 1), scale matrix ``scale`` (the
    dispersion matrix, *not* the covariance), and degrees of freedom ``df`` = nu.
    The density in dimension p is

        f(x) = Gamma((nu + p) / 2)
               / [ Gamma(nu / 2) (nu pi)^{p/2} |scale|^{1/2} ]
               * [1 + (1/nu) (x - loc)^T scale^{-1} (x - loc)]^{-(nu + p)/2}.

    Moments
    -------
    - mean  = loc                      (finite iff nu > 1)
    - cov   = nu / (nu - 2) * scale    (finite iff nu > 2)

    so ``cov()`` differs from ``scale_`` by the factor nu / (nu - 2). The Gaussian
    is the nu -> inf limit, where ``cov()`` -> ``scale_``.

    Closure under marginalization / conditioning
    ---------------------------------------------
    A sub-block is again multivariate-t with the *same* nu, the sub-location and
    the sub-scale. Conditioning on a block of size q (see :meth:`condition`)
    raises the degrees of freedom to nu + q and inflates the Schur-complement
    scale by (nu + d2) / (nu + q); this is what gives the Student-t conditional
    its heteroskedasticity.

    Parameters
    ----------
    loc : array-like of shape (d,)
    scale : array-like of shape (d, d)
        Symmetric positive-definite scale (dispersion) matrix.
    df : float
        Degrees of freedom nu > 0.
    """

    def __init__(self, loc: ArrayLike, scale: ArrayLike, df: float):
        self.loc_ = np.asarray(loc, dtype=float).ravel()
        self.scale_ = np.asarray(scale, dtype=float)
        self.df = float(df)
        d = self.loc_.size
        if self.scale_.shape != (d, d):
            raise ValueError(
                f"scale must have shape ({d}, {d}) to match loc, got {self.scale_.shape}."
            )
        if self.df <= 0:
            raise ValueError(f"df (degrees of freedom) must be > 0, got {self.df}.")
        self.dim = int(d)

    # ----- moments -----
    def mean(self) -> Array:
        """Mean E[X] = loc. Finite only for df > 1."""
        if self.df <= 1.0:
            raise ValueError(f"mean is undefined for df <= 1 (df={self.df}).")
        return self.loc_.copy()

    def cov(self) -> Array:
        """Covariance Cov[X] = df / (df - 2) * scale. Finite only for df > 2."""
        if self.df <= 2.0:
            raise ValueError(f"cov is undefined for df <= 2 (df={self.df}).")
        return (self.df / (self.df - 2.0)) * self.scale_

    # ----- density -----
    def logpdf(self, x: ArrayLike) -> Array:
        """log t_nu(x | loc, scale). Accepts (d,) or (n, d); returns (n,)."""
        X = np.atleast_2d(np.asarray(x, dtype=float))
        if X.shape[1] != self.dim:
            raise ValueError(f"x has {X.shape[1]} columns, expected {self.dim}.")
        p, nu = self.dim, self.df
        L = cholesky(self.scale_, lower=True, check_finite=False)
        diff = (X - self.loc_).T  # (d, n)
        u = solve_triangular(L, diff, lower=True, check_finite=False)
        maha = np.sum(u * u, axis=0)  # (x-loc)^T scale^{-1} (x-loc)
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        log_norm = (
            gammaln(0.5 * (nu + p))
            - gammaln(0.5 * nu)
            - 0.5 * (p * np.log(nu * np.pi) + logdet)
        )
        return log_norm - 0.5 * (nu + p) * np.log1p(maha / nu)

    # ----- sampling -----
    def rvs(self, size: int = 1, random_state=None) -> Array:
        """Draw ``size`` samples; returns shape (size, d).

        Uses the normal mean-variance mixture form: X = loc + sqrt(nu / g) * Z,
        with Z ~ N(0, scale) and g ~ chi^2_nu.
        """
        rng = _as_generator(random_state)
        size = int(size)
        L = cholesky(self.scale_, lower=True, check_finite=False)
        z = rng.standard_normal((size, self.dim)) @ L.T  # ~ N(0, scale)
        g = rng.chisquare(self.df, size=size)
        return self.loc_ + z * np.sqrt(self.df / g)[:, None]

    # ----- marginalization / conditioning (same family) -----
    def marginal(self, idx: Sequence[int]) -> "MultivariateStudentT":
        """Marginal over ``idx``: same df, sub-location, sub-scale."""
        idx = np.asarray(idx, dtype=int)
        return MultivariateStudentT(
            self.loc_[idx], self.scale_[np.ix_(idx, idx)], self.df
        )

    def condition(
        self, cond_idx: Sequence[int], x_cond: ArrayLike
    ) -> "MultivariateStudentT":
        r"""Condition on the block ``cond_idx`` taking values ``x_cond``.

        With the block sizes q = len(cond_idx) and the partition (target | cond):

            mu_{t|c} = loc_t + S_tc S_cc^{-1} (x_cond - loc_c)
            S_{t.c}  = S_tt - S_tc S_cc^{-1} S_ct            (Schur complement)
            d2       = (x_cond - loc_c)^T S_cc^{-1} (x_cond - loc_c)

        the conditional is Student-t with

            df'    = nu + q
            scale' = (nu + d2) / (nu + q) * S_{t.c}
            loc'   = mu_{t|c}.
        """
        cond_idx = np.asarray(cond_idx, dtype=int)
        x_cond = np.asarray(x_cond, dtype=float).ravel()
        q = cond_idx.size
        nu = self.df

        mask = np.ones(self.dim, dtype=bool)
        mask[cond_idx] = False
        tgt = np.nonzero(mask)[0]

        loc_c = self.loc_[cond_idx]
        loc_t = self.loc_[tgt]
        S_cc = self.scale_[np.ix_(cond_idx, cond_idx)]
        S_tc = self.scale_[np.ix_(tgt, cond_idx)]
        S_tt = self.scale_[np.ix_(tgt, tgt)]

        diff_c = x_cond - loc_c
        sol_c = np.linalg.solve(S_cc, diff_c)  # S_cc^{-1} (x_cond - loc_c)
        A = np.linalg.solve(S_cc, S_tc.T).T  # S_tc S_cc^{-1}

        loc_cond = loc_t + A @ diff_c
        d2 = float(diff_c @ sol_c)
        S_schur = S_tt - A @ S_tc.T
        S_schur = 0.5 * (S_schur + S_schur.T)
        scale_cond = ((nu + d2) / (nu + q)) * S_schur
        return MultivariateStudentT(loc_cond, scale_cond, nu + q)


class MultivariateGH:
    r"""A single multivariate Generalized Hyperbolic distribution.

    Normal mean-variance mixture (NMVM) parameterization:

        X = loc + W * gamma + sqrt(W) * A Z,   A A^T = scale,  Z ~ N(0, I),
        W ~ GIG(lambda, chi, psi)   (Generalized Inverse Gaussian).

    Parameters
    ----------
    lambda_ : float
        GIG index (often written lambda).
    chi, psi : float
        GIG parameters, chi >= 0, psi >= 0 (not both zero).
    loc : array-like of shape (d,)
        Location vector (mu).
    scale : array-like of shape (d, d)
        Symmetric positive-definite dispersion matrix (Sigma).
    gamma : array-like of shape (d,)
        Skewness vector. gamma = 0 gives a symmetric distribution.

    Notes
    -----
    The family subsumes the Gaussian (psi -> inf), Student-t, variance-gamma, NIG
    and hyperbolic. ``cov()`` differs from ``scale`` (it adds the variance from the
    mixing scalar W). The density uses the modified Bessel function K (via
    ``scipy.special.kve``).

    The symmetric Student-t boundary ``psi = 0, gamma = 0, lambda = -nu/2,
    chi = nu`` is a degenerate point of the GIG where the generic Bessel
    evaluations break down; it is handled by a dedicated branch that delegates to
    :class:`MultivariateStudentT`. The skew-t boundary (``psi = 0`` with
    ``gamma != 0``) is not yet supported.
    """

    def __init__(self, lambda_, chi, psi, loc, scale, gamma):
        self.lambda_ = float(lambda_)
        self.chi = float(chi)
        self.psi = float(psi)
        self.loc_ = np.asarray(loc, dtype=float).ravel()
        self.scale_ = np.asarray(scale, dtype=float)
        self.gamma_ = np.asarray(gamma, dtype=float).ravel()
        d = self.loc_.size
        if self.scale_.shape != (d, d):
            raise ValueError(
                f"scale must have shape ({d}, {d}) to match loc, got {self.scale_.shape}."
            )
        if self.gamma_.size != d:
            raise ValueError(f"gamma must have length {d}, got {self.gamma_.size}.")
        if self.chi < 0 or self.psi < 0 or (self.chi == 0 and self.psi == 0):
            raise ValueError("Require chi >= 0, psi >= 0 and not both zero.")
        self.dim = int(d)

        # Symmetric Student-t boundary (psi == 0): GIG collapses to inverse-gamma.
        self._is_symmetric_t = self.psi == 0.0
        self._t = None
        if self._is_symmetric_t:
            if not np.allclose(self.gamma_, 0.0):
                raise NotImplementedError(
                    "The skew-t boundary (psi=0 with gamma!=0) is not supported; "
                    "use a small psi>0 instead."
                )
            if self.lambda_ >= 0 or self.chi <= 0:
                raise ValueError(
                    "Symmetric-t boundary requires lambda<0 and chi>0 "
                    "(lambda=-nu/2, chi=nu)."
                )
            df = -2.0 * self.lambda_
            self._t = MultivariateStudentT(self.loc_, (self.chi / df) * self.scale_, df)

    # ----- GIG mixing-scalar moments (for mean/cov) -----
    def _gig_moments(self):
        """Return (E[W], Var[W]) for W ~ GIG(lambda, chi, psi), psi > 0."""
        eta = np.sqrt(self.chi * self.psi)
        s = np.sqrt(self.chi / self.psi)
        k0 = kve(self.lambda_, eta)
        k1 = kve(self.lambda_ + 1.0, eta)
        k2 = kve(self.lambda_ + 2.0, eta)
        e_w = s * k1 / k0
        e_w2 = s * s * k2 / k0
        return e_w, e_w2 - e_w * e_w

    # ----- moments -----
    def mean(self) -> Array:
        if self._is_symmetric_t:
            return self._t.mean()
        e_w, _ = self._gig_moments()
        return self.loc_ + e_w * self.gamma_

    def cov(self) -> Array:
        if self._is_symmetric_t:
            return self._t.cov()
        e_w, var_w = self._gig_moments()
        return e_w * self.scale_ + var_w * np.outer(self.gamma_, self.gamma_)

    # ----- density -----
    def logpdf(self, x: ArrayLike) -> Array:
        X = np.atleast_2d(np.asarray(x, dtype=float))
        if X.shape[1] != self.dim:
            raise ValueError(f"x has {X.shape[1]} columns, expected {self.dim}.")
        if self._is_symmetric_t:
            return self._t.logpdf(X)

        d = self.dim
        lam = self.lambda_
        L = cholesky(self.scale_, lower=True, check_finite=False)
        diff = (X - self.loc_).T  # (d, n)
        sol = solve_triangular(L, diff, lower=True, check_finite=False)  # L^{-1} diff
        Q = np.sum(sol * sol, axis=0)  # (x-mu)' Sinv (x-mu), (n,)
        sol_g = solve_triangular(
            L, self.gamma_, lower=True, check_finite=False
        )  # L^{-1} gamma
        p = float(np.dot(sol_g, sol_g))  # gamma' Sinv gamma
        skew = sol.T @ sol_g  # (x-mu)' Sinv gamma, (n,)
        logdet = 2.0 * np.sum(np.log(np.diag(L)))

        eta = np.sqrt(self.chi * self.psi)
        order = lam - d / 2.0
        arg = np.sqrt((self.chi + Q) * (self.psi + p))

        log_c = (
            0.5 * lam * np.log(self.psi / self.chi)
            + (d / 2.0 - lam) * np.log(self.psi + p)
            - 0.5 * d * np.log(2.0 * np.pi)
            - 0.5 * logdet
            - _log_kv(lam, eta)
        )
        return log_c + _log_kv(order, arg) - (d / 2.0 - lam) * np.log(arg) + skew

    # ----- sampling -----
    def rvs(self, size: int = 1, random_state=None) -> Array:
        from scipy.stats import geninvgauss

        size = int(size)
        if self._is_symmetric_t:
            return self._t.rvs(size=size, random_state=random_state)
        rng = _as_generator(random_state)
        eta = np.sqrt(self.chi * self.psi)
        scale_w = np.sqrt(self.chi / self.psi)
        # W ~ GIG(lambda, chi, psi) = sqrt(chi/psi) * geninvgauss(p=lambda, b=eta)
        W = geninvgauss.rvs(self.lambda_, eta, size=size, random_state=rng) * scale_w
        L = cholesky(self.scale_, lower=True, check_finite=False)
        z = rng.standard_normal((size, self.dim)) @ L.T  # ~ N(0, scale)
        return self.loc_ + W[:, None] * self.gamma_ + np.sqrt(W)[:, None] * z

    # ----- marginalization / conditioning (same family) -----
    def marginal(self, idx: Sequence[int]) -> "MultivariateGH":
        idx = np.asarray(idx, dtype=int)
        return MultivariateGH(
            self.lambda_,
            self.chi,
            self.psi,
            self.loc_[idx],
            self.scale_[np.ix_(idx, idx)],
            self.gamma_[idx],
        )

    def condition(self, cond_idx: Sequence[int], x_cond: ArrayLike) -> "MultivariateGH":
        r"""Condition on the block ``cond_idx`` = ``x_cond``.

        With q = len(cond_idx), the Schur complement S_{1.2}, affine mean shift,
        Mahalanobis term d2 and the GIG-conjugacy updates:

            lambda* = lambda - q/2
            chi*    = chi + d2
            psi*    = psi + gamma_2' S_22^{-1} gamma_2
            loc*    = loc_1 + S_12 S_22^{-1} (x_cond - loc_2)
            scale*  = S_11 - S_12 S_22^{-1} S_21
            gamma*  = gamma_1 - S_12 S_22^{-1} gamma_2
        """
        cond_idx = np.asarray(cond_idx, dtype=int)
        x_cond = np.asarray(x_cond, dtype=float).ravel()
        q = cond_idx.size
        mask = np.ones(self.dim, dtype=bool)
        mask[cond_idx] = False
        tgt = np.nonzero(mask)[0]

        mu_c = self.loc_[cond_idx]
        mu_t = self.loc_[tgt]
        g_c = self.gamma_[cond_idx]
        g_t = self.gamma_[tgt]
        S_cc = self.scale_[np.ix_(cond_idx, cond_idx)]
        S_tc = self.scale_[np.ix_(tgt, cond_idx)]
        S_tt = self.scale_[np.ix_(tgt, tgt)]

        diff_c = x_cond - mu_c
        A = np.linalg.solve(S_cc, S_tc.T).T  # S_tc S_cc^{-1}
        loc_cond = mu_t + A @ diff_c
        S_schur = S_tt - A @ S_tc.T
        S_schur = 0.5 * (S_schur + S_schur.T)
        gamma_cond = g_t - A @ g_c

        d2 = float(diff_c @ np.linalg.solve(S_cc, diff_c))
        psi_star = self.psi + float(g_c @ np.linalg.solve(S_cc, g_c))
        return MultivariateGH(
            self.lambda_ - q / 2.0,
            self.chi + d2,
            psi_star,
            loc_cond,
            S_schur,
            gamma_cond,
        )
