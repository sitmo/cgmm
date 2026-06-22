# cgmm/student_t.py
# Maximum-likelihood EM fitter for a mixture of multivariate Student-t
# distributions, mirroring the scikit-learn GaussianMixture API.
#
# The Student-t is a scale mixture of normals: X | W ~ N(loc, W * scale) with the
# mixing scalar W ~ InverseGamma(nu/2, nu/2). EM augments the data with the latent
# component label and the latent scale W; the E-step computes, per point/component,
# the responsibility tau_ik together with u_ik = E[1/W | x_i, k] and E[log W].
# The M-step reuses the weighted-Gaussian location/scale updates and solves a 1-D
# digamma equation for the degrees of freedom (ECME). See docs/conventions.md.
from __future__ import annotations

from numbers import Real

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import brentq
from scipy.special import digamma, gammaln
from sklearn.mixture import GaussianMixture
from sklearn.mixture._base import BaseMixture
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted

from .distributions import MultivariateStudentT
from .container import MixtureDistribution

Array = np.ndarray

# Degrees of freedom are searched within this range; the upper bound stands in for
# nu -> inf (effectively Gaussian).
_DOF_MIN = 1e-1
_DOF_MAX = 1e6


def _solve_dof(const: float, lo: float = _DOF_MIN, hi: float = _DOF_MAX) -> float:
    """Solve  log(nu/2) - digamma(nu/2) + const = 0  for nu in [lo, hi].

    g(nu) = log(nu/2) - digamma(nu/2) is positive and strictly decreasing to 0 as
    nu -> inf, so a finite root exists iff const < 0. When there is no sign change
    (data effectively Gaussian) we clamp to the boundary nearest the root.
    """

    def f(nu: float) -> float:
        return float(np.log(nu / 2.0) - digamma(nu / 2.0) + const)

    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0.0:
        return hi if abs(f_hi) < abs(f_lo) else lo
    return float(brentq(f, lo, hi, xtol=1e-6, rtol=1e-8))


class StudentTMixture(BaseMixture):
    """Mixture of multivariate Student-t distributions fitted by EM.

    Mirrors :class:`sklearn.mixture.GaussianMixture` so it drops into the same
    workflows (and, later, the conditioner). Scale matrices are always full.

    Parameters
    ----------
    n_components : int, default=1
    dof : {"free", "shared"} or float, default="free"
        Degrees of freedom nu. ``"free"`` estimates one nu per component,
        ``"shared"`` a single nu across components, and a float fixes nu.
    tol, reg_covar, max_iter, n_init, random_state, warm_start, verbose,
    verbose_interval :
        Same meaning as in :class:`sklearn.mixture.GaussianMixture`.
    init_params : {"kmeans", "k-means++", "random", "random_from_data", "gmm"}
        How to initialize. ``"gmm"`` warm-starts from a fitted GaussianMixture.

    Attributes
    ----------
    weights_ : ndarray of shape (n_components,)
    locations_ : ndarray of shape (n_components, n_features)
    scales_ : ndarray of shape (n_components, n_features, n_features)
        Scale (dispersion) matrices -- not covariances.
    dofs_ : ndarray of shape (n_components,)
    converged_ : bool
    n_iter_ : int
    lower_bound_ : float
        Best mean log-likelihood reached.
    lower_bounds_ : list of float
        Per-iteration mean log-likelihood of the best init (non-decreasing).
    """

    _parameter_constraints: dict = {
        **BaseMixture._parameter_constraints,
        "init_params": [
            StrOptions({"kmeans", "k-means++", "random", "random_from_data", "gmm"})
        ],
        "dof": [
            StrOptions({"free", "shared"}),
            Interval(Real, 0.0, None, closed="neither"),
        ],
    }

    def __init__(
        self,
        n_components: int = 1,
        *,
        dof="free",
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: str = "kmeans",
        random_state=None,
        warm_start: bool = False,
        verbose: int = 0,
        verbose_interval: int = 10,
    ):
        super().__init__(
            n_components=n_components,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            random_state=random_state,
            warm_start=warm_start,
            verbose=verbose,
            verbose_interval=verbose_interval,
        )
        self.dof = dof

    # ------------------------------------------------------------------ init
    def _check_parameters(self, X):
        # Scalar parameter ranges are validated via _parameter_constraints.
        pass

    def _initialize_parameters(self, X, random_state):
        if self.init_params == "gmm":
            gmm = GaussianMixture(
                n_components=self.n_components,
                covariance_type="full",
                reg_covar=self.reg_covar,
                random_state=random_state,
            ).fit(X)
            self._initialize(X, gmm.predict_proba(X))
        else:
            super()._initialize_parameters(X, random_state)

    def _initialize(self, X, resp):
        n, p = X.shape
        K = self.n_components
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        locations = (resp.T @ X) / nk[:, None]
        scales = np.empty((K, p, p))
        for k in range(K):
            diff = X - locations[k]
            Sk = (resp[:, k][:, None] * diff).T @ diff / nk[k]
            Sk.flat[:: p + 1] += self.reg_covar
            scales[k] = Sk

        self.weights_ = nk / n
        self.locations_ = locations
        self.scales_ = scales
        if isinstance(self.dof, str):
            self.dofs_ = np.full(K, 20.0)  # start near-Gaussian (design: init nu large)
        else:
            self.dofs_ = np.full(K, float(self.dof))
        self._scale_chol_ = self._cholesky(self.scales_)

    @staticmethod
    def _cholesky(scales: Array) -> Array:
        return np.stack([cholesky(S, lower=True, check_finite=False) for S in scales])

    # -------------------------------------------------------------- E-step bits
    def _mahalanobis(self, X) -> Array:
        """Per-component squared Mahalanobis distance under the current scales."""
        n = X.shape[0]
        K = self.n_components
        delta = np.empty((n, K))
        for k in range(K):
            u = solve_triangular(
                self._scale_chol_[k],
                (X - self.locations_[k]).T,
                lower=True,
                check_finite=False,
            )
            delta[:, k] = np.sum(u * u, axis=0)
        return delta

    def _estimate_log_prob(self, X):
        n, p = X.shape
        nu = self.dofs_  # (K,)
        delta = self._mahalanobis(X)  # (n, K)
        logdet = np.array([2.0 * np.sum(np.log(np.diag(L))) for L in self._scale_chol_])
        log_norm = (
            gammaln(0.5 * (nu + p))
            - gammaln(0.5 * nu)
            - 0.5 * (p * np.log(nu * np.pi) + logdet)
        )  # (K,)
        return log_norm[None, :] - 0.5 * (nu[None, :] + p) * np.log1p(
            delta / nu[None, :]
        )

    def _estimate_log_weights(self):
        return np.log(self.weights_)

    def _compute_lower_bound(self, log_resp, log_prob_norm):
        return log_prob_norm

    # ------------------------------------------------------------------ M-step
    def _m_step(self, X, log_resp):
        n, p = X.shape
        K = self.n_components
        resp = np.exp(log_resp)  # tau_ik, (n, K)

        # E-step augmentation evaluated at the CURRENT parameters.
        delta = self._mahalanobis(X)
        nu = self.dofs_
        u = (nu[None, :] + p) / (nu[None, :] + delta)  # E[1/W], (n, K)
        e_log_u = digamma(0.5 * (nu[None, :] + p)) - np.log(0.5 * (nu[None, :] + delta))

        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        self.weights_ = nk / n

        # Weighted location (by tau*u) and scale (by tau*u, normalized by sum tau).
        tu = resp * u
        locations = (tu.T @ X) / tu.sum(axis=0)[:, None]
        scales = np.empty((K, p, p))
        for k in range(K):
            diff = X - locations[k]
            Sk = (tu[:, k][:, None] * diff).T @ diff / nk[k]
            Sk.flat[:: p + 1] += self.reg_covar
            scales[k] = Sk
        self.locations_ = locations
        self.scales_ = scales
        self._scale_chol_ = self._cholesky(scales)

        # Degrees-of-freedom CM-step (uses old-parameter u / e_log_u -> monotone EM).
        if not isinstance(self.dof, str):
            self.dofs_ = np.full(K, float(self.dof))
        elif self.dof == "shared":
            const = 1.0 + float((resp * (e_log_u - u)).sum()) / nk.sum()
            self.dofs_ = np.full(K, _solve_dof(const))
        else:  # "free"
            dofs = np.empty(K)
            for k in range(K):
                const = (
                    1.0 + float((resp[:, k] * (e_log_u[:, k] - u[:, k])).sum()) / nk[k]
                )
                dofs[k] = _solve_dof(const)
            self.dofs_ = dofs

    # ----------------------------------------------------------- param plumbing
    def _get_parameters(self):
        return (
            self.weights_,
            self.locations_,
            self.scales_,
            self.dofs_,
            self._scale_chol_,
        )

    def _set_parameters(self, params):
        (
            self.weights_,
            self.locations_,
            self.scales_,
            self.dofs_,
            self._scale_chol_,
        ) = params

    def _n_parameters(self):
        p = self.locations_.shape[1]
        K = self.n_components
        cov_params = K * p * (p + 1) // 2
        mean_params = K * p
        if isinstance(self.dof, str):
            dof_params = 1 if self.dof == "shared" else K
        else:
            dof_params = 0
        return int(cov_params + mean_params + dof_params + (K - 1))

    # --------------------------------------------------------- model selection
    def bic(self, X) -> float:
        """Bayesian Information Criterion (lower is better)."""
        return -2.0 * self.score(X) * X.shape[0] + self._n_parameters() * np.log(
            X.shape[0]
        )

    def aic(self, X) -> float:
        """Akaike Information Criterion (lower is better)."""
        return -2.0 * self.score(X) * X.shape[0] + 2.0 * self._n_parameters()

    # ----------------------------------------------------------------- sampling
    def sample(self, n_samples: int = 1):
        """Generate samples; returns (X, y) with component labels y."""
        check_is_fitted(self)
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1.")
        rng = check_random_state(self.random_state)
        n_per = rng.multinomial(n_samples, self.weights_)
        X_parts, y_parts = [], []
        for k, nk in enumerate(n_per):
            if nk == 0:
                continue
            dist = MultivariateStudentT(
                self.locations_[k], self.scales_[k], self.dofs_[k]
            )
            X_parts.append(dist.rvs(size=int(nk), random_state=rng))
            y_parts.append(np.full(int(nk), k))
        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)
        return X, y

    # ----------------------------------------------------- protocol bridge
    def to_mixture_distribution(self) -> MixtureDistribution:
        """Return a MixtureDistribution of MultivariateStudentT components, the
        family-agnostic value object the conditioner operates on."""
        check_is_fitted(self)
        comps = [
            MultivariateStudentT(self.locations_[k], self.scales_[k], self.dofs_[k])
            for k in range(self.n_components)
        ]
        return MixtureDistribution(self.weights_, comps)
