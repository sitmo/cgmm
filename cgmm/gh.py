# cgmm/gh.py
# Maximum-likelihood EM (MCECM) fitter for a mixture of multivariate Generalized
# Hyperbolic distributions, mirroring the scikit-learn GaussianMixture API.
#
# The GH is a normal mean-variance mixture: X | (z=k, W) ~ N(mu_k + W gamma_k,
# W Sigma_k) with W ~ GIG(lambda_k, chi_k, psi_k). EM augments the data with the
# component label and the latent scale W. The E-step needs the GIG posterior
# moments E[W], E[1/W], E[log W] per point/component; the M-step has closed forms
# for (mu, gamma, Sigma) and a small constrained optimization for (lambda, chi,
# psi) with the identifiability constraint E[W] = 1 baked in. See
# docs/conventions.md for the equations.
from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize
from scipy.special import kve
from sklearn.mixture import GaussianMixture
from sklearn.mixture._base import BaseMixture
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import StrOptions
from sklearn.utils.validation import check_is_fitted

from .distributions import MultivariateGH, _log_kv
from .container import MixtureDistribution

Array = np.ndarray

_LAM_BOUNDS = (-50.0, 50.0)
_OMEGA_BOUNDS = (1e-3, 1e4)
_H = 1e-4  # finite-difference step for d/dlambda log K_lambda


def _R(lam: float, omega):
    """Bessel-K ratio K_{lam+1}(omega) / K_lam(omega) via the scaled kve."""
    return kve(lam + 1.0, omega) / kve(lam, omega)


def _gig_moments(lam, chi, psi):
    """E[W], E[1/W], E[log W] for W ~ GIG(lam, chi, psi).

    ``chi`` may be an array (per-point posterior); ``psi`` and ``lam`` scalars.
    """
    chi = np.asarray(chi, dtype=float)
    omega = np.sqrt(chi * psi)
    s = np.sqrt(chi / psi)
    k_lam = kve(lam, omega)
    e_w = s * kve(lam + 1.0, omega) / k_lam
    e_inv_w = kve(lam - 1.0, omega) / k_lam / s
    # E[log W] = 0.5 log(chi/psi) + d/dlam log K_lam(omega)
    dlog = (_log_kv(lam + _H, omega) - _log_kv(lam - _H, omega)) / (2.0 * _H)
    e_log_w = 0.5 * np.log(chi / psi) + dlog
    return e_w, e_inv_w, e_log_w


def _solve_gig(a_bar: float, b_bar: float, c_bar: float, lam0: float, omega0: float):
    """Maximize the GIG expected complete-data log-likelihood over (lambda, omega)
    with the identifiability constraint E[W] = 1, returning (lambda, chi, psi).

    With E[W] = 1 the GIG is reparameterized by (lambda, omega) where
    omega = sqrt(chi psi); the constraint gives chi = omega / R, psi = omega R with
    R = K_{lambda+1}(omega)/K_lambda(omega).
    """

    def neg_q(theta):
        lam, omega = theta
        if omega <= 0:
            return 1e18
        R = _R(lam, omega)
        if not np.isfinite(R) or R <= 0:
            return 1e18
        chi = omega / R
        psi = omega * R
        q = (
            lam * np.log(R)
            - _log_kv(lam, omega)
            + (lam - 1.0) * c_bar
            - 0.5 * chi * b_bar
            - 0.5 * psi * a_bar
        )
        return -float(q)

    res = minimize(
        neg_q,
        x0=np.array([lam0, omega0]),
        method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 200},
    )
    lam, omega = res.x
    lam = float(np.clip(lam, *_LAM_BOUNDS))
    omega = float(np.clip(omega, *_OMEGA_BOUNDS))
    R = _R(lam, omega)
    return lam, omega / R, omega * R  # lambda, chi, psi


class GeneralizedHyperbolicMixture(BaseMixture):
    """Mixture of multivariate Generalized Hyperbolic distributions fitted by EM.

    Mirrors :class:`sklearn.mixture.GaussianMixture`. The identifiability
    constraint E[W] = 1 is imposed per component, so the scale matrices ``scales_``
    are comparable across the redundancy of the GH parameterization.

    Parameters
    ----------
    n_components : int, default=1
    tol, reg_covar, max_iter, n_init, random_state, warm_start, verbose,
    verbose_interval :
        As in :class:`sklearn.mixture.GaussianMixture`.
    init_params : {"kmeans", "k-means++", "random", "random_from_data", "gmm"}
        ``"gmm"`` warm-starts from a fitted GaussianMixture (default).

    Attributes
    ----------
    weights_ : ndarray (n_components,)
    locations_ : ndarray (n_components, n_features)
    scales_ : ndarray (n_components, n_features, n_features)
    gammas_ : ndarray (n_components, n_features)
    lambdas_, chis_, psis_ : ndarray (n_components,)
    lower_bounds_ : list of float (per-iteration mean log-likelihood, non-decreasing)
    """

    _parameter_constraints: dict = {
        **BaseMixture._parameter_constraints,
        "init_params": [
            StrOptions({"kmeans", "k-means++", "random", "random_from_data", "gmm"})
        ],
    }

    def __init__(
        self,
        n_components: int = 1,
        *,
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: str = "gmm",
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

    # ------------------------------------------------------------------ init
    def _check_parameters(self, X, **kwargs):
        # **kwargs absorbs the array-namespace `xp` passed by scikit-learn >= 1.9.
        pass

    def _initialize_parameters(self, X, random_state, **kwargs):
        if self.init_params == "gmm":
            gmm = GaussianMixture(
                n_components=self.n_components,
                covariance_type="full",
                reg_covar=self.reg_covar,
                random_state=random_state,
            ).fit(X)
            self._initialize(X, gmm.predict_proba(X))
        else:
            super()._initialize_parameters(X, random_state, **kwargs)

    def _initialize(self, X, resp):
        n, d = X.shape
        K = self.n_components
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        locations = (resp.T @ X) / nk[:, None]
        scales = np.empty((K, d, d))
        for k in range(K):
            diff = X - locations[k]
            Sk = (resp[:, k][:, None] * diff).T @ diff / nk[k]
            Sk.flat[:: d + 1] += self.reg_covar
            scales[k] = Sk
        self.weights_ = nk / n
        self.locations_ = locations
        self.scales_ = scales
        self.gammas_ = np.zeros((K, d))  # symmetric start
        # near-Gaussian GIG start with E[W] = 1
        lam0, omega0 = 1.0, 5.0
        R = _R(lam0, omega0)
        self.lambdas_ = np.full(K, lam0)
        self.chis_ = np.full(K, omega0 / R)
        self.psis_ = np.full(K, omega0 * R)
        self._scale_chol_ = self._cholesky(self.scales_)

    @staticmethod
    def _cholesky(scales):
        return np.stack([cholesky(S, lower=True, check_finite=False) for S in scales])

    # ----------------------------------------------------- log-prob / weights
    def _component(self, k) -> MultivariateGH:
        return MultivariateGH(
            self.lambdas_[k],
            self.chis_[k],
            self.psis_[k],
            self.locations_[k],
            self.scales_[k],
            self.gammas_[k],
        )

    def _estimate_log_prob(self, X, **kwargs):
        return np.column_stack(
            [self._component(k).logpdf(X) for k in range(self.n_components)]
        )

    def _estimate_log_weights(self, **kwargs):
        return np.log(self.weights_)

    def _compute_lower_bound(self, log_resp, log_prob_norm):
        return log_prob_norm

    # ------------------------------------------------------------------ M-step
    def _m_step(self, X, log_resp, **kwargs):
        n, d = X.shape
        K = self.n_components
        resp = np.exp(log_resp)  # tau_ik
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        self.weights_ = nk / n

        new_loc = np.empty((K, d))
        new_gamma = np.empty((K, d))
        new_scale = np.empty((K, d, d))
        new_lam = np.empty(K)
        new_chi = np.empty(K)
        new_psi = np.empty(K)

        for k in range(K):
            tau = resp[:, k]
            mu, gamma = self.locations_[k], self.gammas_[k]
            L = self._scale_chol_[k]
            # Mahalanobis Q_ik and p_k under current params
            diff = (X - mu).T
            sol = solve_triangular(L, diff, lower=True, check_finite=False)
            Q = np.sum(sol * sol, axis=0)  # (n,)
            sol_g = solve_triangular(L, gamma, lower=True, check_finite=False)
            p = float(np.dot(sol_g, sol_g))

            # GIG posterior moments per point
            lam_post = self.lambdas_[k] - d / 2.0
            chi_post = self.chis_[k] + Q
            psi_post = self.psis_[k] + p
            a, b, c = _gig_moments(lam_post, chi_post, psi_post)

            # weighted averages
            a_bar = float(tau @ a) / nk[k]
            b_bar = float(tau @ b) / nk[k]
            c_bar = float(tau @ c) / nk[k]
            x_bar = (tau @ X) / nk[k]
            bx_bar = (tau * b) @ X / nk[k]

            # closed-form location & skewness
            denom = 1.0 - a_bar * b_bar
            gamma_k = (bx_bar - b_bar * x_bar) / denom
            mu_k = x_bar - gamma_k * a_bar

            # closed-form scale
            dmu = X - mu_k
            S = (tau * b)[:, None] * dmu
            Sigma_k = S.T @ dmu / nk[k] - a_bar * np.outer(gamma_k, gamma_k)
            Sigma_k.flat[:: d + 1] += self.reg_covar

            # GIG parameter update (constrained E[W] = 1)
            omega0 = float(np.sqrt(self.chis_[k] * self.psis_[k]))
            lam_k, chi_k, psi_k = _solve_gig(
                a_bar, b_bar, c_bar, self.lambdas_[k], omega0
            )

            new_loc[k], new_gamma[k], new_scale[k] = mu_k, gamma_k, Sigma_k
            new_lam[k], new_chi[k], new_psi[k] = lam_k, chi_k, psi_k

        self.locations_ = new_loc
        self.gammas_ = new_gamma
        self.scales_ = new_scale
        self.lambdas_ = new_lam
        self.chis_ = new_chi
        self.psis_ = new_psi
        self._scale_chol_ = self._cholesky(self.scales_)

    # ----------------------------------------------------------- param plumbing
    def _get_parameters(self):
        return (
            self.weights_,
            self.locations_,
            self.scales_,
            self.gammas_,
            self.lambdas_,
            self.chis_,
            self.psis_,
            self._scale_chol_,
        )

    def _set_parameters(self, params, **kwargs):
        (
            self.weights_,
            self.locations_,
            self.scales_,
            self.gammas_,
            self.lambdas_,
            self.chis_,
            self.psis_,
            self._scale_chol_,
        ) = params

    def _n_parameters(self):
        d = self.locations_.shape[1]
        K = self.n_components
        per = d + d + d * (d + 1) // 2 + 3  # loc + gamma + scale + (lambda,chi,psi)
        return int(K * per + (K - 1))

    def bic(self, X) -> float:
        return -2.0 * self.score(X) * X.shape[0] + self._n_parameters() * np.log(
            X.shape[0]
        )

    def aic(self, X) -> float:
        return -2.0 * self.score(X) * X.shape[0] + 2.0 * self._n_parameters()

    # ----------------------------------------------------------------- sampling
    def sample(self, n_samples: int = 1):
        check_is_fitted(self)
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1.")
        rng = check_random_state(self.random_state)
        n_per = rng.multinomial(n_samples, self.weights_)
        X_parts, y_parts = [], []
        for k, nk in enumerate(n_per):
            if nk == 0:
                continue
            X_parts.append(self._component(k).rvs(size=int(nk), random_state=rng))
            y_parts.append(np.full(int(nk), k))
        return np.vstack(X_parts), np.concatenate(y_parts)

    # ----------------------------------------------------- protocol bridge
    def to_mixture_distribution(self) -> MixtureDistribution:
        check_is_fitted(self)
        comps = [self._component(k) for k in range(self.n_components)]
        return MixtureDistribution(self.weights_, comps)
