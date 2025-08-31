from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import cholesky, solve_triangular

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.mixture import GaussianMixture


Array = np.ndarray


def _as_1d(x: ArrayLike) -> Array:
    x = np.asarray(x, dtype=float)
    if x.ndim == 0:
        x = x[None]
    if x.ndim == 2 and x.shape[0] == 1:
        return x[0]
    if x.ndim != 1:
        raise ValueError(
            "Expected 1D array for a single x; use shape (n_samples, n_X) for batches."
        )
    return x


def _precision_cholesky_from_cov_full(cov: Array) -> Array:
    """
    Convert full covariance matrix to scikit-learn's precisions_cholesky factor.

    For covariance S = L L^T (Cholesky, lower), precision P = S^{-1} = L^{-T} L^{-1}.
    The 'precisions_cholesky' stored by sklearn for 'full' covariance is the Cholesky
    factor of P (lower), which equals L^{-T}. However, sklearn stores the LOWER factor
    such that precisions_cholesky @ precisions_cholesky.T = P.

    Derivation:
      S = L L^T, with L lower. Then S^{-1} = L^{-T} L^{-1}.
      Let R = L^{-1}. Then P = R^T R. A Cholesky lower factor of P is R^T = L^{-T}.

    Implementation:
      compute L = cholesky(S, lower=True)
      compute L_inv_T = solve_triangular(L, np.eye(L.shape[0]), lower=True).T = L^{-T}
      return L_inv_T  (lower triangular)
    """
    L = cholesky(cov, lower=True, check_finite=False)
    eye = np.eye(L.shape[0])
    # Solve L * Z = I  -> Z = L^{-1}
    Z = solve_triangular(L, eye, lower=True, check_finite=False)
    # Return L^{-T} = Z.T  (this is a lower-triangular factor of the precision)
    return Z.T


def _log_gaussian_cholesky(x: Array, mean: Array, chol: Array) -> float:
    """
    Log N(x | mean, Sigma) given lower-triangular Cholesky factor of Sigma (chol).

    Returns a scalar. No constant batching here; caller can vectorize.
    """
    d = mean.shape[0]
    diff = x - mean
    # Solve L * u = diff  -> u = L^{-1} diff
    u = solve_triangular(chol, diff, lower=True, check_finite=False)
    quad = np.dot(u, u)  # ||u||^2
    logdet = 2.0 * np.sum(np.log(np.diag(chol)))
    return -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)


@dataclass
class _PerComponentCache:
    mu_X: Array  # (d_X,)
    mu_y: Array  # (d_y,)
    chol_SXX: Array  # (d_X, d_X) lower Cholesky of S_XX (with reg)
    A: Array  # (d_y, d_X) = S_yX S_XX^{-1}
    C: Array  # (d_y, d_y) = S_yy - A S_Xy
    lognorm_const: float  # -0.5 * (d_X log(2π) + log|S_XX|) cached part
    # For logpdf we still need the quadratic term which depends on x.


class GMMConditioner(BaseEstimator):
    """
    Precompute reusable terms to condition a fitted GaussianMixture on a fixed set of
    variables X (conditioning block). Returns a new GaussianMixture over the y-block.

    Conventions
    -----------
    - Original feature vector is split into X (conditioning) and y (targets).
      'cond_idx' selects X columns; the complement forms y.
    - Returned GaussianMixture is defined over y-dim only.
    - Only supports covariance_type='full' for the input mixture.

    Parameters
    ----------
    mixture_estimator : GaussianMixture
        A *fitted* sklearn GaussianMixture with covariance_type="full".
    cond_idx : Sequence[int]
        Indices of variables to condition on (X).
    reg_covar : float, default=1e-9
        Added to diagonal of S_XX before inversion / Cholesky.
    """

    def __init__(
        self,
        mixture_estimator: GaussianMixture,
        cond_idx: Sequence[int],
        reg_covar: float = 1e-9,
    ):
        self.mixture_estimator = mixture_estimator
        self.cond_idx = tuple(int(i) for i in cond_idx)
        self.reg_covar = float(reg_covar)

        # Fitted state
        self._prepared: bool = False
        self._target_idx: Optional[Tuple[int, ...]] = None
        self._cache: List[_PerComponentCache] = []
        self._d_X: Optional[int] = None
        self._d_y: Optional[int] = None

    # sklearn estimator plumbing
    def get_params(self, deep: bool = True):
        return {
            "mixture_estimator": self.mixture_estimator,
            "cond_idx": self.cond_idx,
            "reg_covar": self.reg_covar,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        # Invalidate prepared caches
        self._prepared = False
        self._cache = []
        return self

    def _validate_and_prepare(self):
        gmm = self.mixture_estimator
        if not isinstance(gmm, GaussianMixture):
            raise TypeError(
                "mixture_estimator must be a fitted sklearn.mixture.GaussianMixture instance."
            )
        if not hasattr(gmm, "means_"):
            raise ValueError(
                "mixture_estimator must be *fitted* (missing attributes like means_)."
            )
        if gmm.covariance_type != "full":
            raise NotImplementedError(
                "Only covariance_type='full' is supported currently."
            )

        d = gmm.means_.shape[1]
        cond_idx = np.asarray(self.cond_idx, dtype=int)
        if cond_idx.ndim != 1:
            raise ValueError("cond_idx must be a 1D sequence of indices.")
        if np.unique(cond_idx).size != cond_idx.size:
            raise ValueError("cond_idx contains duplicates.")
        if cond_idx.min(initial=0) < 0 or cond_idx.max(initial=-1) >= d:
            raise ValueError("cond_idx contains out-of-range indices.")

        mask = np.ones(d, dtype=bool)
        mask[cond_idx] = False
        target_idx = np.nonzero(mask)[0]

        self._target_idx = tuple(int(i) for i in target_idx)
        self._d_X = cond_idx.size
        self._d_y = target_idx.size

        # Build per-component caches
        K = gmm.n_components
        means = gmm.means_  # (K, d)
        covs = gmm.covariances_  # (K, d, d)

        cache: List[_PerComponentCache] = []
        for k in range(K):
            mu = means[k]
            S = covs[k]

            mu_X = mu[cond_idx]
            mu_y = mu[target_idx]

            S_XX = S[np.ix_(cond_idx, cond_idx)].copy()
            if self.reg_covar > 0.0:
                S_XX.flat[:: S_XX.shape[0] + 1] += self.reg_covar

            S_Xy = S[np.ix_(cond_idx, target_idx)]
            S_yX = S_Xy.T
            S_yy = S[np.ix_(target_idx, target_idx)]

            # Cholesky of S_XX and solve for A = S_yX S_XX^{-1} without explicit inverse
            chol_SXX = cholesky(S_XX, lower=True, check_finite=False)
            # Solve for each row of A: we want A = S_yX @ S_XX^{-1}
            # Equivalent: (A.T) = (S_XX^{-T}) @ (S_yX.T)
            A_T = solve_triangular(chol_SXX, S_yX.T, lower=True, check_finite=False)
            A_T = solve_triangular(chol_SXX.T, A_T, lower=False, check_finite=False)
            A = A_T.T  # (d_y, d_X)

            # C = S_yy - A S_Xy
            C = S_yy - A @ S_Xy

            # log-normalization constant for N(x | mu_X, S_XX) without quadratic term
            logdet_SXX = 2.0 * np.sum(np.log(np.diag(chol_SXX)))
            lognorm_const = -0.5 * (self._d_X * np.log(2.0 * np.pi) + logdet_SXX)

            cache.append(
                _PerComponentCache(
                    mu_X=mu_X,
                    mu_y=mu_y,
                    chol_SXX=chol_SXX,
                    A=A,
                    C=C,
                    lognorm_const=lognorm_const,
                )
            )

        self._cache = cache
        self._prepared = True

    def precompute(self) -> "GMMConditioner":
        self._validate_and_prepare()
        return self

    # ---- Core API ----

    def condition(
        self,
        x: Union[ArrayLike, Array],
    ) -> Union[GaussianMixture, List[GaussianMixture]]:
        """
        Condition the mixture on X = x and return a GaussianMixture over y.

        Parameters
        ----------
        x : array-like
            If 1D of shape (d_X,), returns a single GaussianMixture.
            If 2D of shape (n, d_X), returns a list of length n.

        Returns
        -------
        GaussianMixture or list[GaussianMixture]
            Mixture over the target block (y) with updated weights and
            per-component conditional means/covariances.
        """
        if not self._prepared:
            self._validate_and_prepare()

        gmm = self.mixture_estimator
        K = gmm.n_components
        d_y = self._d_y  # type: ignore
        cond_idx = np.asarray(self.cond_idx, dtype=int)

        X = np.asarray(x, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != len(cond_idx):
            raise ValueError(
                f"Expected x with shape (*, {len(cond_idx)}), got {X.shape}."
            )

        # Pre allocate arrays
        weights = gmm.weights_  # (K,)
        post_mixtures: List[GaussianMixture] = []

        # Precompute per-component *constant* Σ_{y|X} and per-sample μ_{y|x}
        # Also compute per-sample component log-likelihood in X for weight updates.
        for i in range(X.shape[0]):
            xi = X[i]
            log_w = np.empty(K)
            mu_y_given_x = np.empty((K, d_y))
            Sigma_y_given_x = np.empty((K, d_y, d_y))

            for k in range(K):
                c = self._cache[k]
                # log N(x | mu_X, S_XX)
                diff = xi - c.mu_X
                # Solve L u = diff
                u = solve_triangular(c.chol_SXX, diff, lower=True, check_finite=False)
                quad = np.dot(u, u)
                log_w[k] = np.log(weights[k]) + c.lognorm_const - 0.5 * quad

                # μ_{y|x} = μ_y + A (x - μ_X)
                mu_y_given_x[k] = c.mu_y + c.A @ diff

                # Σ_{y|x} = C  (independent of x)
                Sigma_y_given_x[k] = c.C

            # Normalize weights
            # Use log-sum-exp for numerical stability
            m = np.max(log_w)
            w = np.exp(log_w - m)
            w /= w.sum()

            # Build a *fitted* sklearn GaussianMixture over y.
            gmmy = GaussianMixture(
                n_components=K,
                covariance_type="full",
                random_state=getattr(gmm, "random_state", None),
            )
            gmmy.weights_ = w
            gmmy.means_ = mu_y_given_x
            gmmy.covariances_ = Sigma_y_given_x
            # Compute precisions_cholesky_ per component
            precisions_cholesky = np.empty_like(Sigma_y_given_x)
            for k in range(K):
                precisions_cholesky[k] = _precision_cholesky_from_cov_full(
                    Sigma_y_given_x[k]
                )
            gmmy.precisions_cholesky_ = precisions_cholesky

            # Mark as "converged" to make gmmy.sample() etc. usable.
            gmmy.converged_ = True
            gmmy.n_iter_ = 0
            gmmy.lower_bound_ = np.nan

            post_mixtures.append(gmmy)

        return post_mixtures[0] if len(post_mixtures) == 1 else post_mixtures


class ConditionalGMMRegressor(BaseEstimator, RegressorMixin):
    """
    Regressor-style interface computing posterior mean E[y | X=x] (and optionally
    total posterior covariance) from a fitted GaussianMixture.

    Parameters
    ----------
    mixture_estimator : GaussianMixture
        A *fitted* sklearn GaussianMixture with covariance_type="full".
    cond_idx : Sequence[int]
        Indices of conditioning variables X. The complement defines y.
    return_cov : bool, default=False
        If True, `predict_cov` is enabled to return total posterior covariance
        for each x (mixture of covariances + between-component term).
    reg_covar : float, default=1e-9
        Regularization added to S_XX prior to inversion/Cholesky.
    """

    def __init__(
        self,
        mixture_estimator: GaussianMixture,
        cond_idx: Sequence[int],
        return_cov: bool = False,
        reg_covar: float = 1e-9,
    ):
        self.mixture_estimator = mixture_estimator
        self.cond_idx = tuple(int(i) for i in cond_idx)
        self.return_cov = bool(return_cov)
        self.reg_covar = float(reg_covar)

        self._cond: Optional[GMMConditioner] = None

    def get_params(self, deep: bool = True):
        return {
            "mixture_estimator": self.mixture_estimator,
            "cond_idx": self.cond_idx,
            "return_cov": self.return_cov,
            "reg_covar": self.reg_covar,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        self._cond = None
        return self

    # For sklearn API; no fitting performed here beyond validation.
    def fit(self, X: ArrayLike, y=None):  # noqa: N803 (sklearn signature)
        # We accept X only to comply with estimator API; the mixture is already fitted.
        self._cond = GMMConditioner(
            mixture_estimator=self.mixture_estimator,
            cond_idx=self.cond_idx,
            reg_covar=self.reg_covar,
        ).precompute()
        return self

    def _ensure(self):
        if self._cond is None:
            # Try to construct lazily if user forgot fit()
            self._cond = GMMConditioner(
                mixture_estimator=self.mixture_estimator,
                cond_idx=self.cond_idx,
                reg_covar=self.reg_covar,
            ).precompute()

    def predict(self, X: ArrayLike) -> Array:  # noqa: N803 (sklearn signature)
        """
        Posterior mean E[y | X=x] for each row x in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, d_X)

        Returns
        -------
        y_mean : ndarray, shape (n_samples, d_y)
        """
        self._ensure()
        cond = self._cond  # type: ignore

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        mixtures = cond.condition(X)  # list[GaussianMixture]

        if isinstance(mixtures, GaussianMixture):
            mixtures = [mixtures]

        n = len(mixtures)
        d_y = mixtures[0].means_.shape[1]
        out = np.empty((n, d_y))
        for i, gmmy in enumerate(mixtures):
            # E[y | x] = sum_k w_k * mu_k
            out[i] = (gmmy.weights_[:, None] * gmmy.means_).sum(axis=0)
        return out if out.shape[0] > 1 else out[0]

    def predict_cov(self, X: ArrayLike) -> Union[Array, List[Array]]:
        """
        Total posterior covariance Var[y | X=x] for each row in X.

        For a Gaussian mixture posterior:
          Var[y|x] = sum_k w_k (Σ_k + μ_k μ_k^T) - m m^T,
        where m = sum_k w_k μ_k.

        Returns
        -------
        cov : ndarray of shape (d_y, d_y) for a single x,
              or list of ndarrays for multiple x.
        """
        self._ensure()
        cond = self._cond  # type: ignore

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]

        mixtures = cond.condition(X)  # list or single
        if isinstance(mixtures, GaussianMixture):
            mixtures = [mixtures]

        covs: List[Array] = []
        for gmmy in mixtures:
            w = gmmy.weights_[:, None, None]  # (K,1,1)
            mu = gmmy.means_[:, :, None]  # (K,d_y,1)
            cov = gmmy.covariances_  # (K,d_y,d_y)

            # Sum w_k * (Σ_k + μ_k μ_k^T)
            term = (w * (cov + mu @ np.swapaxes(mu, 1, 2))).sum(axis=0)
            m = (gmmy.weights_[:, None] * gmmy.means_).sum(axis=0)  # (d_y,)
            cov_total = term - np.outer(m, m)
            covs.append(cov_total)

        return covs[0] if len(covs) == 1 else covs
