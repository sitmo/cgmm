# cgmm/conditioner.py
# Conditional Gaussian Mixture Model Conditioner
# This file implements a conditional Gaussian mixture model conditioner.
# It is a subclass of the sklearn.base.BaseEstimator class.
# It is used to condition a fitted Gaussian mixture model on a fixed set of variables.
# It is used to condition a fitted Gaussian mixture model on a fixed set of variables.
#
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import cholesky, solve_triangular

from sklearn.base import BaseEstimator
from sklearn.mixture import GaussianMixture

from .container import GaussianMixtureDistribution, MixtureDistribution


Array = np.ndarray


def _full_covariances(gmm: GaussianMixture) -> Array:
    """
    Return per-component full covariance matrices of shape (K, d, d) regardless
    of the fitted GaussianMixture's covariance_type ('full', 'tied', 'diag',
    'spherical'). Conditioning math operates on full matrices; expanding the
    cheaper parameterizations here lets the conditioner accept any sklearn GMM.
    """
    K = gmm.means_.shape[0]
    d = gmm.means_.shape[1]
    cov = np.asarray(gmm.covariances_)
    ctype = gmm.covariance_type
    if ctype == "full":  # (K, d, d)
        return cov
    if ctype == "tied":  # (d, d) shared across components
        return np.broadcast_to(cov, (K, d, d)).copy()
    if ctype == "diag":  # (K, d) variance vectors
        out = np.zeros((K, d, d))
        idx = np.arange(d)
        out[:, idx, idx] = cov
        return out
    if ctype == "spherical":  # (K,) scalar variances
        out = np.zeros((K, d, d))
        idx = np.arange(d)
        out[:, idx, idx] = cov[:, None]
        return out
    raise NotImplementedError(f"Unsupported covariance_type={ctype!r}.")


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


@dataclass
class _PerComponentCache:
    mu_X: Array  # (d_X,)
    mu_y: Array  # (d_y,)
    chol_SXX: Array  # (d_X, d_X) lower Cholesky of S_XX (with reg)
    A: Array  # (d_y, d_X) = S_yX S_XX^{-1}
    C: Array  # (d_y, d_y) = S_yy - A S_Xy
    lognorm_const: float  # -0.5 * (d_X log(2π) + log|S_XX|) cached part
    # For logpdf we still need the quadratic term which depends on x.


class MixtureConditioner(BaseEstimator):
    """
    Precompute reusable terms to condition a fitted mixture on a fixed set of
    variables X (the conditioning block) and return the mixture over the y-block.

    Works for any mixture in the ``Mixture`` protocol:

    - A fitted sklearn ``GaussianMixture`` (or ``GaussianMixtureDistribution``)
      takes a cached fast path and the result is a ``GaussianMixtureDistribution``
      (so ``isinstance(result, GaussianMixture)`` stays True — full backward
      compatibility for Gaussian users).
    - A ``MixtureDistribution`` (any family — Student-t, GH, ...) or any estimator
      exposing ``to_mixture_distribution()`` (e.g. ``StudentTMixture``) takes a
      generic per-component path and the result is a ``MixtureDistribution`` whose
      components stay in the original family.

    ``GMMConditioner`` is kept as an alias of this class.

    Conventions
    -----------
    - The feature vector is split into X (conditioning) and y (targets).
      ``cond_idx`` selects X columns; the complement forms y.
    - The returned mixture is defined over the y-dim only.
    - Gaussian input of any ``covariance_type`` ('full', 'tied', 'diag',
      'spherical') is accepted; cheaper parameterizations are expanded to full.

    Parameters
    ----------
    mixture_estimator : GaussianMixture | MixtureDistribution | estimator with to_mixture_distribution()
        A *fitted* joint mixture over [X, y].
    cond_idx : Sequence[int]
        Indices of variables to condition on (X).
    reg_covar : float, default=1e-9
        Added to the diagonal of S_XX before inversion / Cholesky on the Gaussian
        fast path. (The generic families rely on the scale matrices their fitters
        already regularize, so this is a no-op for them.)
    """

    def __init__(
        self,
        mixture_estimator,
        cond_idx: Sequence[int],
        reg_covar: float = 1e-9,
    ):
        self.mixture_estimator = mixture_estimator
        self.cond_idx = tuple(int(i) for i in cond_idx)
        self.reg_covar = float(reg_covar)

        # Fitted state
        self._prepared: bool = False
        self._mode: Optional[str] = None  # "gaussian" | "generic"
        self._mixture: Optional[MixtureDistribution] = None  # generic path
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
        self._mode = None
        self._mixture = None
        self._cache = []
        return self

    def _validate_cond_idx(self, d: int) -> Array:
        """Validate self.cond_idx against a joint dimension d; return target idx."""
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
        self._d_X = int(cond_idx.size)
        self._d_y = int(target_idx.size)
        return target_idx

    def _validate_and_prepare(self):
        est = self.mixture_estimator
        if isinstance(est, GaussianMixture):
            self._mode = "gaussian"
            self._prepare_gaussian(est)
        elif isinstance(est, MixtureDistribution):
            self._mode = "generic"
            self._prepare_generic(est)
        elif callable(getattr(est, "to_mixture_distribution", None)):
            # fitted family-specific mixture (e.g. StudentTMixture)
            self._mode = "generic"
            self._prepare_generic(est.to_mixture_distribution())
        else:
            raise TypeError(
                "mixture_estimator must be a fitted sklearn GaussianMixture, a "
                "MixtureDistribution, or an estimator exposing "
                "to_mixture_distribution()."
            )
        self._prepared = True

    def _prepare_generic(self, mixture: MixtureDistribution):
        self._mixture = mixture
        self._validate_cond_idx(int(mixture.dim))

    def _prepare_gaussian(self, gmm: GaussianMixture):
        if not hasattr(gmm, "means_"):
            raise ValueError(
                "mixture_estimator must be *fitted* (missing attributes like means_)."
            )
        d = gmm.means_.shape[1]
        cond_idx = np.asarray(self.cond_idx, dtype=int)
        target_idx = self._validate_cond_idx(d)

        # Build per-component caches
        K = gmm.n_components
        means = gmm.means_  # (K, d)
        covs = _full_covariances(gmm)  # (K, d, d) regardless of covariance_type

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

    def precompute(self) -> "MixtureConditioner":
        self._validate_and_prepare()
        return self

    # ---- Core API ----

    def condition(
        self,
        x: Union[ArrayLike, Array],
    ) -> Union[GaussianMixture, List[GaussianMixture]]:
        """
        Condition the mixture on X = x and return the mixture over y.

        Parameters
        ----------
        x : array-like
            If 1D of shape (d_X,), returns a single mixture.
            If 2D of shape (n, d_X), returns a list of length n.

        Returns
        -------
        Mixture or list[Mixture]
            Mixture over the target block (y). A Gaussian input yields a
            ``GaussianMixtureDistribution``; any other family yields a
            ``MixtureDistribution`` whose components stay in that family.
        """
        if not self._prepared:
            self._validate_and_prepare()

        X = np.asarray(x, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != self._d_X:
            raise ValueError(f"Expected x with shape (*, {self._d_X}), got {X.shape}.")

        if self._mode == "gaussian":
            results = [self._condition_gaussian_one(X[i]) for i in range(X.shape[0])]
        else:
            cond_idx = self.cond_idx
            results = [
                self._mixture.condition(cond_idx, X[i]) for i in range(X.shape[0])
            ]

        return results[0] if len(results) == 1 else results

    def _condition_gaussian_one(self, xi: Array) -> GaussianMixtureDistribution:
        """Fast cached Gaussian conditioning for a single query point xi."""
        gmm = self.mixture_estimator
        K = gmm.n_components
        d_y = self._d_y  # type: ignore
        weights = gmm.weights_

        log_w = np.empty(K)
        mu_y_given_x = np.empty((K, d_y))
        Sigma_y_given_x = np.empty((K, d_y, d_y))

        for k in range(K):
            c = self._cache[k]
            # log N(x | mu_X, S_XX) for the weight update
            diff = xi - c.mu_X
            u = solve_triangular(c.chol_SXX, diff, lower=True, check_finite=False)
            log_w[k] = np.log(weights[k]) + c.lognorm_const - 0.5 * np.dot(u, u)
            # mu_{y|x} = mu_y + A (x - mu_X);  Sigma_{y|x} = C (independent of x)
            mu_y_given_x[k] = c.mu_y + c.A @ diff
            Sigma_y_given_x[k] = c.C

        m = np.max(log_w)
        w = np.exp(log_w - m)
        w /= w.sum()

        # Build a *fitted* GaussianMixtureDistribution over y. This is a real
        # sklearn GaussianMixture subclass, so isinstance(..., GaussianMixture)
        # stays True while gaining the Mixture-protocol methods.
        gmmy = GaussianMixtureDistribution(
            n_components=K,
            covariance_type="full",
            random_state=getattr(gmm, "random_state", None),
        )
        gmmy.weights_ = w
        gmmy.means_ = mu_y_given_x
        gmmy.covariances_ = Sigma_y_given_x
        precisions_cholesky = np.empty_like(Sigma_y_given_x)
        for k in range(K):
            precisions_cholesky[k] = _precision_cholesky_from_cov_full(
                Sigma_y_given_x[k]
            )
        gmmy.precisions_cholesky_ = precisions_cholesky
        gmmy.converged_ = True
        gmmy.n_iter_ = 0
        gmmy.lower_bound_ = np.nan
        return gmmy


# Backward-compatible alias: the conditioner was historically named
# GMMConditioner and that name (and its constructor signature) must keep working.
GMMConditioner = MixtureConditioner
