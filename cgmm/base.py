# cgmm/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Union, List

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.mixture import GaussianMixture
from scipy.special import logsumexp


# ---- Small numeric utilities ----
def _log_gaussian_full(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """log N(x | mean, cov) with a Cholesky factorization (lower)."""
    from scipy.linalg import cholesky, solve_triangular

    d = mean.shape[0]
    L = cholesky(cov, lower=True, check_finite=False)
    diff = x - mean
    u = solve_triangular(L, diff, lower=True, check_finite=False)
    quad = float(np.dot(u, u))
    logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
    return -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)


def _log_gaussian_diag(x: np.ndarray, mean: np.ndarray, diag_var: np.ndarray) -> float:
    """log N(x | mean, diag(diag_var)) with diag_var as (Dy,) variance vector."""
    diff = x - mean
    return float(
        -0.5 * (np.sum(np.log(2.0 * np.pi * diag_var)) + np.sum(diff * diff / diag_var))
    )


class BaseConditionalMixture(RegressorMixin, BaseEstimator, ABC):
    """
    Base class for conditional mixture regressors modeling p(y | X).
    Subclasses must implement:
      - fit(self, X, y)
      - _compute_conditional_mixture(self, X) -> dict with keys {"weights","means","covariances"}

    Conventions for _compute_conditional_mixture(X):
      - X: (n, Dx)
      - returns dict:
          weights:     (n, K)                     rows sum to 1
          means:       (n, K, Dy)
          covariances: one of
                       (K, Dy, Dy) or (n, K, Dy, Dy)   # full
                       (K, Dy)     or (n, K, Dy)       # diag (variance vectors)
    """

    def __init__(self, *, return_cov: bool = False):
        self.return_cov = return_cov

    # Sklearn 1.6+: declare tag customizations here (no _more_tags).
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.multi_output = True
        tags.regressor_tags.poor_score = (
            True  # we define score as mean loglik; opt out of default checks
        )
        return tags

    # ----- abstract API -----
    @abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike):  # must set n_features_in_, n_targets_
        ...

    @abstractmethod
    def _compute_conditional_mixture(self, X: ArrayLike) -> Dict[str, np.ndarray]:
        """Compute mixture parameters of p(y | X) without collapsing."""
        ...

    # Optional override for γ(x,y)
    def responsibilities(
        self, X: ArrayLike, y: Optional[ArrayLike] = None
    ) -> np.ndarray:
        """
        If y is None: return gating weights π_k(X) with shape (n, K).
        If y is provided: override in subclass to return γ_{ik} ∝ π_k(X_i) N(y_i; μ_{ik}, Σ_{ik}).
        """
        params = self._compute_conditional_mixture(X)
        W = params["weights"]
        if y is not None:
            raise NotImplementedError("Override responsibilities(X,y) in subclass.")
        return W

    # ----- sklearn-style API -----
    def predict(
        self,
        X: ArrayLike,
        return_cov: Optional[bool] = None,
        return_components: Optional[bool] = None,
    ):
        """
        Predict E[y | X]. Optionally return predictive covariance and/or raw components.

        Returns:
          mean: (n,) for single-output or (n, Dy) for multi-output
          (mean, cov): if return_cov==True, cov has shape (n, Dy, Dy) or (n,) when Dy==1
          (mean, components): if return_components==True
          (mean, cov, components): if both requested
        """
        check_is_fitted(self, attributes=["n_features_in_", "n_targets_"])
        params = self._compute_conditional_mixture(X)
        W, M, S = (
            params["weights"],
            params["means"],
            params["covariances"],
        )  # (n,K), (n,K,Dy), cov spec

        # Mixture mean: μ = Σ_k w_k m_k
        mean = (W[..., None] * M).sum(axis=1)  # (n, Dy)
        single_output = self.n_targets_ == 1
        mean_out = mean[:, 0] if single_output else mean

        need_cov = (return_cov is True) or (return_cov is None and self.return_cov)
        if not need_cov and not return_components:
            return mean_out

        out = [mean_out]

        if need_cov:
            cov = self._mixture_covariance_from_params(W, M, S)  # (n, Dy, Dy)
            if single_output:
                cov = cov[:, 0, 0]  # (n,)
            out.append(cov)

        if return_components:
            out.append(params)

        return out[0] if len(out) == 1 else tuple(out)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Mean conditional log-likelihood: np.mean(log p(y | X))."""
        ll = self.log_prob(X, y)
        return float(np.mean(ll))

    def log_prob(self, X: ArrayLike, y: ArrayLike) -> np.ndarray:
        """
        Pointwise conditional log-likelihoods using current mixture parameters.
        Supports both full and diag covariances.
        """
        check_is_fitted(self, attributes=["n_features_in_", "n_targets_"])
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1 and self.n_targets_ == 1:
            y = y[:, None]

        params = self._compute_conditional_mixture(X)
        W, M, S = (
            params["weights"],
            params["means"],
            params["covariances"],
        )  # (n,K), (n,K,Dy), cov spec
        n, K = W.shape
        Dy = M.shape[-1]
        tiny = np.finfo(float).tiny

        # Normalize S to one of the two canonical forms:
        # full: (n,K,Dy,Dy); diag: (n,K,Dy) variance vectors
        is_full = False
        # is_diag = False
        if (
            S.ndim == 4
            and S.shape[0] == n
            and S.shape[1] == K
            and S.shape[2] == Dy
            and S.shape[3] == Dy
        ):
            is_full = True
        elif S.ndim == 3 and S.shape == (K, Dy, Dy):
            S = np.broadcast_to(S, (n,) + S.shape)  # -> (n,K,Dy,Dy)
            is_full = True
        elif S.ndim == 3 and S.shape == (n, K, Dy):
            pass
            # is_diag = True
        elif S.ndim == 2 and S.shape == (K, Dy):
            S = np.broadcast_to(S, (n,) + S.shape)  # -> (n,K,Dy)
            # is_diag = True
        else:
            raise ValueError(
                "covariances must be (K,Dy,Dy), (n,K,Dy,Dy), (K,Dy), or (n,K,Dy)"
            )

        logp = np.empty(n, dtype=float)
        for i in range(n):
            yi = y[i]
            if is_full:
                log_terms = np.array(
                    [
                        np.log(W[i, k] + tiny)
                        + _log_gaussian_full(yi, M[i, k], S[i, k])
                        for k in range(K)
                    ],
                    dtype=float,
                )
            else:  # diag
                log_terms = np.array(
                    [
                        np.log(W[i, k] + tiny)
                        + _log_gaussian_diag(yi, M[i, k], S[i, k])
                        for k in range(K)
                    ],
                    dtype=float,
                )
            logp[i] = logsumexp(log_terms)
        return logp

    # ----- helpers -----
    @staticmethod
    def _mixture_covariance_from_params(
        W: np.ndarray, M: np.ndarray, S: np.ndarray
    ) -> np.ndarray:
        """
        Compute Var[y|X] for a Gaussian mixture given:
          W: (n,K), M: (n,K,Dy),
          S: (K,Dy,Dy)/(n,K,Dy,Dy) for full, or (K,Dy)/(n,K,Dy) for diag (variance vectors).
        Returns (n,Dy,Dy).
        """
        n, K = W.shape
        Dy = M.shape[-1]

        # Canonicalize S to (n,K,Dy,Dy)
        if S.ndim == 4:
            pass  # already (n,K,Dy,Dy)
        elif S.ndim == 3 and S.shape == (K, Dy, Dy):
            S = np.broadcast_to(S, (n,) + S.shape)
        elif S.ndim == 3 and S.shape == (n, K, Dy):  # diag
            S = np.eye(Dy)[None, None, :, :] * S[..., None]  # (n,K,Dy,Dy)
        elif S.ndim == 2 and S.shape == (K, Dy):  # diag shared
            S = np.broadcast_to(
                np.eye(Dy)[None, None, :, :] * S[None, :, :, None], (n, K, Dy, Dy)
            )
        else:
            raise ValueError(
                "covariances must be (K,Dy,Dy), (n,K,Dy,Dy), (K,Dy), or (n,K,Dy)"
            )

        mean = (W[..., None] * M).sum(axis=1)  # (n, Dy)
        diff = M - mean[:, None, :]  # (n, K, Dy)
        diff_outer = diff[..., :, None] * diff[..., None, :]  # (n, K, Dy, Dy)
        cov = (W[:, :, None, None] * (S + diff_outer)).sum(axis=1)  # (n, Dy, Dy)
        return cov


class ConditionalMixin:
    """
    Mixin providing condition() method for conditional mixture models.
    Subclasses should implement condition() or rely on subclass-specific conditioning.
    """

    def condition(self, X: ArrayLike) -> Union[GaussianMixture, List[GaussianMixture]]:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement condition()."
        )
