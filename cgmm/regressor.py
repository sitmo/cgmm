# cgmm/regressor.py
# Conditional Gaussian Mixture Model Regressor (joint GMM, then condition)
from __future__ import annotations

import numpy as np
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.mixture import GaussianMixture

from .base import BaseConditionalMixture, ConditionalMixin, _log_gaussian_full
from .conditioner import GMMConditioner


class ConditionalGMMRegressor(BaseConditionalMixture, ConditionalMixin):
    """
    Learn a joint GMM over [X, y] and produce p(y|X) by analytic conditioning.

    Public API inherited from BaseConditionalMixture:
      - fit(X, y)
      - predict(X, return_cov=False, return_components=False)
      - _compute_conditional_mixture(X) -> dict(weights, means, covariances)
      - score(X, y) / log_prob(X, y)
      - responsibilities(X, y=None)
      - sample(X, n_samples=1)  # y|X sampling
    """

    def __init__(
        self,
        n_components: int = 2,
        covariance_type: str = "full",
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        max_iter: int = 200,
        random_state=None,
        init_params: str = "kmeans",
        cond_idx=None,
        *,
        return_cov: bool = False,
    ):
        super().__init__(return_cov=return_cov)
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.random_state = random_state
        self.init_params = init_params
        self.cond_idx = cond_idx

    def fit(self, X, y):
        # Validates and sets n_features_in_
        X, y = validate_data(
            self,
            X,
            y,
            accept_sparse=False,
            y_numeric=True,
            multi_output=True,
        )

        y = np.asarray(y)
        y2 = y[:, None] if y.ndim == 1 else y
        Z = np.concatenate([X, y2], axis=1)

        self.dx_ = X.shape[1]
        self.dy_ = y2.shape[1]
        self.n_targets_ = self.dy_  # required by BaseConditionalMixture

        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            tol=self.tol,
            reg_covar=self.reg_covar,
            max_iter=self.max_iter,
            random_state=self.random_state,
            init_params=self.init_params,
        ).fit(Z)

        # Expose iteration count and convergence status
        self.n_iter_ = int(getattr(gmm, "n_iter_", 1)) or 1
        self.converged_ = getattr(gmm, "converged_", True)

        cond_idx = (
            np.arange(self.dx_) if self.cond_idx is None else tuple(self.cond_idx)
        )
        self._cond = GMMConditioner(
            mixture_estimator=gmm,
            cond_idx=cond_idx,
            reg_covar=self.reg_covar,
        ).precompute()
        self._gmm_joint_ = gmm  # keep reference if needed downstream

        # For BaseConditionalMixture checks
        self.n_features_in_ = self.dx_
        return self

    # ---- BaseConditionalMixture required method ----
    def _compute_conditional_mixture(self, X):
        """
        Return mixture params of p(y|X):
          weights: (n, K)
          means: (n, K, Dy)
          covariances: (K, Dy, Dy)  # independent of X for joint-GMM conditioning
        """
        check_is_fitted(self, attributes=["_cond", "n_features_in_", "n_targets_"])
        X = validate_data(self, X, reset=False)

        gm = self._cond.condition(X)  # GaussianMixture or list[GaussianMixture]
        K = self.n_components
        Dy = self.dy_

        if isinstance(gm, GaussianMixture):
            # Single-sample path
            w = gm.weights_[None, ...]  # (1, K)
            m = gm.means_[None, ...]  # (1, K, Dy)
            S = gm.covariances_  # (K, Dy, Dy)
            return {"weights": w, "means": m, "covariances": S}

        # Batch path
        n = len(gm)
        W = np.empty((n, K))
        M = np.empty((n, K, Dy))
        # Covariances are x-independent; take from the first mixture
        S = gm[0].covariances_
        for i, g in enumerate(gm):
            W[i] = g.weights_
            M[i] = g.means_
        return {"weights": W, "means": M, "covariances": S}

    # ---- Optional: posteriors and sampling convenience ----
    def responsibilities(self, X, y=None):
        """
        If y is None: return gating weights after conditioning, shape (n, K).
        If y provided: return γ_{ik} ∝ w_{ik} N(y_i; m_{ik}, S_k) normalized over k.
        """
        params = self._compute_conditional_mixture(X)
        W, M, S = params["weights"], params["means"], params["covariances"]
        if y is None:
            return W

        y = np.asarray(y, dtype=float)
        if y.ndim == 1 and self.dy_ == 1:
            y = y[:, None]
        n, K = W.shape
        out = np.empty_like(W)
        # S is (K,Dy,Dy); same for all samples
        for i in range(n):
            log_terms = np.empty(K)
            for k in range(K):
                log_terms[k] = np.log(W[i, k] + 1e-300) + _log_gaussian_full(
                    y[i], M[i, k], S[k]
                )
            m = np.max(log_terms)
            r = np.exp(log_terms - m)
            out[i] = r / r.sum()
        return out

    def sample(self, X, n_samples=1):
        """
        Sample y|X.
        Returns:
          if X is (n,Dx): (n, n_samples, Dy)
          if X is a single sample (Dx,): (n_samples, Dy)
        """
        check_is_fitted(self, attributes=["_cond"])
        X = validate_data(self, X, reset=False)

        gm = self._cond.condition(X)

        # Use truly random seed for independence (scikit-learn compatible)
        rng = np.random.RandomState()

        if isinstance(gm, GaussianMixture):
            # Create a new GaussianMixture with random state for independence
            new_gm = GaussianMixture(
                n_components=gm.n_components,
                covariance_type=gm.covariance_type,
                random_state=rng,
            )
            # Copy the fitted parameters
            new_gm.weights_ = gm.weights_.copy()
            new_gm.means_ = gm.means_.copy()
            new_gm.covariances_ = gm.covariances_.copy()
            new_gm.precisions_cholesky_ = gm.precisions_cholesky_.copy()
            new_gm.converged_ = gm.converged_
            new_gm.n_iter_ = gm.n_iter_
            new_gm.lower_bound_ = gm.lower_bound_

            S = new_gm.sample(n_samples=n_samples)[0]  # (n_samples, Dy)
            return S

        out = []
        for g in gm:
            # Create a new GaussianMixture with random state for independence
            new_g = GaussianMixture(
                n_components=g.n_components,
                covariance_type=g.covariance_type,
                random_state=rng,
            )
            # Copy the fitted parameters
            new_g.weights_ = g.weights_.copy()
            new_g.means_ = g.means_.copy()
            new_g.covariances_ = g.covariances_.copy()
            new_g.precisions_cholesky_ = g.precisions_cholesky_.copy()
            new_g.converged_ = g.converged_
            new_g.n_iter_ = g.n_iter_
            new_g.lower_bound_ = g.lower_bound_

            S = new_g.sample(n_samples=n_samples)[0]
            out.append(S)
        out = np.stack(out, axis=0)  # (n, n_samples, Dy)
        return out

    def condition(self, X):
        """
        Condition the mixture on X and return a GaussianMixture over y.

        This method delegates to the internal conditioner's condition() method,
        providing a consistent interface across all conditional mixture models.
        """
        check_is_fitted(self, attributes=["_cond"])
        X = validate_data(self, X, reset=False)
        return self._cond.condition(X)
