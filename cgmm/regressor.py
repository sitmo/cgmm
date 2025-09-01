from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.mixture import GaussianMixture

from .conditioner import GMMConditioner


class ConditionalGMMRegressor(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        n_components=2,
        covariance_type="full",
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=200,
        random_state=None,
        init_params="kmeans",
        cond_idx=None,
        return_cov=False,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.random_state = random_state
        self.init_params = init_params
        self.cond_idx = cond_idx
        self.return_cov = return_cov

    # Sklearn 1.6+: declare tag customizations here (no _more_tags).
    def __sklearn_tags__(self):
        # Get the default EstimatorTags object from BaseEstimator
        tags = super().__sklearn_tags__()
        # Declare that we support multi-output (2D) targets; this disables the
        # DataConversionWarning expectation in sklearn’s checks.
        tags.target_tags.multi_output = True
        return tags

    def fit(self, X, y):
        # sets n_features_in_
        X, y = validate_data(
            self,
            X,
            y,
            accept_sparse=False,
            y_numeric=True,
            multi_output=True,  # allow 2D Y
        )

        y = np.asarray(y)
        y2 = y[:, None] if y.ndim == 1 else y
        Z = np.concatenate([X, y2], axis=1)

        self.dx_ = X.shape[1]
        self.dy_ = y2.shape[1]

        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            tol=self.tol,
            reg_covar=self.reg_covar,
            max_iter=self.max_iter,
            random_state=self.random_state,
            init_params=self.init_params,
        ).fit(Z)

        # Expose iteration count for sklearn's estimator checks
        self.n_iter_ = int(getattr(gmm, "n_iter_", 1)) or 1

        cond_idx = (
            np.arange(self.dx_) if self.cond_idx is None else tuple(self.cond_idx)
        )
        self._cond = GMMConditioner(
            mixture_estimator=gmm,
            cond_idx=cond_idx,
            reg_covar=self.reg_covar,
        ).precompute()
        return self

    @staticmethod
    def _mixture_moments_from_gmm(gm: GaussianMixture):
        """Return (mean, cov) from a fitted GaussianMixture over y."""
        w = gm.weights_  # (K,)
        M = gm.means_  # (K, dy)
        S = gm.covariances_  # (K, dy, dy)
        mu = (w[:, None] * M).sum(axis=0)  # (dy,)
        E_S = (w[:, None, None] * S).sum(axis=0)  # (dy,dy)
        E_MM = (w[:, None, None] * np.einsum("ki,kj->kij", M, M)).sum(axis=0)
        cov = E_S + E_MM - np.outer(mu, mu)  # (dy,dy)
        return mu, cov

    def predict(self, X):
        from sklearn.mixture import GaussianMixture

        check_is_fitted(self, "_cond")
        X = validate_data(self, X, reset=False)

        gm = self._cond.condition(X)  # GaussianMixture or list[GaussianMixture]

        # --- Single mixture path (happens only if condition() got a 1D x) ---
        if isinstance(gm, GaussianMixture):
            mu, cov = self._mixture_moments_from_gmm(gm)
            if self.dy_ == 1:
                # Return shape (1,) or ((1,), (1,))
                return (
                    np.array([mu[0]])
                    if not self.return_cov
                    else (np.array([mu[0]]), np.array([cov[0, 0]]))
                )
            # dy > 1: (1, dy) and (1, dy, dy) if cov requested
            return (mu[None, :], cov[None, :, :]) if self.return_cov else mu[None, :]

        # --- Batch path: list[GaussianMixture] of length n_samples ---
        n = len(gm)
        dy = self.dy_
        means = np.empty((n, dy))
        covs = np.empty((n, dy, dy)) if self.return_cov else None
        for i, g in enumerate(gm):
            m_i, C_i = self._mixture_moments_from_gmm(g)
            means[i] = m_i
            if self.return_cov:
                covs[i] = C_i

        if dy == 1:
            return means[:, 0] if not self.return_cov else (means[:, 0], covs[:, 0, 0])
        return (means, covs) if self.return_cov else means

    def sample(self, X, n_samples=1, random_state=None):
        check_is_fitted(self, "_cond")
        X = validate_data(self, X, reset=False)

        gm = self._cond.condition(X)
        if isinstance(gm, GaussianMixture):
            S = gm.sample(n_samples=n_samples)[0]  # (n_samples, dy)
            # keep (1, n_samples, dy) for dy>1; and (n_samples,) for dy==1
            return S[:, 0] if self.dy_ == 1 else S[None, ...]

        # batch case: list of gmms
        out = []
        for g in gm:
            S = g.sample(n_samples=n_samples)[0]  # (n_samples, dy)
            out.append(S)
        out = np.stack(out, axis=0)  # (n, n_samples, dy)
        return out[:, :, 0] if self.dy_ == 1 else out
