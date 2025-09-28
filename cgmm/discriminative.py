# cgmm/discriminative.py
# Discriminative Conditional GMM Regressor based on Salojärvi–Puolamäki–Kaski (ICML 2005)
from __future__ import annotations

from typing import Union, List
import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.mixture import GaussianMixture
from scipy.linalg import cholesky
from scipy.special import logsumexp

from .base import BaseConditionalMixture, ConditionalMixin, _log_gaussian_full


def _log_gaussian_diag(x: np.ndarray, mean: np.ndarray, diag_var: np.ndarray) -> float:
    """log N(x|mean, diag(diag_var)) with diag_var as a 1D variance vector."""
    diff = x - mean
    return float(
        -0.5 * (np.sum(np.log(2.0 * np.pi * diag_var)) + np.sum(diff * diff / diag_var))
    )


class DiscriminativeConditionalGMMRegressor(BaseConditionalMixture, ConditionalMixin):
    """
    Discriminative Conditional GMM Regressor based on Salojärvi–Puolamäki–Kaski (ICML 2005).

    Optimizes the conditional likelihood:
        L_cond = Σ_i log p(y_i | x_i) =
                 Σ_i log [Σ_k π_k N([x_i, y_i] | μ_k, Σ_k)] - log [Σ_k π_k N(x_i | μ_{X,k}, Σ_{XX,k})]

    We use a surrogate with two posteriors per E-step:
        r_{ik} ∝ π_k N([x_i,y_i]|μ_k,Σ_k)     (joint)
        s_{ik} ∝ π_k N(x_i|μ_{X,k},Σ_{XX,k})  (X-only)

    M-step:
      • Update joint (μ_k, Σ_k) using r_{ik} exactly like EM for a joint GMM.
      • Update mixture weights via logits α using gradient ∂/∂α_k = Σ_i (r_{ik} - s_{ik}).

    Notes:
      - Only one parameter store: weights_, means_, covariances_ for the joint over [X,Y].
        Any X/Y blocks are always sliced on-the-fly to avoid inconsistencies.
    """

    def __init__(
        self,
        n_components: int = 2,
        covariance_type: str = "full",  # 'full' or 'diag'
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        max_iter: int = 200,
        random_state=None,
        init_params: str = "kmeans",
        *,
        return_cov: bool = False,
        weight_step: (
            float | None
        ) = 0.01,  # step size for weight updates (default: 0.01)
    ):
        super().__init__(return_cov=return_cov)
        self.n_components = int(n_components)
        if self.n_components < 1:
            raise ValueError("n_components must be positive")
        if covariance_type not in {"full", "diag"}:
            raise ValueError("covariance_type must be 'full' or 'diag'")
        self.covariance_type = covariance_type
        self.tol = float(tol)
        self.reg_covar = float(reg_covar)
        self.max_iter = int(max_iter)
        self.random_state = random_state
        self.init_params = init_params
        self.weight_step = weight_step  # if None, use ~2/n each step

    # ------------------------------- public API -------------------------------

    def fit(self, X: ArrayLike, y: ArrayLike):
        # Validate and shape data
        X, y = validate_data(
            self,
            X,
            y,
            accept_sparse=False,
            y_numeric=True,
            multi_output=True,
        )
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]

        n, Dx = X.shape
        Dy = y.shape[1]
        # D = Dx + Dy
        self.n_features_in_ = Dx
        self.n_targets_ = Dy

        # Init by joint GMM on Z=[X|Y]
        Z = np.concatenate([X, y], axis=1)
        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            tol=self.tol,
            reg_covar=max(self.reg_covar, 1e-6),  # small but safe init floor
            max_iter=self.max_iter,
            random_state=self.random_state,
            init_params=self.init_params,
        ).fit(Z)

        # Store joint params
        self.weights_ = gmm.weights_.astype(float, copy=True)  # (K,)
        self.means_ = gmm.means_.astype(float, copy=True)  # (K, D)
        self.covariances_ = gmm.covariances_.astype(
            float, copy=True
        )  # (K, D, D) or (K, D)

        # Ensure PD / positivity
        if self.covariance_type == "full":
            for k in range(self.n_components):
                self.covariances_[k] = self._ensure_positive_definite(
                    self.covariances_[k]
                )
        else:  # "diag" as vector of variances
            self.covariances_ = np.maximum(self.covariances_, self.reg_covar)

        # Initialize logits for weights
        self._alpha_ = np.log(self.weights_ + np.finfo(float).tiny)  # softmax logits

        # Run discriminative EM with parameter-delta stopping
        # prev_ll = -np.inf
        self.converged_ = False
        for it in range(1, self.max_iter + 1):
            prev_means = self.means_.copy()
            prev_covs = self.covariances_.copy()
            prev_weights = self.weights_.copy()

            r, s = self._e_step(X, y)  # (n,K) each
            self._m_step_joint(X, y, r)  # update μ, Σ using r
            self._m_step_weights(r, s)  # update π via α-gradient

            # Stopping on parameter deltas (stable for our surrogate)
            param_delta = max(
                float(np.abs(self.means_ - prev_means).max()),
                float(np.abs(self.covariances_ - prev_covs).max()),
                float(np.abs(self.weights_ - prev_weights).max()),
            )
            if param_delta < self.tol:
                self.converged_ = True
                break

            # Track conditional ll for reporting/debug
            # ll = self._mean_conditional_loglik(X, y)
            # prev_ll = ll

        self.lower_bound_ = float(self._mean_conditional_loglik(X, y))
        self.n_iter_ = int(it)
        return self

    def _compute_conditional_mixture(self, X: ArrayLike) -> dict:
        """Return mixture parameters of p(y|X) by conditioning the joint GMM."""
        check_is_fitted(self, attributes=["weights_", "means_", "covariances_"])
        X = validate_data(self, X, reset=False)

        n = X.shape[0]
        Dx = self.n_features_in_
        Dy = self.n_targets_
        K = self.n_components

        # Gating weights: proportional to π_k N(x | μ_xk, Σ_xxk)
        log_gate = np.empty((n, K))
        tiny = np.finfo(float).tiny
        for k in range(K):
            mu_x, S_xx, _, _ = self._slice_blocks(k, Dx)
            log_pi = np.log(self.weights_[k] + tiny)
            if self.covariance_type == "full":
                # per-sample log N(x|...)
                for i in range(n):
                    log_gate[i, k] = log_pi + _log_gaussian_full(X[i], mu_x, S_xx)
            else:
                for i in range(n):
                    log_gate[i, k] = log_pi + _log_gaussian_diag(X[i], mu_x, S_xx)

        W = _softmax_rows(log_gate)  # (n,K)

        # Conditional means/covariances per component
        M = np.empty((n, K, Dy))
        if self.covariance_type == "full":
            S = np.empty((K, Dy, Dy))
        else:
            S = np.empty((K, Dy))  # store diag as variance vector

        for k in range(K):
            mu_x, S_xx, mu_y, S_yy = self._slice_blocks(k, Dx)
            if self.covariance_type == "full":
                if False:
                    S_xy = self.covariances_[k, :Dx, Dx:]  # (Dx, Dy)
                    S_yx = S_xy.T
                    # Σ_{y|x} = S_yy - S_yx S_xx^{-1} S_xy
                    S_xx_inv = np.linalg.inv(S_xx)
                    S_k = S_yy - S_yx @ S_xx_inv @ S_xy
                    S_k = self._ensure_positive_definite(S_k)
                    S[k] = S_k
                    # μ_{y|x} = μ_y + S_yx S_xx^{-1} (x - μ_x)
                    A = S_yx @ S_xx_inv
                    diff = X - mu_x[None, :]
                    M[:, k, :] = mu_y[None, :] + diff @ A.T
                else:
                    # Cross blocks
                    S_xy = self.covariances_[k, :Dx, Dx:]  # (Dx, Dy)
                    S_yx = S_xy.T  # (Dy, Dx)

                    # Ensure Σ_xx is PD (adds tiny diag if needed)
                    S_xx_pd = self._ensure_positive_definite(S_xx)

                    # Cholesky of Σ_xx
                    L = np.linalg.cholesky(S_xx_pd)  # S_xx = L L^T

                    # -------- conditional covariance --------
                    # Compute Σ_xx^{-1} Σ_xy via two triangular solves
                    # Solve L Z = Σ_xy  -> Z
                    Z = np.linalg.solve(L, S_xy)
                    # Solve L^T A = Z   -> A = Σ_xx^{-1} Σ_xy   (shape Dx×Dy)
                    A = np.linalg.solve(L.T, Z)

                    # Σ_{y|x} = Σ_yy − Σ_yx (Σ_xx^{-1} Σ_xy)
                    S_k = S_yy - S_yx @ A
                    S_k = self._ensure_positive_definite(S_k)
                    S[k] = S_k

                    # -------- conditional mean --------
                    # Need (Σ_yx Σ_xx^{-1}) (x − μ_x)
                    # Compute B = Σ_yx Σ_xx^{-1} via solves on the transpose:
                    # Solve L T = Σ_yx^T    -> T
                    T = np.linalg.solve(L, S_yx.T)
                    # Solve L^T B_T = T     -> B_T = Σ_xx^{-1} Σ_yx^T
                    B = np.linalg.solve(L.T, T).T  # B = Σ_yx Σ_xx^{-1}  (shape Dy×Dx)

                    diff = X - mu_x[None, :]  # (n×Dx)
                    M[:, k, :] = mu_y[None, :] + diff @ B.T

            else:
                # diag: no cross-cov; conditional mean is μ_y; cov = S_yy (as variance vector)
                S[k] = S_yy
                M[:, k, :] = mu_y[None, :]

        return {"weights": W, "means": M, "covariances": S}

    def responsibilities(self, X: ArrayLike, y: ArrayLike | None = None) -> np.ndarray:
        """If y is None: return gating weights; else: return joint posteriors r_ik."""
        if y is None:
            return self._compute_conditional_mixture(X)["weights"]

        y = np.asarray(y, dtype=float)
        if y.ndim == 1 and self.n_targets_ == 1:
            y = y[:, None]
        X = validate_data(self, X, reset=False)

        n, Dx = X.shape
        K = self.n_components
        tiny = np.finfo(float).tiny

        log_r = np.empty((n, K))
        for k in range(K):
            mu = self.means_[k]
            if self.covariance_type == "full":
                S = self.covariances_[k]
            else:
                S = np.diag(self.covariances_[k])
            for i in range(n):
                xy = np.concatenate([X[i], y[i]])
                log_r[i, k] = np.log(self.weights_[k] + tiny) + _log_gaussian_full(
                    xy, mu, S
                )
        return _softmax_rows(log_r)

    # ------------------------------- internals -------------------------------

    def _e_step(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute r_ik (joint) and s_ik (X-only) with current joint params."""
        n, Dx = X.shape
        K = self.n_components
        tiny = np.finfo(float).tiny

        # r_ik ∝ π_k N([x,y] | μ_k, Σ_k)
        log_r = np.empty((n, K))
        for k in range(K):
            mu = self.means_[k]
            if self.covariance_type == "full":
                S = self.covariances_[k]
            else:
                S = np.diag(self.covariances_[k])
            for i in range(n):
                xy = np.concatenate([X[i], y[i]])
                log_r[i, k] = np.log(self.weights_[k] + tiny) + _log_gaussian_full(
                    xy, mu, S
                )
        r = _softmax_rows(log_r)

        # s_ik ∝ π_k N(x | μ_xk, Σ_xxk)
        log_s = np.empty((n, K))
        for k in range(K):
            mu_x, S_xx, _, _ = self._slice_blocks(k, Dx)
            log_pi = np.log(self.weights_[k] + tiny)
            if self.covariance_type == "full":
                for i in range(n):
                    log_s[i, k] = log_pi + _log_gaussian_full(X[i], mu_x, S_xx)
            else:
                for i in range(n):
                    log_s[i, k] = log_pi + _log_gaussian_diag(X[i], mu_x, S_xx)
        s = _softmax_rows(log_s)

        return r, s

    def _m_step_joint(self, X: np.ndarray, y: np.ndarray, r: np.ndarray) -> None:
        """Update joint means and covariances from r (standard EM for a joint GMM)."""
        n, Dx = X.shape
        Dy = self.n_targets_
        D = Dx + Dy
        K = self.n_components

        Z = np.concatenate([X, y], axis=1)  # (n,D)

        # Effective component weights
        Nk = r.sum(axis=0) + 1e-12  # (K,)

        # Means
        self.means_ = (r.T @ Z) / Nk[:, None]  # (K,D)

        # Covariances
        if self.covariance_type == "full":
            for k in range(K):
                Zc = Z - self.means_[k][None, :]
                # weighted covariance
                Ck = (Zc.T * r[:, k]) @ Zc / Nk[k]
                # no extra add here; enforce PD + reg in helper
                self.covariances_[k] = self._ensure_positive_definite(Ck)
        else:
            # Diagonal: store as variance vector of length D
            cov = np.empty((K, D))
            for k in range(K):
                Zc = Z - self.means_[k][None, :]
                var = (r[:, k][:, None] * (Zc * Zc)).sum(axis=0) / Nk[k]
                cov[k] = np.maximum(var, self.reg_covar)
            self.covariances_ = cov

    def _m_step_weights(self, r: np.ndarray, s: np.ndarray) -> None:
        """Update mixture weights via logits α with gradient ∝ sum_i (r_ik - s_ik)."""
        grad = r.sum(axis=0) - s.sum(axis=0)  # (K,)
        n = r.shape[0]
        # Normalize gradient by dataset size for numerical stability
        grad = grad / max(n, 1.0)
        eta = self.weight_step if self.weight_step is not None else 0.01
        self._alpha_ += eta * grad
        # stabilize and project via softmax
        self._alpha_ -= self._alpha_.max()
        w = np.exp(self._alpha_)
        self.weights_ = (w / w.sum()).astype(float)

    def _slice_blocks(
        self, k: int, Dx: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (μ_x, Σ_xx, μ_y, Σ_yy) for component k sliced from the joint."""
        mu = self.means_[k]
        mu_x = mu[:Dx]
        mu_y = mu[Dx:]
        if self.covariance_type == "full":
            S = self.covariances_[k]
            S_xx = S[:Dx, :Dx]
            S_yy = S[Dx:, Dx:]
            return mu_x, S_xx, mu_y, S_yy
        else:
            # diag stored as (D,) variances
            v = self.covariances_[k]
            S_xx = v[:Dx]
            S_yy = v[Dx:]
            return mu_x, S_xx, mu_y, S_yy

    # ---------- fixed, direct conditional log p(y|x) ----------
    def log_prob(self, X: ArrayLike, y: ArrayLike) -> np.ndarray:
        """
        Pointwise conditional log-likelihoods log p(y|x), computed directly
        from conditioned mixture parameters (no sklearn dependency).
        """
        check_is_fitted(self, attributes=["weights_", "means_", "covariances_"])
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1 and self.n_targets_ == 1:
            y = y[:, None]
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X = validate_data(self, X, reset=False)

        params = self._compute_conditional_mixture(X)
        W, M, S = params["weights"], params["means"], params["covariances"]
        n, K = W.shape
        # Dy = y.shape[1]
        tiny = np.finfo(float).tiny

        log_comp = np.empty((n, K))
        if self.covariance_type == "full":
            for k in range(K):
                Sk = S[k]
                for i in range(n):
                    log_comp[i, k] = np.log(W[i, k] + tiny) + _log_gaussian_full(
                        y[i], M[i, k], Sk
                    )
        else:  # diag: S has shape (K, Dy) variance vectors
            for k in range(K):
                var = S[k]
                for i in range(n):
                    log_comp[i, k] = np.log(W[i, k] + tiny) + _log_gaussian_diag(
                        y[i], M[i, k], var
                    )

        return logsumexp(log_comp, axis=1)

    def _mean_conditional_loglik(self, X: np.ndarray, y: np.ndarray) -> float:
        """Mean conditional log-likelihood log p(y|x)."""
        return float(np.mean(self.log_prob(X, y)))

    def _ensure_positive_definite(self, C: np.ndarray) -> np.ndarray:
        """Make a covariance matrix PD with diagonal inflation if needed."""
        C = np.array(C, dtype=float, copy=True)
        # symmetric
        C = (C + C.T) * 0.5
        # quick path
        try:
            cholesky(C, lower=True, check_finite=False)
            # add minimal reg to be safe
            C.flat[:: C.shape[0] + 1] += self.reg_covar
            return C
        except Exception:
            pass
        # eigen inflate
        vals, vecs = np.linalg.eigh(C)
        vals = np.maximum(vals, 0.0)
        vals += self.reg_covar
        C_pd = (vecs * vals) @ vecs.T
        return C_pd

    def condition(self, X: ArrayLike) -> Union[GaussianMixture, List[GaussianMixture]]:
        """
        Condition the mixture on X and return a GaussianMixture over y.

        Note: sets required sklearn fields correctly for 'full' and 'diag'.
        """
        check_is_fitted(self, attributes=["weights_", "means_", "covariances_"])
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X = validate_data(self, X, reset=False)

        params = self._compute_conditional_mixture(X)
        W, M, S = params["weights"], params["means"], params["covariances"]

        def _mk(i: int) -> GaussianMixture:
            gmm = GaussianMixture(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
            )
            gmm.weights_ = W[i]
            gmm.means_ = M[i]
            if self.covariance_type == "full":
                gmm.covariances_ = S  # (K, Dy, Dy)
                # precisions cholesky (lower)
                prec_chol = []
                for k in range(self.n_components):
                    prec = np.linalg.inv(S[k])
                    L = np.linalg.cholesky(prec)
                    prec_chol.append(L)
                gmm.precisions_cholesky_ = np.stack(prec_chol, axis=0)
            else:  # diag: S is (K, Dy) variance vectors
                gmm.covariances_ = S
                gmm.precisions_cholesky_ = 1.0 / np.sqrt(S)
            gmm.converged_ = True
            gmm.n_iter_ = 0
            return gmm

        if X.shape[0] == 1:
            return _mk(0)
        return [_mk(i) for i in range(X.shape[0])]

    def sample(self, X: ArrayLike, n_samples: int = 1, random_state=None):
        """
        Sample y|X.
        Returns:
          if X is (n,Dx): (n, n_samples, Dy)
          if X is a single sample (Dx,): (n_samples, Dy) or (n_samples,) if Dy==1
        """
        check_is_fitted(self, attributes=["weights_", "means_", "covariances_"])
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X = validate_data(self, X, reset=False)

        params = self._compute_conditional_mixture(X)
        W, M, S = params["weights"], params["means"], params["covariances"]

        rng = np.random.RandomState(random_state)

        samples = []
        for i in range(X.shape[0]):
            idx = rng.choice(self.n_components, size=n_samples, p=W[i])
            comp_samples = []
            for k in idx:
                if self.covariance_type == "full":
                    cov_matrix = S[k]
                else:  # diag: S[k] is variance vector
                    cov_matrix = np.diag(S[k])
                comp_samples.append(rng.multivariate_normal(M[i, k], cov_matrix))
            samples.append(np.array(comp_samples))

        samples = np.array(samples)  # (n_inputs, n_samples, Dy)

        if X.shape[0] == 1:
            samples = samples[0]  # (n_samples, Dy)
            if self.n_targets_ == 1:
                samples = samples[:, 0]  # (n_samples,)
        else:
            if self.n_targets_ == 1:
                samples = samples[:, :, 0]  # (n_inputs, n_samples)

        return samples


# ------------------------------- small helpers -------------------------------


def _softmax_rows(logM: np.ndarray) -> np.ndarray:
    """Row-wise softmax of log-scores."""
    M = logM - logM.max(axis=1, keepdims=True)
    np.exp(M, out=M)
    M /= M.sum(axis=1, keepdims=True)
    return M
