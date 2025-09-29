# cgmm/discriminative.py
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
    diff = x - mean
    return float(
        -0.5 * (np.sum(np.log(2.0 * np.pi * diag_var)) + np.sum(diff * diff / diag_var))
    )


class DiscriminativeConditionalGMMRegressor(BaseConditionalMixture, ConditionalMixin):
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
        weight_step: float | None = 0.01,  # per-sample stepsize
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
        self.weight_step = weight_step
        self._cache: dict = {}

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
        self.n_features_in_ = Dx
        self.n_targets_ = Dy

        # Init by joint GMM on Z=[X|Y]
        Z = np.concatenate([X, y], axis=1)
        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            tol=self.tol,
            reg_covar=max(self.reg_covar, 1e-6),
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
        else:
            self.covariances_ = np.maximum(self.covariances_, self.reg_covar)

        # Initialize logits for weights
        self._alpha_ = np.log(self.weights_ + np.finfo(float).tiny)

        # Build cache for first E-step
        self._refresh_cholesky_cache(Dx)

        # Run discriminative EM with parameter-delta stopping
        self.converged_ = False
        for it in range(1, self.max_iter + 1):
            prev_means = self.means_.copy()
            prev_covs = self.covariances_.copy()
            prev_weights = self.weights_.copy()

            r, s = self._e_step(X, y)  # (n,K)
            self._m_step_joint(X, y, r, Z=Z)  # update μ, Σ
            self._refresh_cholesky_cache(Dx)  # cache factors for next E-step
            self._m_step_weights(r, s)  # update π via α-gradient

            # Stop on parameter deltas
            param_delta = max(
                float(np.abs(self.means_ - prev_means).max()),
                float(np.abs(self.covariances_ - prev_covs).max()),
                float(np.abs(self.weights_ - prev_weights).max()),
            )
            if param_delta < self.tol:
                self.converged_ = True
                break

        self.lower_bound_ = float(self._mean_conditional_loglik(X, y))
        self.n_iter_ = int(it)
        return self

    # ------------ cache of Cholesky/log-dets (rebuilt once per iteration) ------------
    def _refresh_cholesky_cache(self, Dx: int) -> None:
        """Cache Cholesky factors and log-determinants for joint Σ and X-block Σ_xx."""
        K = self.n_components
        if self.covariance_type == "full":
            D = self.means_.shape[1]
            joint_L = np.empty((K, D, D))
            joint_logdet = np.empty(K)
            L_xx = np.empty((K, Dx, Dx))
            logdet_xx = np.empty(K)
            for k in range(K):
                S = self.covariances_[k]
                L = np.linalg.cholesky(S)
                joint_L[k] = L
                joint_logdet[k] = 2.0 * np.log(np.diag(L)).sum()
                S_xx = S[:Dx, :Dx]
                Lx = np.linalg.cholesky(S_xx)
                L_xx[k] = Lx
                logdet_xx[k] = 2.0 * np.log(np.diag(Lx)).sum()
            self._cache = {
                "joint_L": joint_L,  # (K,D,D)
                "joint_logdet": joint_logdet,  # (K,)
                "L_xx": L_xx,  # (K,Dx,Dx)
                "logdet_xx": logdet_xx,  # (K,)
            }
        else:  # diag
            v = self.covariances_  # (K, D)
            self._cache = {
                "inv_v_joint": 1.0 / v,  # (K,D)
                "joint_logdet": np.log(v).sum(axis=1),  # (K,)
                "inv_v_xx": 1.0 / v[:, :Dx],  # (K,Dx)
                "logdet_xx": np.log(v[:, :Dx]).sum(axis=1),  # (K,)
            }

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
            S = np.empty((K, Dy))  # diag variance vectors

        for k in range(K):
            mu_x, S_xx, mu_y, S_yy = self._slice_blocks(k, Dx)
            if self.covariance_type == "full":
                # Cross blocks via Cholesky solves (no explicit inverse)
                S_xy = self.covariances_[k, :Dx, Dx:]  # (Dx,Dy)
                S_yx = S_xy.T  # (Dy,Dx)
                S_xx_pd = self._ensure_positive_definite(S_xx)
                L = np.linalg.cholesky(S_xx_pd)

                # Σ_{y|x} = Σ_yy − Σ_yx Σ_xx^{-1} Σ_xy
                Z = np.linalg.solve(L, S_xy)
                A = np.linalg.solve(L.T, Z)  # Σ_xx^{-1} Σ_xy
                S_k = S_yy - S_yx @ A
                S_k = self._ensure_positive_definite(S_k)
                S[k] = S_k

                # μ_{y|x} = μ_y + Σ_yx Σ_xx^{-1} (x − μ_x)
                T = np.linalg.solve(L, S_yx.T)
                B = np.linalg.solve(L.T, T).T  # Σ_yx Σ_xx^{-1}  (Dy×Dx)
                diff = X - mu_x[None, :]
                M[:, k, :] = mu_y[None, :] + diff @ B.T
            else:
                S[k] = S_yy
                M[:, k, :] = mu_y[None, :]

        return {"weights": W, "means": M, "covariances": S}

    def responsibilities(self, X: ArrayLike, y: ArrayLike | None = None) -> np.ndarray:
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
        """Vectorized E-step using cached Cholesky/diag; returns (r, s) with shape (n,K)."""
        n, Dx = X.shape
        # Dy = self.n_targets_
        K = self.n_components
        tiny = np.finfo(float).tiny
        log2pi = np.log(2.0 * np.pi)
        log_pi = np.log(self.weights_ + tiny)  # (K,)

        # Prepare diffs
        Z = np.concatenate([X, y], axis=1)  # (n, D)
        mu = self.means_  # (K, D)
        D = mu.shape[1]
        diffZ = Z[:, None, :] - mu[None, :, :]  # (n, K, D)

        # JOINT: log π_k + log N([x,y] | μ_k, Σ_k)
        if self.covariance_type == "full":
            joint_L = self._cache["joint_L"]  # (K,D,D)
            joint_logdet = self._cache["joint_logdet"]  # (K,)
            log_joint = np.empty((n, K))
            for k in range(K):
                L = joint_L[k]
                u = np.linalg.solve(L, diffZ[:, k, :].T).T  # (n,D)
                md2 = np.einsum("ij,ij->i", u, u)  # (n,)
                log_joint[:, k] = log_pi[k] - 0.5 * (joint_logdet[k] + md2 + D * log2pi)
        else:
            inv_v = self._cache["inv_v_joint"]  # (K,D)
            joint_logdet = self._cache["joint_logdet"]  # (K,)
            md2 = np.einsum("nkd,kd->nk", diffZ * diffZ, inv_v)  # (n,K)
            log_joint = log_pi[None, :] - 0.5 * (
                joint_logdet[None, :] + md2 + D * log2pi
            )
        r = _softmax_rows(log_joint)

        # X-ONLY: log π_k + log N(x | μ_xk, Σ_xxk)
        mu_x = mu[:, :Dx]  # (K,Dx)
        diffX = X[:, None, :] - mu_x[None, :, :]  # (n,K,Dx)
        if self.covariance_type == "full":
            L_xx = self._cache["L_xx"]  # (K,Dx,Dx)
            logdet_xx = self._cache["logdet_xx"]  # (K,)
            log_x = np.empty((n, K))
            for k in range(K):
                Lx = L_xx[k]
                ux = np.linalg.solve(Lx, diffX[:, k, :].T).T
                md2x = np.einsum("ij,ij->i", ux, ux)
                log_x[:, k] = log_pi[k] - 0.5 * (logdet_xx[k] + md2x + Dx * log2pi)
        else:
            inv_v_xx = self._cache["inv_v_xx"]  # (K,Dx)
            logdet_xx = self._cache["logdet_xx"]  # (K,)
            md2x = np.einsum("nkd,kd->nk", diffX * diffX, inv_v_xx)
            log_x = log_pi[None, :] - 0.5 * (logdet_xx[None, :] + md2x + Dx * log2pi)
        s = _softmax_rows(log_x)

        return r, s

    def _m_step_joint(
        self, X: np.ndarray, y: np.ndarray, r: np.ndarray, Z: np.ndarray | None = None
    ) -> None:
        """Update joint means and covariances from r (standard EM for a joint GMM)."""
        n, Dx = X.shape
        Dy = self.n_targets_
        D = Dx + Dy
        K = self.n_components

        if Z is None:
            Z = np.concatenate([X, y], axis=1)  # (n,D)

        # Effective component weights
        Nk = r.sum(axis=0) + 1e-12  # (K,)

        # Means
        self.means_ = (r.T @ Z) / Nk[:, None]  # (K,D)

        # Covariances
        if self.covariance_type == "full":
            for k in range(K):
                Zc = Z - self.means_[k][None, :]
                Ck = (Zc.T * r[:, k]) @ Zc / Nk[k]
                self.covariances_[k] = self._ensure_positive_definite(Ck)
        else:
            cov = np.empty((K, D))
            for k in range(K):
                Zc = Z - self.means_[k][None, :]
                var = (r[:, k][:, None] * (Zc * Zc)).sum(axis=0) / Nk[k]
                cov[k] = np.maximum(var, self.reg_covar)
            self.covariances_ = cov

    def _m_step_weights(self, r: np.ndarray, s: np.ndarray) -> None:
        """Update mixture weights via logits α; gradient is averaged over samples."""
        grad = (r.sum(axis=0) - s.sum(axis=0)) / max(r.shape[0], 1.0)  # (K,)
        eta = self.weight_step if self.weight_step is not None else 0.01
        self._alpha_ += eta * grad
        self._alpha_ -= self._alpha_.max()  # stabilize
        w = np.exp(self._alpha_)
        self.weights_ = (w / w.sum()).astype(float)

    def _slice_blocks(
        self, k: int, Dx: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mu = self.means_[k]
        mu_x = mu[:Dx]
        mu_y = mu[Dx:]
        if self.covariance_type == "full":
            S = self.covariances_[k]
            S_xx = S[:Dx, :Dx]
            S_yy = S[Dx:, Dx:]
            return mu_x, S_xx, mu_y, S_yy
        else:
            v = self.covariances_[k]  # (D,)
            return mu_x, v[:Dx], mu_y, v[Dx:]

    # ---------- direct conditional log p(y|x) ----------
    def log_prob(self, X: ArrayLike, y: ArrayLike) -> np.ndarray:
        check_is_fitted(self, attributes=["weights_", "means_", "covariances_"])
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1 and self.n_targets_ == 1:
            y = y.reshape(-1, 1)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X = validate_data(self, X, reset=False)

        params = self._compute_conditional_mixture(X)
        W, M, S = params["weights"], params["means"], params["covariances"]
        n, K = W.shape
        tiny = np.finfo(float).tiny

        log_comp = np.empty((n, K))
        if self.covariance_type == "full":
            for k in range(K):
                Sk = S[k]
                for i in range(n):
                    log_comp[i, k] = np.log(W[i, k] + tiny) + _log_gaussian_full(
                        y[i], M[i, k], Sk
                    )
        else:
            for k in range(K):
                var = S[k]
                for i in range(n):
                    log_comp[i, k] = np.log(W[i, k] + tiny) + _log_gaussian_diag(
                        y[i], M[i, k], var
                    )

        return logsumexp(log_comp, axis=1)

    def _mean_conditional_loglik(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.log_prob(X, y)))

    def _ensure_positive_definite(self, C: np.ndarray) -> np.ndarray:
        C = np.array(C, dtype=float, copy=True)
        C = (C + C.T) * 0.5
        try:
            cholesky(C, lower=True, check_finite=False)
            C.flat[:: C.shape[0] + 1] += self.reg_covar
            return C
        except Exception:
            pass
        vals, vecs = np.linalg.eigh(C)
        vals = np.maximum(vals, 0.0) + self.reg_covar
        return (vecs * vals) @ vecs.T

    def condition(self, X: ArrayLike) -> Union[GaussianMixture, List[GaussianMixture]]:
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
                gmm.covariances_ = S
                prec_chol = []
                for k in range(self.n_components):
                    prec = np.linalg.inv(S[k])
                    L = np.linalg.cholesky(prec)
                    prec_chol.append(L)
                gmm.precisions_cholesky_ = np.stack(prec_chol, axis=0)
            else:
                gmm.covariances_ = S
                gmm.precisions_cholesky_ = 1.0 / np.sqrt(S)
            gmm.converged_ = True
            gmm.n_iter_ = 0
            return gmm

        if X.shape[0] == 1:
            return _mk(0)
        return [_mk(i) for i in range(X.shape[0])]

    def sample(self, X: ArrayLike, n_samples: int = 1):
        check_is_fitted(self, attributes=["weights_", "means_", "covariances_"])
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X = validate_data(self, X, reset=False)

        params = self._compute_conditional_mixture(X)
        W, M, S = params["weights"], params["means"], params["covariances"]

        # Use truly random seed for independence (scikit-learn compatible)
        rng = np.random.RandomState()
        samples = []
        for i in range(X.shape[0]):
            idx = rng.choice(self.n_components, size=n_samples, p=W[i])
            comp_samples = []
            for k in idx:
                cov_matrix = S[k] if self.covariance_type == "full" else np.diag(S[k])
                comp_samples.append(rng.multivariate_normal(M[i, k], cov_matrix))
            samples.append(np.array(comp_samples))
        samples = np.array(samples)

        if X.shape[0] == 1:
            samples = samples[0]
        return samples


# ------------------------------- small helpers -------------------------------


def _softmax_rows(logM: np.ndarray) -> np.ndarray:
    M = logM - logM.max(axis=1, keepdims=True)
    np.exp(M, out=M)
    M /= M.sum(axis=1, keepdims=True)
    return M
