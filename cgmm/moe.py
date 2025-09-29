# cgmm/moe.py
# Mixture-of-Experts Regressor: p(y | X) = sum_k pi_k(X) N(y; A_k X + b_k, Sigma_k)
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Union, List

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import cholesky, solve_triangular
from scipy.special import logsumexp
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from .base import BaseConditionalMixture, ConditionalMixin, _log_gaussian_full

Array = np.ndarray


@dataclass
class _Params:
    A: Optional[Array]  # (K, Dy, Dx) or None if mean_function="constant"
    b: Array  # (K, Dy)
    cov: Array  # (K, Dy, Dy) or (1, Dy, Dy) if shared
    W: Array  # gating weights ((K-1), Dx) for baseline softmax
    c: Array  # gating intercepts ((K-1),)
    chol: Optional[Array] = None  # (K, Dy, Dy) cholesky of cov (kept for speed)


class MixtureOfExpertsRegressor(BaseConditionalMixture, ConditionalMixin):
    """
    Conditional Mixture-of-Experts (MoE) with linear-softmax gating and
    Gaussian experts (affine mean).

    Baseline softmax (identifiable):
      Let K be number of experts. We parameterize logits for classes 1..K-1;
      class K has z_K(x) = 0. Then
        pi_k(x) = softmax( z_1(x),...,z_{K-1}(x), 0 ).

    Model:
      z_k(x) = W_k x + c_k,  k=1..K-1;  z_K(x) = 0
      mu_k(x) = A_k x + b_k        (if mean_function='affine') or b_k (if 'constant')
      y | x ~ sum_k pi_k(x) N( mu_k(x), Sigma_k )
    """

    def __init__(
        self,
        n_components: int = 3,
        *,
        covariance_type: str = "full",
        shared_covariance: bool = False,
        mean_function: str = "affine",
        reg_covar: float = 1e-6,
        gating_penalty: float = 1e-2,
        gating_max_iter: int = 50,
        gating_penalty_bias: float | None = None,
        gating_tol: float = 1e-6,
        gating_init_scale: float = 1e-1,
        max_iter: int = 200,
        tol: float = 1e-4,
        n_init: int = 1,
        init_params: str = "kmeans",
        random_state=None,
        return_cov: bool = False,
        verbose: int = 0,
    ):
        super().__init__(return_cov=return_cov)
        # Simplified parameter validation - let sklearn handle most validation
        self.n_components = int(n_components)
        if self.n_components < 1:
            raise ValueError("n_components must be positive")
        self.covariance_type = covariance_type
        self.shared_covariance = bool(shared_covariance)
        self.mean_function = mean_function
        self.reg_covar = float(reg_covar)
        self.gating_penalty = float(gating_penalty)
        self.gating_max_iter = int(gating_max_iter)
        self.gating_tol = float(gating_tol)
        self.gating_init_scale = float(gating_init_scale)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.n_init = int(n_init)
        self.init_params = init_params
        self.random_state = random_state
        self.verbose = int(verbose)
        self.gating_penalty = float(gating_penalty)  # weights L2
        self.gating_penalty_bias = gating_penalty_bias  # if None -> use gating_penalty

    # ------------------------ public API ------------------------

    def fit(self, X: ArrayLike, y: ArrayLike):
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
        # K = self.n_components

        self.n_features_in_ = Dx
        self.n_targets_ = Dy

        best_ll = -np.inf
        best_params: Optional[_Params] = None
        best_iters = 0

        rng = np.random.RandomState(self.random_state)

        for init_try in range(self.n_init):
            params = self._init_params(X, y, rng)
            prev_lb = -np.inf

            for it in range(1, self.max_iter + 1):
                # E: responsibilities gamma (n, K), and expected stats
                gamma, ll = self._e_step(X, y, params)

                # M: update experts (A,b,Sigma) and gating (W,c)
                params = self._m_step(X, y, gamma, params)

                # Check EM convergence on mean log-likelihood
                rel_impr = (ll - prev_lb) / (1e-12 + abs(prev_lb))
                prev_lb = ll
                if self.verbose:
                    print(
                        f"[init {init_try+1}/{self.n_init}] iter {it}: ll={ll:.6f}, rel_impr={rel_impr:.3e}"
                    )
                if it > 1 and rel_impr < self.tol:
                    break

            if prev_lb > best_ll:
                best_ll = prev_lb
                best_params = params
                best_iters = it

        # Store best
        assert best_params is not None
        self._params = best_params
        self.lower_bound_ = float(best_ll)
        self.n_iter_ = int(best_iters)
        self.converged_ = True
        return self

    def _compute_conditional_mixture(self, X: ArrayLike) -> Dict[str, Array]:
        check_is_fitted(self, attributes=["_params", "n_features_in_", "n_targets_"])
        X = validate_data(self, X, reset=False)
        P = self._params
        K = self.n_components
        n = X.shape[0]
        Dy = self.n_targets_

        # Gating weights per sample (n, K)
        W = self._gating_softmax_baseline(X, P.W, P.c)

        # Expert means per sample
        if self.mean_function == "affine":
            # (n, K, Dy) = einsum over (K,Dy,Dx) @ (n,Dx)
            M = np.einsum("kij,nj->nki", P.A, X) + P.b[None, :, :]
        else:  # 'constant'
            M = np.broadcast_to(P.b[None, :, :], (n, K, Dy))

        # Covariances: independent of X, shape (K,Dy,Dy) (or (1,Dy,Dy) if shared)
        cov = P.cov if not self.shared_covariance else P.cov  # keep as stored
        if self.shared_covariance and cov.shape[0] == 1:
            cov = np.broadcast_to(cov, (K,) + cov.shape[1:])

        return {"weights": W, "means": M, "covariances": cov}

    def responsibilities(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> Array:
        params = self._compute_conditional_mixture(X)
        W, M, S = params["weights"], params["means"], params["covariances"]
        if y is None:
            return W
        y = np.asarray(y, dtype=float)
        if y.ndim == 1 and self.n_targets_ == 1:
            y = y[:, None]
        n, K = W.shape

        if S.ndim == 4:  # shouldn't happen here, but keep robust
            S = S[0]
        out = np.empty_like(W)
        for i in range(n):
            log_terms = np.empty(K)
            for k in range(K):
                log_terms[k] = np.log(W[i, k] + 1e-300) + (
                    _log_gaussian_full(y[i], M[i, k], S[k])
                    if self.covariance_type == "full"
                    else _log_gaussian_diag(y[i], M[i, k], S[k])
                )
            out[i] = _softmax_from_log(log_terms)
        return out

    def sample(self, X: ArrayLike, n_samples: int = 1):
        """
        Sample y|X.
        Returns:
          if X is (n,Dx): (n, n_samples, Dy)
          if X is a single sample (Dx,): (n_samples, Dy)
        """
        check_is_fitted(self, attributes=["_params"])
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X = validate_data(self, X, reset=False)
        # Use truly random seed for independence (scikit-learn compatible)
        rng = np.random.RandomState()

        params = self._compute_conditional_mixture(X)
        W, M, S = params["weights"], params["means"], params["covariances"]
        Dy = self.n_targets_

        if S.ndim == 4:
            S = S[0]

        def _diag_std(Sk: np.ndarray) -> np.ndarray:
            if Sk.ndim == 2:
                var = np.diag(Sk)
            else:
                var = Sk
            return np.sqrt(var)

        # single-sample
        if X.shape[0] == 1:
            i = 0
            K = W.shape[1]
            comp = rng.choice(K, size=n_samples, p=W[i])
            Y = np.empty((n_samples, Dy))
            for t in range(n_samples):
                k = comp[t]
                if self.covariance_type == "full":
                    L = cholesky(S[k], lower=True, check_finite=False)
                    noise = L @ rng.normal(size=Dy)
                else:
                    std = _diag_std(S[k])
                    noise = rng.normal(size=Dy) * std
                Y[t] = M[i, k] + noise
            return Y

        # batch
        n = X.shape[0]
        K = W.shape[1]
        Y = np.empty((n, n_samples, Dy))
        for i in range(n):
            comp = rng.choice(K, size=n_samples, p=W[i])
            for t in range(n_samples):
                k = comp[t]
                if self.covariance_type == "full":
                    L = cholesky(S[k], lower=True, check_finite=False)
                    noise = L @ rng.normal(size=Dy)
                else:
                    std = _diag_std(S[k])
                    noise = rng.normal(size=Dy) * std
                Y[i, t] = M[i, k] + noise
        return Y

    # ------------------------ EM internals ------------------------

    def _init_params(self, X: Array, y: Array, rng: np.random.RandomState) -> _Params:
        n, Dx = X.shape
        Dy = y.shape[1]
        K = self.n_components

        # Initialize means (b_k) and covariances by clustering y
        if self.init_params == "kmeans":
            km = KMeans(n_clusters=K, n_init=5, random_state=rng)
            labels = km.fit_predict(y)
            b = np.zeros((K, Dy))
            cov = (
                np.zeros((K, Dy, Dy))
                if not self.shared_covariance
                else np.zeros((1, Dy, Dy))
            )
            for k in range(K):
                idx = labels == k
                if np.any(idx):
                    b[k] = y[idx].mean(axis=0)
                    C = _emp_cov(
                        y[idx], reg=self.reg_covar, cov_type=self.covariance_type
                    )
                else:
                    b[k] = y.mean(axis=0) + 1e-2 * rng.normal(size=Dy)
                    C = _emp_cov(y, reg=self.reg_covar, cov_type=self.covariance_type)
                if self.shared_covariance:
                    cov[0] += C / K
                else:
                    cov[k] = C
        else:
            b = y.mean(axis=0)[None, :] + 0.1 * rng.normal(size=(K, Dy))
            baseC = _emp_cov(y, reg=self.reg_covar, cov_type=self.covariance_type)
            cov = (
                np.broadcast_to(baseC, (1, Dy, Dy))
                if self.shared_covariance
                else np.repeat(baseC[None, :, :], K, axis=0)
            )

        # Initialize A to zeros (affine experts start as constant)
        A = None if self.mean_function == "constant" else np.zeros((K, Dy, Dx))

        # Gating (baseline): param blocks for first K-1 classes only
        W = self.gating_init_scale * rng.normal(size=(K - 1, Dx))
        c = np.zeros(K - 1)

        chol = _chol_from_cov(cov, self.covariance_type)
        return _Params(A=A, b=b, cov=cov, W=W, c=c, chol=chol)

    def _e_step(self, X: Array, y: Array, P: _Params) -> Tuple[Array, float]:
        n, Dx = X.shape
        Dy = y.shape[1]
        K = self.n_components

        # pi(x): (n,K) via baseline softmax
        log_pi = self._gating_logits_baseline(X, P.W, P.c)
        log_pi = log_pi - logsumexp(log_pi, axis=1, keepdims=True)

        # mu_k(x): (n,K,Dy)
        if self.mean_function == "affine":
            M = np.einsum("kij,nj->nki", P.A, X) + P.b[None, :, :]
        else:
            M = np.broadcast_to(P.b[None, :, :], (n, K, Dy))

        # log-likelihood per component
        logN = np.empty((n, K))
        if self.covariance_type == "full":
            if P.chol is None:
                P.chol = _chol_from_cov(P.cov, self.covariance_type)
            for i in range(n):
                yi = y[i]
                for k in range(K):
                    L = P.chol[0] if self.shared_covariance else P.chol[k]
                    logN[i, k] = _log_gaussian_full(yi, M[i, k], L @ L.T)
        else:
            for i in range(n):
                yi = y[i]
                for k in range(K):
                    Sk = P.cov[0] if self.shared_covariance else P.cov[k]
                    logN[i, k] = _log_gaussian_diag(yi, M[i, k], Sk)

        # gamma ∝ exp(log_pi + logN)
        log_posts = log_pi + logN
        ll = float(np.mean(logsumexp(log_posts, axis=1)))
        gamma = np.exp(log_posts - logsumexp(log_posts, axis=1, keepdims=True))
        return gamma, ll

    def _m_step(self, X: Array, y: Array, gamma: Array, P: _Params) -> _Params:
        n, Dx = X.shape
        Dy = y.shape[1]
        K = self.n_components

        # ---- Experts: weighted least squares for A,b; weighted covariances
        if self.mean_function == "affine":
            XA = np.hstack([X, np.ones((n, 1))])  # (n, Dx+1)
            for k in range(K):
                w = gamma[:, k]
                Ak, bk = _weighted_linear_multiout(XA, y, w, Dx, Dy, reg=0.0)
                if P.A is None:
                    P.A = np.zeros((K, Dy, Dx))
                P.A[k] = Ak
                P.b[k] = bk
        else:
            for k in range(K):
                wk = gamma[:, k]
                s = wk.sum() + 1e-12
                P.b[k] = (wk[:, None] * y).sum(axis=0) / s

        # Residuals and covariance
        if self.mean_function == "affine":
            M = np.einsum("kij,nj->nki", P.A, X) + P.b[None, :, :]
        else:
            M = np.broadcast_to(P.b[None, :, :], (n, K, Dy))

        if self.shared_covariance:
            S = np.zeros((Dy, Dy))
            wtot = 0.0
            for k in range(K):
                R = y - M[:, k, :]
                S += _weighted_cov(
                    R, gamma[:, k], self.covariance_type, reg=self.reg_covar
                )
                wtot += gamma[:, k].sum()
            S /= max(wtot, 1e-12)
            if self.covariance_type == "diag":
                S = np.diag(np.maximum(np.diag(S), self.reg_covar))
            P.cov = S[None, :, :]
        else:
            cov = np.zeros((K, Dy, Dy))
            for k in range(K):
                R = y - M[:, k, :]
                cov[k] = _weighted_cov(
                    R, gamma[:, k], self.covariance_type, reg=self.reg_covar
                )
                if self.covariance_type == "diag":
                    cov[k] = np.diag(np.maximum(np.diag(cov[k]), self.reg_covar))
            P.cov = cov

        P.chol = _chol_from_cov(P.cov, self.covariance_type)

        # ---- Gating: weighted multinomial logistic regression with soft labels gamma
        W_new, c_new = _fit_softmax_gating_baseline(
            X,
            gamma,
            l2_w=self.gating_penalty,
            l2_b=(
                self.gating_penalty
                if self.gating_penalty_bias is None
                else self.gating_penalty_bias
            ),
            W_init=P.W,
            c_init=P.c,
            max_iter=self.gating_max_iter,
            tol=self.gating_tol,
        )
        P.W, P.c = W_new, c_new
        return P

    # ------------------------ gating utilities (baseline) ------------------------

    def _gating_logits_baseline(self, X: Array, W: Array, c: Array) -> Array:
        """
        Return logits with baseline class appended as zero column.
        W: (K-1, Dx), c: (K-1,)
        Output: (n, K)
        """
        Z = X @ W.T + c[None, :]  # (n, K-1)
        if Z.ndim == 1:
            Z = Z[:, None]
        Z = np.hstack([Z, np.zeros((Z.shape[0], 1))])  # append baseline 0
        return Z

    def _gating_softmax_baseline(self, X: Array, W: Array, c: Array) -> Array:
        Z = self._gating_logits_baseline(X, W, c)
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True)
        return P

    def condition(self, X: ArrayLike) -> Union[GaussianMixture, List[GaussianMixture]]:
        """
        Condition the mixture on X and return a GaussianMixture over y.

        This method creates sklearn GaussianMixture objects from the MoE's
        conditional mixture parameters.
        """
        check_is_fitted(self, attributes=["_params"])
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X = validate_data(self, X, reset=False)

        # Get mixture parameters
        params = self._compute_conditional_mixture(X)
        W, M, S = params["weights"], params["means"], params["covariances"]

        # Handle single sample
        if X.shape[0] == 1:
            # Create single GaussianMixture
            gmm = GaussianMixture(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
            )
            gmm.weights_ = W[0]
            gmm.means_ = M[0]
            gmm.covariances_ = S if S.ndim == 3 else S[0]
            gmm.precisions_cholesky_ = self._params.chol
            gmm.converged_ = True
            gmm.n_iter_ = 0
            return gmm

        # Handle batch - return list of GaussianMixture objects
        gmms = []
        for i in range(X.shape[0]):
            gmm = GaussianMixture(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
            )
            gmm.weights_ = W[i]
            gmm.means_ = M[i]
            gmm.covariances_ = S if S.ndim == 3 else S[i]
            gmm.precisions_cholesky_ = self._params.chol
            gmm.converged_ = True
            gmm.n_iter_ = 0
            gmms.append(gmm)

        return gmms


# ------------------------ helpers ------------------------


def _softmax_from_log(logw: Array) -> Array:
    m = np.max(logw)
    w = np.exp(logw - m)
    return w / (w.sum() + 1e-300)


def _emp_cov(Y: Array, reg: float, cov_type: str) -> Array:
    Yc = Y - Y.mean(axis=0, keepdims=True)
    C = (Yc.T @ Yc) / max(len(Y) - 1, 1)
    if cov_type == "diag":
        C = np.diag(np.maximum(np.diag(C), reg))
    else:
        C.flat[:: C.shape[0] + 1] += reg
    return C


def _chol_from_cov(cov: Array, cov_type: str) -> Optional[Array]:
    if cov_type != "full":
        return None
    if cov.ndim == 2:
        cov = cov[None, ...]
    K = cov.shape[0]
    out = np.empty_like(cov)
    for k in range(K):
        out[k] = cholesky(cov[k], lower=True, check_finite=False)
    return out


def _weighted_linear_multiout(
    XA: Array, Y: Array, w: Array, Dx: int, Dy: int, reg: float = 0.0
) -> Tuple[Array, Array]:
    sqrtw = np.sqrt(w + 1e-300)[:, None]
    Xw = XA * sqrtw
    Yw = Y * sqrtw
    XtX = Xw.T @ Xw
    if reg > 0:
        XtX.flat[:: XtX.shape[0] + 1] += reg
    XtY = Xw.T @ Yw
    XtX.flat[:: XtX.shape[0] + 1] += 1e-6
    try:
        L = cholesky(XtX, lower=True, check_finite=False)
        Z = solve_triangular(L, XtY, lower=True, check_finite=False)
        Beta = solve_triangular(L.T, Z, lower=False, check_finite=False)
    except np.linalg.LinAlgError:
        Beta = np.linalg.pinv(XtX) @ XtY
    A = Beta[:-1, :].T
    b = Beta[-1, :].copy()
    return A, b


def _weighted_cov(R: Array, w: Array, cov_type: str, reg: float) -> Array:
    Dy = R.shape[1]
    s = w.sum() + 1e-12
    R_mean = (w[:, None] * R).sum(axis=0) / s
    Rc = R - R_mean
    C = (Rc.T * w) @ Rc / s
    if cov_type == "diag":
        C = np.diag(np.maximum(np.diag(C), reg))
    else:
        C.flat[:: Dy + 1] += reg
    return C


def _log_gaussian_diag(x: Array, mean: Array, diag_cov: Array) -> float:
    if diag_cov.ndim == 2:
        diag_cov = np.diag(diag_cov)
    var = diag_cov
    diff = x - mean
    return float(-0.5 * (np.sum(np.log(2.0 * np.pi * var)) + np.sum(diff * diff / var)))


def _fit_softmax_gating_baseline(
    X: Array,
    Gamma: Array,
    l2_w: float,
    l2_b: float,
    W_init: Array,  # (K-1, Dx)
    c_init: Array,  # (K-1,)
    max_iter: int,
    tol: float,
    damping: float = 1e-6,
    backtrack_beta: float = 0.5,
    backtrack_max: int = 10,
) -> Tuple[Array, Array]:
    """
    Multinomial logistic regression with soft labels (Gamma) under baseline parameterization.
    Optimizes: sum_i sum_k Gamma_{ik} log P_{ik}(W,c) - 0.5*l2_w||W||^2 - 0.5*l2_b||c||^2
    using damped Newton with per-class block Hessians + Armijo backtracking.
    """
    n, Dx = X.shape
    K = Gamma.shape[1]
    Km1 = K - 1

    W = W_init.copy()
    c = c_init.copy()

    X1 = np.hstack([X, np.ones((n, 1))])  # (n, Dx+1)
    D = Dx + 1

    def _objective(W, c) -> float:
        Z_km1 = X @ W.T + c[None, :]  # (n, K-1)
        Z = np.hstack([Z_km1, np.zeros((n, 1))])  # append baseline
        Z -= Z.max(axis=1, keepdims=True)
        logP = Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))
        # only weights/biases for first K-1 classes are penalized
        return float(
            (Gamma[:, :Km1] * logP[:, :Km1]).sum()
            + (Gamma[:, Km1:] * logP[:, Km1:]).sum()
            - 0.5 * l2_w * np.sum(W * W)
            - 0.5 * l2_b * np.sum(c * c)
        )

    for _ in range(max_iter):
        # Forward probs
        Z_km1 = X @ W.T + c[None, :]  # (n, K-1)
        Z = np.hstack([Z_km1, np.zeros((n, 1))])  # (n, K)
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True)

        # Residuals and gradient on param classes (0..K-2)
        R = P - Gamma  # (n, K)
        G = X1.T @ R[:, :Km1]  # (D, K-1)
        # L2 penalties
        G[:-1, :] += l2_w * W.T
        G[-1, :] += l2_b * c

        grad_norm = np.linalg.norm(G) / (1.0 + np.linalg.norm(W) + np.linalg.norm(c))
        if grad_norm < tol:
            break

        # Compute per-class block Newton step with damping
        delta = np.empty((D, Km1))
        for k in range(Km1):
            pk = P[:, k]
            Wdiag = pk * (1.0 - pk)  # (n,)
            Xw = X1 * Wdiag[:, None]
            Hk = X1.T @ Xw  # (D,D)
            # L2 on weights/bias separately
            Hk[:-1, :-1] += l2_w * np.eye(Dx)
            Hk[-1, -1] += l2_b
            # LM damping
            Hk.flat[:: D + 1] += damping
            rhs = -(G[:, k])
            try:
                L = cholesky(Hk, lower=True, check_isfinite=False)
                z = solve_triangular(L, rhs, lower=True, check_finite=False)
                delta[:, k] = solve_triangular(L.T, z, lower=False, check_finite=False)
            except Exception:
                delta[:, k] = -np.linalg.pinv(Hk) @ rhs

        # Backtracking line search on the true objective
        obj0 = _objective(W, c)
        step = 1.0
        for _bt in range(backtrack_max):
            W_try = W + step * delta[:-1, :].T
            c_try = c + step * delta[-1, :]
            if _objective(W_try, c_try) > obj0:
                W, c = W_try, c_try
                break
            step *= backtrack_beta
        else:
            # if no improvement even after backtracking, stop
            break

    return W, c
