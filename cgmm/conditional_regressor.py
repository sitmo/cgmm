# cgmm/conditional_regressor.py
# Family-parameterized conditional mixture regressor: fit a joint mixture over
# [X, y], then analytically condition to obtain p(y | X). The same code path
# serves Gaussian and Student-t (and, later, Generalized Hyperbolic) families via
# the Mixture protocol; only the joint fitter differs.
#
# `family` is a real constructor parameter ONLY on the generic base
# (ConditionalMixtureRegressor). Named presets (ConditionalStudentTRegressor) keep
# an explicit __init__ that hardcodes the family, so `family` never appears in
# their get_params (sklearn clone / GridSearchCV stay happy).
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_is_fitted, validate_data

from .base import BaseConditionalMixture, ConditionalMixin
from .conditioner import MixtureConditioner
from .student_t import StudentTMixture
from .gh import GeneralizedHyperbolicMixture

_FAMILIES = ("gaussian", "student_t", "gh")


class ConditionalMixtureRegressor(BaseConditionalMixture, ConditionalMixin):
    """Conditional mixture regressor: learn a joint mixture over [X, y] and
    analytically condition to model p(y | X).

    Parameters
    ----------
    family : {"gaussian", "student_t"}, default="gaussian"
        Distribution family of the joint mixture.
    n_components : int, default=2
    covariance_type : str, default="full"
        Joint covariance type for the Gaussian family (ignored for "student_t",
        whose scale matrices are always full).
    dof : {"free", "shared"} or float, default="free"
        Degrees of freedom for the Student-t family (ignored for "gaussian").
    tol, reg_covar, max_iter, n_init, random_state, init_params :
        Passed through to the joint mixture fitter (sklearn names).
    cond_idx : sequence of int or None, default=None
        Columns of [X, y] to condition on. Defaults to the X columns (0..dx-1).
    return_cov : bool, default=False
        Whether ``predict`` returns the predictive covariance by default.

    Notes
    -----
    For the Student-t family the conditional is heteroskedastic: the per-component
    scale depends on the conditioning value, so ``predict_cov`` returns per-sample
    covariances. ``predict_cov`` requires the conditional degrees of freedom
    (nu + q) to exceed 2.
    """

    def __init__(
        self,
        family: str = "gaussian",
        *,
        n_components: int = 2,
        covariance_type: str = "full",
        dof="free",
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        max_iter: int = 200,
        n_init: int = 1,
        random_state=None,
        init_params: str = "kmeans",
        cond_idx=None,
        return_cov: bool = False,
    ):
        super().__init__(return_cov=return_cov)
        self.family = family
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.dof = dof
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.init_params = init_params
        self.cond_idx = cond_idx

    # ------------------------------------------------------------------ fit
    def _make_joint_fitter(self):
        if self.family == "gaussian":
            return GaussianMixture(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                tol=self.tol,
                reg_covar=self.reg_covar,
                max_iter=self.max_iter,
                n_init=self.n_init,
                random_state=self.random_state,
                init_params=self.init_params,
            )
        if self.family == "student_t":
            return StudentTMixture(
                n_components=self.n_components,
                dof=self.dof,
                tol=self.tol,
                reg_covar=self.reg_covar,
                max_iter=self.max_iter,
                n_init=self.n_init,
                random_state=self.random_state,
                init_params=self.init_params,
            )
        if self.family == "gh":
            return GeneralizedHyperbolicMixture(
                n_components=self.n_components,
                tol=self.tol,
                reg_covar=self.reg_covar,
                max_iter=self.max_iter,
                n_init=self.n_init,
                random_state=self.random_state,
                init_params=self.init_params,
            )
        raise ValueError(f"family must be one of {_FAMILIES}, got {self.family!r}.")

    def fit(self, X: ArrayLike, y: ArrayLike):
        X, y = validate_data(
            self, X, y, accept_sparse=False, y_numeric=True, multi_output=True
        )
        y = np.asarray(y)
        y2 = y[:, None] if y.ndim == 1 else y
        Z = np.concatenate([X, y2], axis=1)

        self.dx_ = X.shape[1]
        self.dy_ = y2.shape[1]
        self.n_targets_ = self.dy_

        joint = self._make_joint_fitter().fit(Z)
        self.n_iter_ = int(getattr(joint, "n_iter_", 1)) or 1
        self.converged_ = bool(getattr(joint, "converged_", True))

        cond_idx = (
            np.arange(self.dx_) if self.cond_idx is None else tuple(self.cond_idx)
        )
        self._cond = MixtureConditioner(
            mixture_estimator=joint, cond_idx=cond_idx, reg_covar=self.reg_covar
        ).precompute()
        self._joint_ = joint
        self.n_features_in_ = self.dx_
        return self

    # ------------------------------------------- conditioned mixtures helper
    def _conditioned(self, X):
        """Return a list of conditioned Mixture objects, one per row of X."""
        out = self._cond.condition(X)
        return out if isinstance(out, list) else [out]

    # ---- BaseConditionalMixture required method (drives predict/predict_cov) ----
    def _compute_conditional_mixture(self, X):
        check_is_fitted(self, attributes=["_cond", "n_features_in_", "n_targets_"])
        X = validate_data(self, X, reset=False)
        mixtures = self._conditioned(X)

        n = len(mixtures)
        K = self.n_components
        Dy = self.dy_
        W = np.empty((n, K))
        M = np.empty((n, K, Dy))

        if self.family == "gaussian":
            # conditional covariance is x-independent -> (K, Dy, Dy)
            S = np.empty((K, Dy, Dy))
            for i, mx in enumerate(mixtures):
                W[i] = mx.weights_
                M[i] = mx.means_
            S[:] = mixtures[0].covariances_
            return {"weights": W, "means": M, "covariances": S}

        # generic / heteroskedastic families -> per-sample (n, K, Dy, Dy)
        S = np.empty((n, K, Dy, Dy))
        for i, mx in enumerate(mixtures):
            W[i] = mx.weights_
            for k, comp in enumerate(mx.components_):
                M[i, k] = comp.mean()
                S[i, k] = comp.cov()
        return {"weights": W, "means": M, "covariances": S}

    # ---- family-correct density (overrides the Gaussian-only base impl) ----
    def log_prob(self, X: ArrayLike, y: ArrayLike) -> np.ndarray:
        check_is_fitted(self, attributes=["_cond", "n_features_in_", "n_targets_"])
        Xv = validate_data(self, X, reset=False)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1 and self.n_targets_ == 1:
            y = y[:, None]
        mixtures = self._conditioned(Xv)
        return np.array(
            [mx.logpdf(y[i])[0] for i, mx in enumerate(mixtures)], dtype=float
        )

    # ---- sampling p(y | X) ----
    def sample(self, X: ArrayLike, n_samples: int = 1):
        """Sample y | X.

        Returns (n_samples, Dy) for a single input row, else (n, n_samples, Dy).
        """
        check_is_fitted(self, attributes=["_cond"])
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        Xv = validate_data(self, X, reset=False)
        rng = np.random.default_rng(
            None if self.random_state is None else int(self.random_state)
        )
        mixtures = self._conditioned(Xv)
        draws = [mx.rvs(size=n_samples, random_state=rng) for mx in mixtures]
        if len(draws) == 1:
            return draws[0]
        return np.stack(draws, axis=0)

    # ---- conditioned mixture object(s) ----
    def condition(self, X: ArrayLike):
        check_is_fitted(self, attributes=["_cond"])
        Xv = validate_data(self, X, reset=False)
        return self._cond.condition(Xv)

    # ---- responsibilities ----
    def responsibilities(self, X: ArrayLike, y=None) -> np.ndarray:
        check_is_fitted(self, attributes=["_cond"])
        Xv = validate_data(self, X, reset=False)
        mixtures = self._conditioned(Xv)
        W = np.array([mx.weights_ for mx in mixtures])
        if y is None:
            return W
        y = np.asarray(y, dtype=float)
        if y.ndim == 1 and self.n_targets_ == 1:
            y = y[:, None]
        out = np.empty_like(W)
        for i, mx in enumerate(mixtures):
            log_terms = np.array(
                [
                    np.log(mx.weights_[k] + 1e-300) + comp.logpdf(y[i])[0]
                    for k, comp in enumerate(mx.components_)
                ]
            )
            m = np.max(log_terms)
            r = np.exp(log_terms - m)
            out[i] = r / r.sum()
        return out


class ConditionalStudentTRegressor(ConditionalMixtureRegressor):
    """Conditional Student-t mixture regressor (robust, heavy-tailed).

    A preset of :class:`ConditionalMixtureRegressor` with ``family="student_t"``.
    ``family`` is intentionally absent from this constructor so it never appears in
    ``get_params`` (keeping sklearn clone / GridSearchCV behavior identical to the
    other named regressors).
    """

    def __init__(
        self,
        n_components: int = 3,
        dof="free",
        *,
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        max_iter: int = 200,
        n_init: int = 1,
        random_state=None,
        init_params: str = "kmeans",
        cond_idx=None,
        return_cov: bool = False,
    ):
        super().__init__(
            family="student_t",
            n_components=n_components,
            dof=dof,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state,
            init_params=init_params,
            cond_idx=cond_idx,
            return_cov=return_cov,
        )


class ConditionalGHRegressor(ConditionalMixtureRegressor):
    """Conditional Generalized Hyperbolic mixture regressor (heavy tails + skew).

    A preset of :class:`ConditionalMixtureRegressor` with ``family="gh"``. As with
    the other named presets, ``family`` is absent from the constructor so it never
    appears in ``get_params``.
    """

    def __init__(
        self,
        n_components: int = 3,
        *,
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        max_iter: int = 100,
        n_init: int = 1,
        random_state=None,
        init_params: str = "gmm",
        cond_idx=None,
        return_cov: bool = False,
    ):
        super().__init__(
            family="gh",
            n_components=n_components,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state,
            init_params=init_params,
            cond_idx=cond_idx,
            return_cov=return_cov,
        )
