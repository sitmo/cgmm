"""
Tests for the BaseConditionalMixture machinery: covariance-shape handling in
log_prob / predict_cov, the mixture-covariance helper, and the base
responsibilities / condition contracts.

These use a tiny controlled subclass that injects hand-crafted conditional
mixture parameters, so we can exercise every documented covariance form
(full (K,Dy,Dy) and (n,K,Dy,Dy); diag (K,Dy) and (n,K,Dy)) and the error path.
"""

import numpy as np
import pytest

from cgmm.base import BaseConditionalMixture, ConditionalMixin, _log_gaussian_diag


class _InjectedMixture(BaseConditionalMixture):
    """Returns pre-built conditional mixture params, ignoring X."""

    def __init__(self, params, n_targets, *, return_cov=False):
        super().__init__(return_cov=return_cov)
        self._params = params
        self.n_targets_ = n_targets
        self.n_features_in_ = 1

    def fit(self, X, y):  # pragma: no cover - not used
        return self

    def _compute_conditional_mixture(self, X):
        return self._params


def _spd(d, seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d))
    return A @ A.T + d * np.eye(d)


def _params(cov_kind, n=3, K=2, Dy=2, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.random((n, K))
    W /= W.sum(axis=1, keepdims=True)
    M = rng.normal(size=(n, K, Dy))
    if cov_kind == "full_nk":
        S = np.array([[_spd(Dy, i * 10 + k) for k in range(K)] for i in range(n)])
    elif cov_kind == "full_k":
        S = np.array([_spd(Dy, k) for k in range(K)])
    elif cov_kind == "diag_nk":
        S = rng.random((n, K, Dy)) + 0.5
    elif cov_kind == "diag_k":
        S = rng.random((K, Dy)) + 0.5
    elif cov_kind == "bad":
        S = rng.random((K,))  # 1-D: unsupported
    return {"weights": W, "means": M, "covariances": S}, (n, K, Dy)


# --------------------------------------------------------------------------
# _mixture_covariance_from_params: every covariance form -> (n, Dy, Dy)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["full_nk", "full_k", "diag_nk", "diag_k"])
def test_mixture_covariance_shapes(kind):
    p, (n, K, Dy) = _params(kind)
    C = BaseConditionalMixture._mixture_covariance_from_params(
        p["weights"], p["means"], p["covariances"]
    )
    assert C.shape == (n, Dy, Dy)
    # symmetric PSD
    for Ci in C:
        assert np.allclose(Ci, Ci.T)
        assert np.all(np.linalg.eigvalsh(Ci) > -1e-9)


def test_mixture_covariance_invalid_raises():
    p, _ = _params("bad")
    with pytest.raises(ValueError, match="covariances must be"):
        BaseConditionalMixture._mixture_covariance_from_params(
            p["weights"], p["means"], p["covariances"]
        )


def test_diag_full_covariance_agree():
    """A diag spec and the equivalent full spec give the same predictive cov."""
    n, K, Dy = 3, 2, 2
    rng = np.random.default_rng(1)
    W = rng.random((n, K))
    W /= W.sum(1, keepdims=True)
    M = rng.normal(size=(n, K, Dy))
    var = rng.random((K, Dy)) + 0.5
    full = np.array([np.diag(var[k]) for k in range(K)])
    Cd = BaseConditionalMixture._mixture_covariance_from_params(W, M, var)
    Cf = BaseConditionalMixture._mixture_covariance_from_params(W, M, full)
    assert np.allclose(Cd, Cf)


# --------------------------------------------------------------------------
# log_prob across covariance forms (full 4-D, diag (n,K,Dy) and (K,Dy)) + error
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["full_nk", "full_k", "diag_nk", "diag_k"])
def test_log_prob_covariance_forms(kind):
    p, (n, K, Dy) = _params(kind)
    m = _InjectedMixture(p, n_targets=Dy)
    y = np.random.default_rng(2).normal(size=(n, Dy))
    lp = m.log_prob(np.zeros((n, 1)), y)
    assert lp.shape == (n,)
    assert np.all(np.isfinite(lp))


def test_log_prob_invalid_covariance_raises():
    p, (n, K, Dy) = _params("bad")
    m = _InjectedMixture(p, n_targets=Dy)
    with pytest.raises(ValueError, match="covariances must be"):
        m.log_prob(np.zeros((n, 1)), np.zeros((n, Dy)))


def test_log_prob_diag_matches_full():
    """Diag-cov log_prob equals the full-cov log_prob built from the same vars."""
    n, K, Dy = 3, 2, 2
    rng = np.random.default_rng(3)
    W = rng.random((n, K))
    W /= W.sum(1, keepdims=True)
    M = rng.normal(size=(n, K, Dy))
    var = rng.random((K, Dy)) + 0.5
    y = rng.normal(size=(n, Dy))
    diag = _InjectedMixture({"weights": W, "means": M, "covariances": var}, Dy)
    full_cov = np.array([np.diag(var[k]) for k in range(K)])
    full = _InjectedMixture({"weights": W, "means": M, "covariances": full_cov}, Dy)
    assert np.allclose(
        diag.log_prob(np.zeros((n, 1)), y), full.log_prob(np.zeros((n, 1)), y)
    )


def test_predict_cov_diag_path():
    p, (n, K, Dy) = _params("diag_nk")
    m = _InjectedMixture(p, n_targets=Dy)
    C = m.predict_cov(np.zeros((n, 1)))
    assert C.shape == (n, Dy, Dy)


# --------------------------------------------------------------------------
# base responsibilities + ConditionalMixin.condition contracts
# --------------------------------------------------------------------------
def test_base_responsibilities_contract():
    p, (n, K, Dy) = _params("full_k")
    m = _InjectedMixture(p, n_targets=Dy)
    W = m.responsibilities(np.zeros((n, 1)))  # y=None -> gating weights
    assert np.allclose(W, p["weights"])
    with pytest.raises(NotImplementedError):
        m.responsibilities(np.zeros((n, 1)), y=np.zeros((n, Dy)))


def test_conditional_mixin_condition_not_implemented():
    class _C(ConditionalMixin):
        pass

    with pytest.raises(NotImplementedError, match="must implement condition"):
        _C().condition(np.zeros((2, 1)))


def test_log_gaussian_diag_matches_scipy():
    from scipy.stats import multivariate_normal as mvn

    x = np.array([0.3, -1.2])
    mean = np.array([0.1, 0.5])
    var = np.array([0.7, 1.3])
    got = _log_gaussian_diag(x, mean, var)
    exp = mvn(mean, np.diag(var)).logpdf(x)
    assert np.isclose(got, exp)
