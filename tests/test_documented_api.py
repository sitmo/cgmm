"""
Interface tests for documented API use cases that previously lacked coverage.

These pin down behaviors promised by README.md / docs/index.md / docs/api.md so
the documented contract becomes a regression gate. Each test is numbered and
cross-references the gap it closes (see the coverage review):

  GAP-A : predict_cov() documented (README, index.md Quick Start) but untested.
  GAP-C : covariance_type "diag"/"tied"/"spherical" documented but uncovered.
  GAP-D : mean_function="linear" documented as an affine alias.
  GAP-E : .bic()/.aic() must keep working on conditioned GaussianMixture objects
          (the GaussianMixtureDistribution guarantee the heavy-tailed work relies
          on for backward compatibility).
"""

import numpy as np
import pytest
from sklearn.mixture import GaussianMixture

from cgmm import (
    ConditionalGMMRegressor,
    MixtureOfExpertsRegressor,
    DiscriminativeConditionalGMMRegressor,
    GMMConditioner,
)


def _data_single(seed=0, n=300):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = 2.0 * X[:, 0] + X[:, 1] + 0.1 * rng.normal(size=n)
    return X, y


def _data_multi(seed=0, n=300):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    Y = np.c_[
        2.0 * X[:, 0] + 0.1 * rng.normal(size=n),
        X[:, 1] - X[:, 0] + 0.1 * rng.normal(size=n),
    ]
    return X, Y


ALL_REGRESSORS = [
    ConditionalGMMRegressor,
    MixtureOfExpertsRegressor,
    DiscriminativeConditionalGMMRegressor,
]


# --------------------------------------------------------------------------
# GAP-A : predict_cov()
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cls", ALL_REGRESSORS)
def test_01_predict_cov_single_output_shape(cls):
    """predict_cov returns full (n, 1, 1) covariance matrices for single output."""
    X, y = _data_single()
    model = cls(n_components=3, random_state=42).fit(X, y)
    cov = model.predict_cov(X[:5])
    assert cov.shape == (5, 1, 1)
    assert np.all(cov[:, 0, 0] > 0)  # variances positive


@pytest.mark.parametrize("cls", ALL_REGRESSORS)
def test_02_predict_cov_multi_output_shape(cls):
    """predict_cov returns (n, Dy, Dy) symmetric PSD matrices for multi-output."""
    X, Y = _data_multi()
    model = cls(n_components=3, random_state=42).fit(X, Y)
    cov = model.predict_cov(X[:4])
    assert cov.shape == (4, 2, 2)
    for C in cov:
        assert np.allclose(C, C.T, atol=1e-8)
        assert np.all(np.linalg.eigvalsh(C) > -1e-9)


def test_03_predict_cov_consistent_with_return_cov():
    """predict_cov[:,0,0] equals predict(return_cov=True) for single output."""
    X, y = _data_single()
    model = ConditionalGMMRegressor(n_components=3, random_state=42).fit(X, y)
    cov_full = model.predict_cov(X[:6])
    _, cov_predict = model.predict(X[:6], return_cov=True)
    assert np.allclose(cov_full[:, 0, 0], cov_predict)


# --------------------------------------------------------------------------
# GAP-D : mean_function="linear" alias
# --------------------------------------------------------------------------
def test_04_mean_function_linear_aliases_affine():
    """mean_function='linear' must behave identically to 'affine'."""
    X, y = _data_single()
    affine = MixtureOfExpertsRegressor(
        n_components=2, mean_function="affine", random_state=0
    ).fit(X, y)
    linear = MixtureOfExpertsRegressor(
        n_components=2, mean_function="linear", random_state=0
    ).fit(X, y)
    assert np.allclose(affine.predict(X[:10]), linear.predict(X[:10]))


def test_05_mean_function_linear_not_constant():
    """'linear' must be a genuine affine fit, distinct from 'constant'."""
    X, y = _data_single()
    linear = MixtureOfExpertsRegressor(
        n_components=2, mean_function="linear", random_state=0
    ).fit(X, y)
    constant = MixtureOfExpertsRegressor(
        n_components=2, mean_function="constant", random_state=0
    ).fit(X, y)
    assert not np.allclose(linear.predict(X[:10]), constant.predict(X[:10]))


def test_06_invalid_mean_function_raises():
    """An unknown mean_function is rejected in fit (not silently ignored)."""
    X, y = _data_single()
    with pytest.raises(ValueError, match="mean_function"):
        MixtureOfExpertsRegressor(mean_function="bogus", random_state=0).fit(X, y)


# --------------------------------------------------------------------------
# GAP-C : covariance_type diag/tied/spherical conditioning
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ctype", ["full", "diag", "tied", "spherical"])
def test_07_regressor_supports_all_covariance_types(ctype):
    """ConditionalGMMRegressor predicts for every documented covariance_type."""
    X, y = _data_single()
    model = ConditionalGMMRegressor(
        n_components=3, covariance_type=ctype, random_state=42
    ).fit(X, y)
    pred = model.predict(X[:5])
    assert pred.shape == (5,)
    assert np.all(np.isfinite(pred))


@pytest.mark.parametrize("ctype", ["full", "diag", "tied", "spherical"])
def test_08_conditioner_accepts_all_covariance_types(ctype):
    """GMMConditioner conditions a GMM of any covariance_type and returns a
    GaussianMixture with full conditional covariance."""
    X, y = _data_single()
    Z = np.c_[X, y]
    gmm = GaussianMixture(n_components=3, covariance_type=ctype, random_state=42).fit(Z)
    cond = GMMConditioner(gmm, cond_idx=[0, 1]).precompute()
    out = cond.condition(np.array([0.3, -0.2]))
    assert isinstance(out, GaussianMixture)
    assert out.covariances_.shape == (3, 1, 1)


# --------------------------------------------------------------------------
# GAP-E : .bic()/.aic() on conditioned GaussianMixture (back-compat keystone)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cls", ALL_REGRESSORS)
def test_09_conditioned_gmm_supports_bic_aic(cls):
    """condition() output must remain a usable sklearn GaussianMixture exposing
    bic()/aic(). This is the contract GaussianMixtureDistribution must preserve."""
    X, y = _data_single()
    model = cls(n_components=3, random_state=42).fit(X, y)
    gmm = model.condition(X[:1])
    if isinstance(gmm, list):
        gmm = gmm[0]
    assert isinstance(gmm, GaussianMixture)
    y_eval = y[:1].reshape(1, -1)
    assert np.isfinite(gmm.bic(y_eval))
    assert np.isfinite(gmm.aic(y_eval))
