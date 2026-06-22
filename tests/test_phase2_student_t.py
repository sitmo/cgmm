"""
Phase 2 tests: MultivariateStudentT.

Numbered P2-xx. The key guarantees:
  * logpdf matches scipy.stats.multivariate_t.
  * Marginal / conditional stay Student-t with the documented parameter transforms
    (df -> df + q, scale inflation by (nu + d2)/(nu + q)).
  * The conditional satisfies the universal identity
        log p(x_t | x_c) = log p(x_t, x_c) - log p(x_c).
  * As nu -> inf the distribution and its conditioning reduce to the Gaussian
    (Phase-1 MultivariateNormal) results.
"""

import numpy as np
import pytest
from scipy.stats import multivariate_t

from cgmm import Distribution, MultivariateNormal, MultivariateStudentT


def _spd(d, seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d))
    return A @ A.T + d * np.eye(d)


# --------------------------------------------------------------------------
# Density & moments
# --------------------------------------------------------------------------
def test_P2_01_logpdf_matches_scipy():
    loc = np.array([1.0, -2.0, 0.5])
    scale = _spd(3, 1)
    df = 5.0
    dist = MultivariateStudentT(loc, scale, df)
    X = np.random.default_rng(2).normal(size=(7, 3))
    ref = multivariate_t(loc=loc, shape=scale, df=df).logpdf(X)
    assert np.allclose(dist.logpdf(X), ref)


def test_P2_02_mean_and_cov_definitions():
    loc = np.array([1.0, -2.0])
    scale = _spd(2, 2)
    df = 6.0
    dist = MultivariateStudentT(loc, scale, df)
    assert np.allclose(dist.mean(), loc)
    assert np.allclose(dist.cov(), df / (df - 2.0) * scale)


def test_P2_03_moments_undefined_raise():
    scale = _spd(2, 3)
    with pytest.raises(ValueError, match="mean is undefined"):
        MultivariateStudentT([0, 0], scale, df=0.8).mean()
    with pytest.raises(ValueError, match="cov is undefined"):
        MultivariateStudentT([0, 0], scale, df=1.5).cov()


def test_P2_04_invalid_df_raises():
    with pytest.raises(ValueError, match="df"):
        MultivariateStudentT([0, 0], _spd(2, 4), df=0.0)


def test_P2_05_rvs_moments():
    loc = np.array([2.0, -1.0])
    scale = _spd(2, 6)
    df = 8.0
    dist = MultivariateStudentT(loc, scale, df)
    S = dist.rvs(size=200_000, random_state=0)
    assert S.shape == (200_000, 2)
    assert np.allclose(S.mean(axis=0), loc, atol=0.05)
    # empirical covariance approaches df/(df-2) * scale
    assert np.allclose(np.cov(S.T), dist.cov(), rtol=0.1, atol=0.1)


def test_P2_06_satisfies_distribution_protocol():
    assert isinstance(MultivariateStudentT([0.0, 0.0], np.eye(2), df=4.0), Distribution)


# --------------------------------------------------------------------------
# Marginal & conditional
# --------------------------------------------------------------------------
def test_P2_07_marginal_is_student_t():
    loc = np.array([1.0, -2.0, 0.5])
    scale = _spd(3, 7)
    df = 5.0
    dist = MultivariateStudentT(loc, scale, df)
    m = dist.marginal([0, 2])
    assert isinstance(m, MultivariateStudentT)
    assert m.df == df
    assert np.allclose(m.loc_, loc[[0, 2]])
    assert np.allclose(m.scale_, scale[np.ix_([0, 2], [0, 2])])
    # cross-check density against scipy on the sub-block
    X = np.random.default_rng(0).normal(size=(4, 2))
    ref = multivariate_t(
        loc=loc[[0, 2]], shape=scale[np.ix_([0, 2], [0, 2])], df=df
    ).logpdf(X)
    assert np.allclose(m.logpdf(X), ref)


def test_P2_08_conditional_parameter_transforms():
    loc = np.array([1.0, -2.0, 0.5])
    scale = _spd(3, 8)
    df = 5.0
    dist = MultivariateStudentT(loc, scale, df)
    ci, ti = [0], [1, 2]
    x_c = np.array([0.7])

    cond = dist.condition(ci, x_c)
    # df increases by q
    assert cond.df == df + len(ci)
    # closed-form location, scale
    S_cc = scale[np.ix_(ci, ci)]
    S_tc = scale[np.ix_(ti, ci)]
    A = S_tc @ np.linalg.inv(S_cc)
    diff = x_c - loc[ci]
    loc_expected = loc[ti] + A @ diff
    schur = scale[np.ix_(ti, ti)] - A @ scale[np.ix_(ci, ti)]
    d2 = float(diff @ np.linalg.solve(S_cc, diff))
    scale_expected = (df + d2) / (df + len(ci)) * schur
    assert np.allclose(cond.loc_, loc_expected)
    assert np.allclose(cond.scale_, scale_expected)


def test_P2_09_conditional_identity():
    """log p(x_t | x_c) == log p(x_t, x_c) - log p(x_c) for the Student-t."""
    loc = np.array([1.0, -2.0, 0.5])
    scale = _spd(3, 9)
    df = 4.0
    dist = MultivariateStudentT(loc, scale, df)
    cond_idx = [0, 1]
    x_full = np.array([0.7, -1.3, 0.2])
    x_c = x_full[cond_idx]
    x_t = x_full[[2]]

    cond = dist.condition(cond_idx, x_c)
    lhs = cond.logpdf(x_t)[0]
    rhs = dist.logpdf(x_full)[0] - dist.marginal(cond_idx).logpdf(x_c)[0]
    assert np.isclose(lhs, rhs)


def test_P2_10_marginalize_condition_consistency():
    loc = np.arange(4, dtype=float)
    scale = _spd(4, 10)
    df = 7.0
    dist = MultivariateStudentT(loc, scale, df)
    x_c = np.array([0.5])  # condition on index 0
    a = dist.condition([0], x_c).marginal([0, 2])  # {1,3} within {1,2,3}
    b = dist.marginal([0, 1, 3]).condition([0], x_c)
    assert a.df == b.df
    assert np.allclose(a.loc_, b.loc_)
    assert np.allclose(a.scale_, b.scale_)


# --------------------------------------------------------------------------
# nu -> inf reduces to the Gaussian (Phase-1) results
# --------------------------------------------------------------------------
def test_P2_11_logpdf_reduces_to_gaussian():
    loc = np.array([1.0, -2.0, 0.5])
    scale = _spd(3, 11)
    X = np.random.default_rng(3).normal(size=(5, 3))
    t = MultivariateStudentT(loc, scale, df=1e9)
    g = MultivariateNormal(loc, scale)
    assert np.allclose(t.logpdf(X), g.logpdf(X), atol=1e-4)


def test_P2_12_conditional_reduces_to_gaussian():
    loc = np.array([1.0, -2.0, 0.5])
    scale = _spd(3, 12)
    x_c = np.array([0.7])
    t_cond = MultivariateStudentT(loc, scale, df=1e9).condition([0], x_c)
    g_cond = MultivariateNormal(loc, scale).condition([0], x_c)
    assert np.allclose(t_cond.loc_, g_cond.mean())
    # scale' -> Schur complement == Gaussian conditional covariance
    assert np.allclose(t_cond.scale_, g_cond.cov(), rtol=1e-6, atol=1e-6)
    # and the conditional covariance itself -> Gaussian conditional covariance
    assert np.allclose(t_cond.cov(), g_cond.cov(), rtol=1e-6, atol=1e-6)
