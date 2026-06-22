"""
Phase 1 tests: Mixture/Distribution protocols, MultivariateNormal,
MixtureDistribution, and GaussianMixtureDistribution.

These pin the conditioning contract for the heavy-tailed work. Numbered P1-xx.

Key guarantees exercised here:
  * GaussianMixtureDistribution remains a real sklearn GaussianMixture (keystone
    back-compat invariant) while satisfying the Mixture protocol.
  * The two independent conditioning code paths -- MixtureDistribution.condition
    (generic, per-component) and GMMConditioner / GaussianMixtureDistribution
    (cached matrices) -- agree to numerical tolerance.
  * MultivariateNormal conditioning satisfies the analytic identity
    log p(x_t | x_c) = log p(x_t, x_c) - log p(x_c).
"""

import numpy as np
import pytest
from scipy.stats import multivariate_normal as mvn
from sklearn.mixture import GaussianMixture

from cgmm import (
    GMMConditioner,
    Distribution,
    MultivariateNormal,
    Mixture,
    MixtureDistribution,
    GaussianMixtureDistribution,
)


def _spd(d, seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d))
    return A @ A.T + d * np.eye(d)


def _fit_joint_gmm(d=3, k=3, n=600, seed=0, cov_type="full"):
    rng = np.random.default_rng(seed)
    Z = np.vstack([rng.normal(loc=c, size=(n, d)) for c in range(k)])
    return GaussianMixture(
        n_components=k, covariance_type=cov_type, random_state=seed
    ).fit(Z)


# --------------------------------------------------------------------------
# MultivariateNormal (Layer 1)
# --------------------------------------------------------------------------
def test_P1_01_mvn_logpdf_matches_scipy():
    mean = np.array([1.0, -2.0, 0.5])
    cov = _spd(3, 1)
    dist = MultivariateNormal(mean, cov)
    X = np.random.default_rng(2).normal(size=(7, 3))
    assert np.allclose(dist.logpdf(X), mvn(mean, cov).logpdf(X))


def test_P1_02_mvn_conditional_identity():
    """log p(x_t | x_c) == log p(x_t, x_c) - log p(x_c)."""
    mean = np.array([1.0, -2.0, 0.5])
    cov = _spd(3, 3)
    dist = MultivariateNormal(mean, cov)
    cond_idx = [0]
    x_full = np.array([0.7, -1.3, 0.2])
    x_c = x_full[cond_idx]
    x_t = x_full[[1, 2]]

    cond = dist.condition(cond_idx, x_c)
    lhs = cond.logpdf(x_t)[0]
    rhs = dist.logpdf(x_full)[0] - dist.marginal(cond_idx).logpdf(x_c)[0]
    assert np.isclose(lhs, rhs)


def test_P1_03_mvn_conditional_matches_closed_form():
    mean = np.array([1.0, -2.0, 0.5])
    cov = _spd(3, 4)
    dist = MultivariateNormal(mean, cov)
    ci, ti = [0, 1], [2]
    x_c = np.array([0.3, 0.9])

    cond = dist.condition(ci, x_c)
    S_cc = cov[np.ix_(ci, ci)]
    S_tc = cov[np.ix_(ti, ci)]
    A = S_tc @ np.linalg.inv(S_cc)
    mu_expected = mean[ti] + A @ (x_c - mean[ci])
    S_expected = cov[np.ix_(ti, ti)] - A @ cov[np.ix_(ci, ti)]
    assert np.allclose(cond.mean(), mu_expected)
    assert np.allclose(cond.cov(), S_expected)


def test_P1_04_mvn_marginal():
    mean = np.array([1.0, -2.0, 0.5])
    cov = _spd(3, 5)
    dist = MultivariateNormal(mean, cov)
    m = dist.marginal([0, 2])
    assert np.allclose(m.mean(), mean[[0, 2]])
    assert np.allclose(m.cov(), cov[np.ix_([0, 2], [0, 2])])


def test_P1_05_mvn_rvs_moments():
    mean = np.array([2.0, -1.0])
    cov = _spd(2, 6)
    dist = MultivariateNormal(mean, cov)
    S = dist.rvs(size=50_000, random_state=0)
    assert S.shape == (50_000, 2)
    assert np.allclose(S.mean(axis=0), mean, atol=0.05)
    assert np.allclose(np.cov(S.T), cov, atol=0.1)


def test_P1_06_mvn_marginalize_then_condition_consistency():
    """Conditioning on a block, then marginalizing, equals marginalizing to the
    union block first then conditioning."""
    mean = np.arange(4, dtype=float)
    cov = _spd(4, 7)
    dist = MultivariateNormal(mean, cov)
    x_c = np.array([0.5])  # condition on index 0

    # path A: condition on 0 (-> dims 1,2,3), then marginalize to dims 1,3
    a = dist.condition([0], x_c).marginal([0, 2])  # local idx of {1,3} in {1,2,3}
    # path B: marginalize to {0,1,3} first, then condition on 0 (-> {1,3})
    b = dist.marginal([0, 1, 3]).condition([0], x_c)
    assert np.allclose(a.mean(), b.mean())
    assert np.allclose(a.cov(), b.cov())


def test_P1_07_mvn_satisfies_distribution_protocol():
    dist = MultivariateNormal([0.0, 0.0], np.eye(2))
    assert isinstance(dist, Distribution)


# --------------------------------------------------------------------------
# GaussianMixtureDistribution (Layer 2, back-compat keystone)
# --------------------------------------------------------------------------
def test_P1_08_gmd_is_real_gaussian_mixture():
    gmm = _fit_joint_gmm()
    gmd = GMMConditioner(gmm, cond_idx=[0]).precompute().condition([0.5])
    assert isinstance(gmd, GaussianMixtureDistribution)
    assert isinstance(gmd, GaussianMixture)  # KEYSTONE
    assert isinstance(gmd, Mixture)
    # sklearn methods still work
    y = np.zeros((1, 2))
    assert np.isfinite(gmd.bic(y)) and np.isfinite(gmd.aic(y))
    assert gmd.sample(3)[0].shape == (3, 2)


def test_P1_09_gmd_logpdf_equals_score_samples():
    gmm = _fit_joint_gmm()
    gmd = GMMConditioner(gmm, cond_idx=[0]).precompute().condition([0.5])
    Y = np.random.default_rng(0).normal(size=(5, 2))
    assert np.allclose(gmd.logpdf(Y), gmd.score_samples(Y))


def test_P1_10_gmd_components_view_matches_arrays():
    gmm = _fit_joint_gmm()
    gmd = GMMConditioner(gmm, cond_idx=[0]).precompute().condition([0.5])
    comps = gmd.components_
    assert all(isinstance(c, MultivariateNormal) for c in comps)
    for k, c in enumerate(comps):
        assert np.allclose(c.mean(), gmd.means_[k])
        assert np.allclose(c.cov(), gmd.covariances_[k])


def test_P1_11_gmd_mean_cov_match_mixture_formula():
    gmm = _fit_joint_gmm()
    gmd = GMMConditioner(gmm, cond_idx=[0]).precompute().condition([0.5])
    # reference mixture mean/cov from the same components
    md = MixtureDistribution(gmd.weights_, gmd.components_)
    assert np.allclose(gmd.mean(), md.mean())
    assert np.allclose(gmd.cov(), md.cov())


# --------------------------------------------------------------------------
# MixtureDistribution vs GaussianMixtureDistribution conditioning (cross-check)
# --------------------------------------------------------------------------
def test_P1_12_mixture_logpdf_matches_sklearn_gmm():
    """A MixtureDistribution of MVN components reproduces sklearn GMM density."""
    gmm = _fit_joint_gmm(d=2)
    md = MixtureDistribution(
        gmm.weights_,
        [MultivariateNormal(m, S) for m, S in zip(gmm.means_, gmm.covariances_)],
    )
    X = np.random.default_rng(1).normal(size=(6, 2))
    assert np.allclose(md.logpdf(X), gmm.score_samples(X))


def test_P1_13_two_conditioning_paths_agree():
    """MixtureDistribution.condition (generic per-component) must agree with
    GMMConditioner / GaussianMixtureDistribution.condition (cached matrices)."""
    gmm = _fit_joint_gmm(d=3)
    x_c = np.array([0.4])

    # Path A: cached-matrix conditioner
    gmd = GMMConditioner(gmm, cond_idx=[0]).precompute().condition(x_c)

    # Path B: generic value-object conditioner
    md = MixtureDistribution(
        gmm.weights_,
        [MultivariateNormal(m, S) for m, S in zip(gmm.means_, gmm.covariances_)],
    )
    md_cond = md.condition([0], x_c)

    assert np.allclose(gmd.weights_, md_cond.weights_)
    means_b = np.array([c.mean() for c in md_cond.components_])
    covs_b = np.array([c.cov() for c in md_cond.components_])
    assert np.allclose(gmd.means_, means_b)
    assert np.allclose(gmd.covariances_, covs_b)


def test_P1_14_gmd_condition_is_chainable():
    """GaussianMixtureDistribution.condition returns another GMD (so conditioning
    composes)."""
    gmm = _fit_joint_gmm(d=3)
    gmd = GMMConditioner(gmm, cond_idx=[0]).precompute().condition([0.5])
    # condition the resulting 2D mixture further on its first dim
    gmd2 = gmd.condition([0], [0.1])
    assert isinstance(gmd2, GaussianMixtureDistribution)
    assert gmd2.means_.shape[1] == 1


@pytest.mark.parametrize("ctype", ["full", "diag", "tied", "spherical"])
def test_P1_15_conditioner_returns_gmd_all_cov_types(ctype):
    gmm = _fit_joint_gmm(cov_type=ctype)
    out = GMMConditioner(gmm, cond_idx=[0]).precompute().condition([0.5])
    assert isinstance(out, GaussianMixtureDistribution)
    batch = GMMConditioner(gmm, cond_idx=[0]).precompute().condition([[0.5], [0.1]])
    assert isinstance(batch, list)
    assert all(isinstance(b, GaussianMixtureDistribution) for b in batch)
