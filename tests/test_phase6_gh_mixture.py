"""
Phase 6 tests: GeneralizedHyperbolicMixture (MCECM) and ConditionalGHRegressor.

Numbered P6M-xx. Key guarantees:
  * EM log-likelihood is monotone non-decreasing.
  * The identifiability constraint E[W]=1 holds per component.
  * Cluster structure is recovered on GH-generated data; sklearn contract holds.
  * ConditionalGHRegressor conditions to a GH MixtureDistribution and honors the
    no-`family`-leak rule.
"""

import numpy as np
from sklearn.base import clone

from cgmm import (
    GeneralizedHyperbolicMixture,
    MultivariateGH,
    MixtureDistribution,
    ConditionalGHRegressor,
    ConditionalMixtureRegressor,
)
from cgmm.gh import _R


def _two_gh_clusters(n=500, seed=0):
    c0 = MultivariateGH(-0.5, 1.5, 1.5, [0, 0], [[1, 0.2], [0.2, 1]], [0.5, -0.3])
    c1 = MultivariateGH(-0.5, 1.5, 1.5, [7, 7], [[1, 0], [0, 1]], [-0.4, 0.2])
    return np.vstack([c0.rvs(n, random_state=seed), c1.rvs(n, random_state=seed + 1)])


# --------------------------------------------------------------------------
def test_P6M_01_loglik_monotone():
    X = _two_gh_clusters()
    m = GeneralizedHyperbolicMixture(
        n_components=2, init_params="gmm", random_state=0, max_iter=80
    ).fit(X)
    d = np.diff(m.lower_bounds_)
    assert np.all(d >= -1e-6), f"min step {d.min()}"
    assert len(m.lower_bounds_) >= 2


def test_P6M_02_identifiability_EW_equals_one():
    X = _two_gh_clusters()
    m = GeneralizedHyperbolicMixture(
        n_components=2, init_params="gmm", random_state=0, max_iter=80
    ).fit(X)
    for k in range(2):
        omega = np.sqrt(m.chis_[k] * m.psis_[k])
        e_w = _R(m.lambdas_[k], omega) * np.sqrt(m.chis_[k] / m.psis_[k])
        assert np.isclose(e_w, 1.0, atol=1e-3)


def test_P6M_03_recovers_cluster_structure():
    X = _two_gh_clusters()
    m = GeneralizedHyperbolicMixture(
        n_components=2, init_params="gmm", random_state=0, max_iter=120
    ).fit(X)
    locs = m.locations_[np.argsort(m.locations_[:, 0])]
    assert np.allclose(locs[0], [0, 0], atol=1.0)
    assert np.allclose(locs[1], [7, 7], atol=1.0)
    assert np.allclose(np.sort(m.weights_), [0.5, 0.5], atol=0.1)
    # the two clusters are clearly separated in the labels
    labels = m.predict(X)
    assert len(np.unique(labels)) == 2


def test_P6M_04_fitted_attributes_and_model_selection():
    X = _two_gh_clusters()
    m = GeneralizedHyperbolicMixture(n_components=2, random_state=0, max_iter=60).fit(X)
    assert m.weights_.shape == (2,)
    assert m.locations_.shape == (2, 2)
    assert m.scales_.shape == (2, 2, 2)
    assert m.gammas_.shape == (2, 2)
    assert m.lambdas_.shape == (2,) and m.chis_.shape == (2,) and m.psis_.shape == (2,)
    assert np.isfinite(m.bic(X)) and np.isfinite(m.aic(X))


def test_P6M_05_sklearn_clone_and_predict():
    m = GeneralizedHyperbolicMixture(n_components=3, reg_covar=1e-5, random_state=1)
    assert clone(m).get_params() == m.get_params()
    X = _two_gh_clusters()
    m2 = GeneralizedHyperbolicMixture(n_components=2, random_state=0, max_iter=40).fit(
        X
    )
    assert m2.score_samples(X).shape == (X.shape[0],)
    proba = m2.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_P6M_06_sample_and_to_mixture_distribution():
    X = _two_gh_clusters()
    m = GeneralizedHyperbolicMixture(n_components=2, random_state=0, max_iter=40).fit(X)
    Xs, ys = m.sample(30)
    assert Xs.shape == (30, 2) and ys.shape == (30,)
    md = m.to_mixture_distribution()
    assert isinstance(md, MixtureDistribution)
    assert all(isinstance(c, MultivariateGH) for c in md.components_)
    assert np.allclose(md.logpdf(X[:10]), m.score_samples(X[:10]))


# --------------------------------------------------------------------------
# ConditionalGHRegressor
# --------------------------------------------------------------------------
def _reg_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    from scipy.stats import skewnorm

    y = 1.5 * X[:, 0] - X[:, 1] + skewnorm(a=6).rvs(n, random_state=seed + 1)
    return X, y


def test_P6M_07_gh_regressor_fit_predict_condition():
    X, y = _reg_data()
    m = ConditionalGHRegressor(n_components=2, random_state=0, max_iter=60).fit(X, y)
    assert m.predict(X[:5]).shape == (5,)
    assert m.predict_cov(X[:5]).shape == (5, 1, 1)
    cond = m.condition(X[:1])
    assert isinstance(cond, MixtureDistribution)
    assert all(isinstance(c, MultivariateGH) for c in cond.components_)


def test_P6M_08_gh_regressor_no_family_leak_and_clone():
    m = ConditionalGHRegressor(n_components=2, random_state=0)
    params = m.get_params()
    assert "family" not in params
    assert clone(m).get_params() == params


def test_P6M_09_family_dispatch_gh():
    X, y = _reg_data()
    m = ConditionalMixtureRegressor(
        family="gh", n_components=2, random_state=0, max_iter=60
    ).fit(X, y)
    assert m.predict(X[:3]).shape == (3,)
    assert "family" in m.get_params()


def test_P6M_10_score_consistency():
    X, y = _reg_data()
    m = ConditionalGHRegressor(n_components=2, random_state=0, max_iter=60).fit(X, y)
    assert np.isclose(m.score(X, y), np.mean(m.log_prob(X, y)))
