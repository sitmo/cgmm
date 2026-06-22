"""
Phase 3 tests: StudentTMixture EM fitter.

Numbered P3-xx. Key guarantees:
  * EM log-likelihood is monotone non-decreasing (lower_bounds_).
  * As the fit pushes nu large on Gaussian data, it approaches a Gaussian fit.
  * Degrees of freedom are recovered (loosely) on simulated Student-t data.
  * The estimator honors the scikit-learn contract (clone/get_params, predict,
    score_samples, bic/aic) and the documented dof modes.
"""

import numpy as np
import pytest
from scipy.stats import multivariate_t
from sklearn.base import clone
from sklearn.mixture import GaussianMixture

from cgmm import StudentTMixture, MixtureDistribution


def _two_t_clusters(df=4.0, n=400, seed=0):
    a = multivariate_t(loc=[0, 0], shape=np.eye(2), df=df).rvs(n, random_state=seed)
    b = multivariate_t(loc=[7, 7], shape=np.eye(2), df=df).rvs(n, random_state=seed + 1)
    return np.vstack([a, b])


# --------------------------------------------------------------------------
# Core EM behavior
# --------------------------------------------------------------------------
def test_P3_01_loglik_monotone_non_decreasing():
    X = _two_t_clusters()
    m = StudentTMixture(n_components=2, random_state=0, max_iter=100).fit(X)
    diffs = np.diff(m.lower_bounds_)
    assert np.all(diffs >= -1e-9), f"min step {diffs.min()}"
    assert len(m.lower_bounds_) >= 2


def test_P3_02_recovers_cluster_structure():
    X = _two_t_clusters(df=5.0)
    m = StudentTMixture(
        n_components=2, dof="free", init_params="gmm", random_state=0, max_iter=300
    ).fit(X)
    # the two locations should be near (0,0) and (7,7) in some order
    locs = m.locations_[np.argsort(m.locations_[:, 0])]
    assert np.allclose(locs[0], [0, 0], atol=0.6)
    assert np.allclose(locs[1], [7, 7], atol=0.6)
    assert np.allclose(np.sort(m.weights_), [0.5, 0.5], atol=0.1)


def test_P3_03_dof_recovered_single_component():
    true_df = 4.0
    X = multivariate_t(loc=[1.0, -1.0], shape=np.eye(2), df=true_df).rvs(
        5000, random_state=3
    )
    m = StudentTMixture(n_components=1, dof="free", random_state=0, max_iter=300).fit(X)
    assert 2.5 < m.dofs_[0] < 6.5  # loose recovery of nu=4
    assert np.allclose(m.locations_[0], [1.0, -1.0], atol=0.1)


def test_P3_04_large_dof_on_gaussian_data_matches_gaussian():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(2000, 2)) @ np.array([[1.5, 0.4], [0.0, 1.0]]).T + [2.0, -3.0]
    st = StudentTMixture(n_components=1, dof="free", random_state=0, max_iter=300).fit(
        X
    )
    # Gaussian data -> EM should drive nu large (heavy tails rejected)
    assert st.dofs_[0] > 10.0
    gm = GaussianMixture(n_components=1, random_state=0).fit(X)
    # location matches the Gaussian mean
    assert np.allclose(st.locations_[0], gm.means_[0], atol=0.1)
    # the *implied covariance* nu/(nu-2)*scale matches the Gaussian covariance
    nu = st.dofs_[0]
    implied_cov = nu / (nu - 2.0) * st.scales_[0]
    assert np.allclose(implied_cov, gm.covariances_[0], rtol=0.1, atol=0.1)


# --------------------------------------------------------------------------
# dof modes
# --------------------------------------------------------------------------
def test_P3_05_dof_shared_gives_single_value():
    X = _two_t_clusters()
    m = StudentTMixture(n_components=2, dof="shared", random_state=0, max_iter=200).fit(
        X
    )
    assert np.allclose(m.dofs_, m.dofs_[0])  # all equal


def test_P3_06_dof_fixed_is_not_updated():
    X = _two_t_clusters()
    m = StudentTMixture(n_components=2, dof=7.0, random_state=0, max_iter=50).fit(X)
    assert np.allclose(m.dofs_, 7.0)


def test_P3_07_invalid_dof_rejected():
    X = _two_t_clusters()
    with pytest.raises(ValueError):
        StudentTMixture(n_components=2, dof="bogus").fit(X)


# --------------------------------------------------------------------------
# scikit-learn contract & API parity
# --------------------------------------------------------------------------
def test_P3_08_sklearn_clone_and_params():
    m = StudentTMixture(n_components=3, dof="shared", reg_covar=1e-5, random_state=1)
    params = m.get_params()
    assert params["n_components"] == 3 and params["dof"] == "shared"
    m2 = clone(m)
    assert m2.get_params() == params


def test_P3_09_predict_and_score_samples():
    X = _two_t_clusters()
    m = StudentTMixture(n_components=2, random_state=0).fit(X)
    assert m.score_samples(X).shape == (X.shape[0],)
    assert set(np.unique(m.predict(X))).issubset({0, 1})
    proba = m.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_P3_10_sample_shapes_and_bic_aic():
    X = _two_t_clusters()
    m = StudentTMixture(n_components=2, random_state=0).fit(X)
    Xs, ys = m.sample(50)
    assert Xs.shape == (50, 2) and ys.shape == (50,)
    assert np.isfinite(m.bic(X)) and np.isfinite(m.aic(X))


def test_P3_11_fitted_attributes_present():
    X = _two_t_clusters()
    m = StudentTMixture(n_components=2, random_state=0).fit(X)
    assert m.weights_.shape == (2,)
    assert m.locations_.shape == (2, 2)
    assert m.scales_.shape == (2, 2, 2)
    assert m.dofs_.shape == (2,)


def test_P3_12_to_mixture_distribution_matches_density():
    X = _two_t_clusters()
    m = StudentTMixture(n_components=2, random_state=0).fit(X)
    md = m.to_mixture_distribution()
    assert isinstance(md, MixtureDistribution)
    # MixtureDistribution.logpdf must equal the fitter's score_samples
    assert np.allclose(md.logpdf(X[:20]), m.score_samples(X[:20]))
