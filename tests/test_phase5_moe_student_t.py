"""
Phase 5 (MoE) tests: MixtureOfExpertsRegressor with expert="student_t".

Numbered P5M-xx. Key guarantees:
  * expert defaults to "gaussian" (existing behavior untouched).
  * Student-t experts: EM log-likelihood monotone, robust to heavy tails,
    family-correct predict/predict_cov/log_prob/condition/sample.
  * Unsupported configurations are rejected explicitly.
"""

import numpy as np
import pytest
from scipy.stats import t as student_t

from cgmm import (
    MixtureOfExpertsRegressor,
    MixtureDistribution,
    MultivariateStudentT,
)


def _data(n=500, df=3.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = 1.5 * X[:, 0] - 0.5 * X[:, 1] + student_t(df=df).rvs(n, random_state=seed + 1)
    return X, y


def test_P5M_01_default_expert_is_gaussian():
    m = MixtureOfExpertsRegressor(n_components=2)
    assert m.get_params()["expert"] == "gaussian"
    X, y = _data()
    m.fit(X, y)
    assert m.predict(X[:3]).shape == (3,)


def test_P5M_02_loglik_monotone():
    X, y = _data()
    m = MixtureOfExpertsRegressor(
        n_components=2, expert="student_t", random_state=0, max_iter=100
    ).fit(X, y)
    d = np.diff(m.lower_bounds_)
    assert np.all(d >= -1e-7), f"min step {d.min()}"


def test_P5M_03_predict_and_cov_shapes():
    X, y = _data()
    m = MixtureOfExpertsRegressor(
        n_components=2, expert="student_t", random_state=0
    ).fit(X, y)
    assert m.predict(X[:4]).shape == (4,)
    assert m.predict_cov(X[:4]).shape == (4, 1, 1)


def test_P5M_04_condition_returns_student_t_mixture():
    X, y = _data()
    m = MixtureOfExpertsRegressor(
        n_components=2, expert="student_t", random_state=0
    ).fit(X, y)
    cond = m.condition(X[:1])
    assert isinstance(cond, MixtureDistribution)
    assert all(isinstance(c, MultivariateStudentT) for c in cond.components_)


def test_P5M_05_sample_shapes():
    X, y = _data()
    m = MixtureOfExpertsRegressor(
        n_components=2, expert="student_t", random_state=0
    ).fit(X, y)
    assert m.sample(X[0], n_samples=6).shape == (6, 1)
    assert m.sample(X[:3], n_samples=6).shape == (3, 6, 1)


def test_P5M_06_log_prob_uses_t_density():
    X, y = _data()
    m = MixtureOfExpertsRegressor(
        n_components=2, expert="student_t", random_state=0
    ).fit(X, y)
    assert np.isclose(m.score(X, y), np.mean(m.log_prob(X, y)))
    # density must match the conditioned t-mixture's logpdf
    cond = m.condition(X[:5])
    manual = np.array([cond[i].logpdf(y[i : i + 1])[0] for i in range(5)])
    assert np.allclose(m.log_prob(X[:5], y[:5]), manual)


def test_P5M_07_unsupported_configs_raise():
    X, y = _data()
    with pytest.raises(NotImplementedError):
        MixtureOfExpertsRegressor(expert="student_t", covariance_type="diag").fit(X, y)
    with pytest.raises(NotImplementedError):
        MixtureOfExpertsRegressor(expert="student_t", shared_covariance=True).fit(X, y)
    with pytest.raises(ValueError, match="expert"):
        MixtureOfExpertsRegressor(expert="cauchy").fit(X, y)


def test_P5M_08_t_experts_beat_gaussian_on_heavy_tails():
    Xtr, ytr = _data(n=800, df=2.5, seed=0)
    Xte, yte = _data(n=800, df=2.5, seed=42)
    mt = MixtureOfExpertsRegressor(
        n_components=2, expert="student_t", random_state=0, max_iter=200
    ).fit(Xtr, ytr)
    mg = MixtureOfExpertsRegressor(
        n_components=2, expert="gaussian", random_state=0, max_iter=200
    ).fit(Xtr, ytr)
    assert mt.score(Xte, yte) > mg.score(Xte, yte)


def test_P5M_09_responsibilities_t():
    X, y = _data()
    m = MixtureOfExpertsRegressor(
        n_components=3, expert="student_t", random_state=0
    ).fit(X, y)
    W = m.responsibilities(X[:5])
    assert W.shape == (5, 3) and np.allclose(W.sum(axis=1), 1.0)
    G = m.responsibilities(X[:5], y[:5])
    assert np.allclose(G.sum(axis=1), 1.0)
