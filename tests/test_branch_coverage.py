"""
Targeted branch-coverage tests: validation errors, sampling edges, the
degrees-of-freedom solver, the protocol value objects' sampling, and the
MoE configuration branches (diag / constant-mean / shared / n_init).

Each test exercises a real behavior or contract, not just a line.
"""

import numpy as np
import pytest
from sklearn.mixture import GaussianMixture

from cgmm import (
    GMMConditioner,
    ConditionalGMMRegressor,
    MixtureOfExpertsRegressor,
    DiscriminativeConditionalGMMRegressor,
    MultivariateNormal,
    MultivariateStudentT,
    MultivariateGH,
    MixtureDistribution,
    GaussianMixtureDistribution,
    StudentTMixture,
    GeneralizedHyperbolicMixture,
)
from cgmm.student_t import _solve_dof


# ==========================================================================
# distributions: validation errors
# ==========================================================================
def test_mvn_validation_errors():
    with pytest.raises(ValueError, match="cov must have shape"):
        MultivariateNormal([0, 0], np.eye(3))
    d = MultivariateNormal([0, 0], np.eye(2))
    with pytest.raises(ValueError, match="expected 2"):
        d.logpdf(np.zeros((4, 3)))


def test_student_t_validation_errors():
    with pytest.raises(ValueError, match="scale must have shape"):
        MultivariateStudentT([0, 0], np.eye(3), df=5)
    d = MultivariateStudentT([0, 0], np.eye(2), df=5)
    with pytest.raises(ValueError, match="expected 2"):
        d.logpdf(np.zeros((4, 3)))


def test_gh_validation_errors():
    with pytest.raises(ValueError, match="scale must have shape"):
        MultivariateGH(-0.5, 1.0, 1.0, [0, 0], np.eye(3), [0, 0])
    with pytest.raises(ValueError, match="gamma must have length"):
        MultivariateGH(-0.5, 1.0, 1.0, [0, 0], np.eye(2), [0, 0, 0])
    d = MultivariateGH(-0.5, 1.0, 1.0, [0, 0], np.eye(2), [0.1, 0.2])
    with pytest.raises(ValueError, match="expected 2"):
        d.logpdf(np.zeros((4, 3)))
    # symmetric-t boundary (psi=0) requires lambda<0 and chi>0
    with pytest.raises(ValueError, match="Symmetric-t boundary"):
        MultivariateGH(0.5, 1.0, 0.0, [0, 0], np.eye(2), [0, 0])


# ==========================================================================
# distributions: symmetric-t GH boundary delegates moments/sampling to t
# ==========================================================================
def test_gh_symmetric_t_moments_and_rvs():
    nu = 6.0
    Sig = np.array([[1.5, 0.3], [0.3, 2.0]])
    gh = MultivariateGH(
        lambda_=-nu / 2, chi=nu, psi=0.0, loc=[1.0, -1.0], scale=Sig, gamma=[0, 0]
    )
    t = MultivariateStudentT([1.0, -1.0], Sig, nu)
    assert np.allclose(gh.mean(), t.mean())
    assert np.allclose(gh.cov(), t.cov())
    S = gh.rvs(size=2000, random_state=0)
    assert S.shape == (2000, 2)


def test_as_generator_accepts_legacy_randomstate():
    # passing a numpy RandomState must work (legacy stream support)
    rs = np.random.RandomState(0)
    S = MultivariateNormal([0.0, 0.0], np.eye(2)).rvs(size=5, random_state=rs)
    assert S.shape == (5, 2)


# ==========================================================================
# student_t: the 1-D dof solver
# ==========================================================================
def test_solve_dof_has_root_and_clamps():
    # const < 0 -> a finite root exists; recompute residual is ~0 there
    from scipy.special import digamma

    nu = _solve_dof(-0.3)
    resid = np.log(nu / 2) - digamma(nu / 2) + (-0.3)
    assert abs(resid) < 1e-4
    # const >= 0 -> no finite root -> clamps to the upper bound (Gaussian limit)
    nu_big = _solve_dof(2.0)
    assert nu_big >= 1e3


def test_student_t_mixture_fixed_dof_bic_and_sample_error():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(size=(200, 2)), rng.normal(loc=5, size=(200, 2))])
    m = StudentTMixture(n_components=2, dof=6.0, random_state=0, max_iter=30).fit(X)
    assert np.isfinite(m.bic(X)) and np.isfinite(
        m.aic(X)
    )  # fixed-dof n_parameters path
    with pytest.raises(ValueError, match="n_samples"):
        m.sample(0)


def test_gh_mixture_sample_error():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(size=(200, 2)), rng.normal(loc=6, size=(200, 2))])
    m = GeneralizedHyperbolicMixture(n_components=2, random_state=0, max_iter=20).fit(X)
    with pytest.raises(ValueError, match="n_samples"):
        m.sample(0)


# ==========================================================================
# container value objects
# ==========================================================================
def test_mixture_distribution_weights_length_mismatch():
    comps = [MultivariateNormal([0, 0], np.eye(2)) for _ in range(2)]
    with pytest.raises(ValueError, match="weights has length"):
        MixtureDistribution([1.0, 2.0, 3.0], comps)


def test_gaussian_mixture_distribution_rvs():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(400, 3))
    gmm = GaussianMixture(n_components=2, random_state=0).fit(Z)
    gmd = GMMConditioner(gmm, cond_idx=[0]).precompute().condition([0.3])
    assert isinstance(gmd, GaussianMixtureDistribution)
    S = gmd.rvs(size=5000, random_state=0)
    assert S.shape == (5000, 2)
    # sample mean approximates the mixture mean
    assert np.allclose(S.mean(axis=0), gmd.mean(), atol=0.15)


# ==========================================================================
# ConditionalGMMRegressor.responsibilities (gating + posterior)
# ==========================================================================
def test_conditional_gmm_responsibilities():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 2))
    y = X[:, 0] - 0.5 * X[:, 1] + 0.1 * rng.normal(size=300)
    m = ConditionalGMMRegressor(n_components=3, random_state=0).fit(X, y)
    W = m.responsibilities(X[:5])  # gating weights
    assert W.shape == (5, 3) and np.allclose(W.sum(axis=1), 1.0)
    G = m.responsibilities(X[:5], y[:5])  # posterior responsibilities
    assert G.shape == (5, 3) and np.allclose(G.sum(axis=1), 1.0)


# ==========================================================================
# discriminative: covariance_type validation
# ==========================================================================
def test_discriminative_invalid_covariance_type():
    with pytest.raises(ValueError, match="covariance_type"):
        DiscriminativeConditionalGMMRegressor(covariance_type="spherical")


# ==========================================================================
# MoE configuration branches: diag / constant mean / shared / n_init
# ==========================================================================
def _moe_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = 1.3 * X[:, 0] - 0.7 * X[:, 1] + 0.2 * rng.normal(size=n)
    return X, y


@pytest.mark.parametrize(
    "kwargs",
    [
        {"covariance_type": "diag"},
        {"mean_function": "constant"},
        {"shared_covariance": True},
        {"n_init": 2},
    ],
)
def test_moe_config_branches(kwargs):
    X, y = _moe_data()
    m = MixtureOfExpertsRegressor(n_components=2, random_state=0, max_iter=30, **kwargs)
    m.fit(X, y)
    assert m.predict(X[:5]).shape == (5,)
    assert np.isfinite(m.score(X, y))
    assert m.sample(X[:3], n_samples=4).shape == (3, 4, 1)
    # responsibilities with y exercises the per-expert density branch
    G = m.responsibilities(X[:5], y[:5])
    assert np.allclose(G.sum(axis=1), 1.0)
    gm = m.condition(X[:1])
    assert isinstance(gm, GaussianMixture)


def test_moe_diag_predict_cov_and_log_prob():
    X, y = _moe_data()
    m = MixtureOfExpertsRegressor(
        n_components=2, covariance_type="diag", random_state=0, max_iter=30
    ).fit(X, y)
    assert m.predict_cov(X[:4]).shape == (4, 1, 1)
    assert m.log_prob(X[:4], y[:4]).shape == (4,)


# ==========================================================================
# MixtureConditioner: raw MixtureDistribution input, lazy prepare, sklearn params
# ==========================================================================
def test_conditioner_accepts_raw_mixture_distribution():
    # a hand-built Student-t MixtureDistribution over 3 dims, conditioned directly
    comps = [
        MultivariateStudentT([0, 0, 0], np.eye(3), df=6),
        MultivariateStudentT([2, 1, -1], 1.5 * np.eye(3), df=6),
    ]
    md = MixtureDistribution([0.4, 0.6], comps)
    out = GMMConditioner(md, cond_idx=[0]).precompute().condition([0.5])
    assert isinstance(out, MixtureDistribution)
    assert out.dim == 2
    # matches the value object's own conditioning
    ref = md.condition([0], [0.5])
    assert np.allclose(out.weights_, ref.weights_)


def test_conditioner_lazy_prepare_without_precompute():
    rng = np.random.default_rng(0)
    gmm = GaussianMixture(n_components=2, random_state=0).fit(rng.normal(size=(300, 3)))
    cond = GMMConditioner(gmm, cond_idx=[0])  # no .precompute()
    out = cond.condition([0.2])  # must lazily prepare
    assert isinstance(out, GaussianMixtureDistribution)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mean_function": "constant"},
        {"dof": "shared"},
        {"dof": 5.0},
    ],
)
def test_moe_student_t_expert_configs(kwargs):
    X, y = _moe_data()
    m = MixtureOfExpertsRegressor(
        n_components=2, expert="student_t", random_state=0, max_iter=40, **kwargs
    ).fit(X, y)
    assert m.predict(X[:3]).shape == (3,)
    assert np.isfinite(m.score(X, y))
    if isinstance(kwargs.get("dof"), float):
        assert np.allclose(m._params.dofs, 5.0)  # fixed dof is not updated


def test_moe_random_init():
    X, y = _moe_data()
    m = MixtureOfExpertsRegressor(
        n_components=2, init_params="random", random_state=0, max_iter=30
    ).fit(X, y)
    assert m.predict(X[:3]).shape == (3,)


def test_conditioner_get_set_params():
    rng = np.random.default_rng(0)
    gmm = GaussianMixture(n_components=2, random_state=0).fit(rng.normal(size=(200, 3)))
    cond = GMMConditioner(gmm, cond_idx=[0], reg_covar=1e-8).precompute()
    p = cond.get_params()
    assert p["cond_idx"] == (0,) and p["reg_covar"] == 1e-8
    # set_params invalidates the prepared state, then re-conditions fine
    cond.set_params(cond_idx=(1,))
    assert cond._prepared is False
    out = cond.condition([0.1])
    assert isinstance(out, GaussianMixtureDistribution)
