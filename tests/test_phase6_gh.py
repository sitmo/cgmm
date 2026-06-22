"""
Phase 6 tests: MultivariateGH (Generalized Hyperbolic) distribution.

Numbered P6-xx. Key guarantees:
  * The GH density is correctly normalized (integrates to 1).
  * GH is closed under marginalization/conditioning with the documented parameter
    transforms, validated by the universal identity
        log p(x_t | x_c) = log p(x_t, x_c) - log p(x_c).
  * The symmetric Student-t boundary (psi=0, gamma=0, lambda=-nu/2, chi=nu) agrees
    with the direct MultivariateStudentT path -- for BOTH the density and the
    conditioning -- cross-validating the two implementations.
  * mean()/cov() (GIG moments) agree with Monte-Carlo from rvs().
"""

import numpy as np
import pytest
from scipy.integrate import dblquad, quad

from cgmm import Distribution, MultivariateGH, MultivariateStudentT


def _spd(d, seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d))
    return A @ A.T + d * np.eye(d)


# --------------------------------------------------------------------------
# Density normalization
# --------------------------------------------------------------------------
def test_P6_01_pdf_integrates_to_one_1d_asymmetric():
    g = MultivariateGH(
        lambda_=-0.7, chi=1.5, psi=2.0, loc=[0.3], scale=[[1.2]], gamma=[0.8]
    )
    integral, _ = quad(lambda t: np.exp(g.logpdf([[t]])[0]), -50, 50)
    assert abs(integral - 1.0) < 1e-4


def test_P6_02_pdf_integrates_to_one_2d():
    S = _spd(2, 1)
    g = MultivariateGH(
        lambda_=-0.5, chi=2.0, psi=1.3, loc=[0.0, 0.0], scale=S, gamma=[0.5, -0.3]
    )
    val, _ = dblquad(lambda y, x: np.exp(g.logpdf([[x, y]])[0]), -15, 15, -15, 15)
    assert abs(val - 1.0) < 1e-3


def test_P6_03_symmetric_when_gamma_zero():
    S = _spd(1, 2)
    g = MultivariateGH(lambda_=-0.6, chi=1.0, psi=1.0, loc=[0.5], scale=S, gamma=[0.0])
    a = g.logpdf([[0.5 + 1.3]])[0]
    b = g.logpdf([[0.5 - 1.3]])[0]
    assert np.isclose(a, b)


# --------------------------------------------------------------------------
# Closure under marginalization / conditioning
# --------------------------------------------------------------------------
def test_P6_04_conditional_identity():
    S = _spd(3, 3)
    g = MultivariateGH(
        lambda_=-0.5,
        chi=2.0,
        psi=1.3,
        loc=[0.5, -1.0, 0.2],
        scale=S,
        gamma=[0.4, -0.3, 0.1],
    )
    x_full = np.array([0.7, -1.3, 0.2])
    ci = [0, 1]
    cond = g.condition(ci, x_full[ci])
    lhs = cond.logpdf(x_full[[2]])[0]
    rhs = g.logpdf(x_full)[0] - g.marginal(ci).logpdf(x_full[ci])[0]
    assert np.isclose(lhs, rhs)


def test_P6_05_condition_parameter_transforms():
    S = _spd(3, 4)
    lam, chi, psi = -0.5, 2.0, 1.3
    g = MultivariateGH(lam, chi, psi, [0.5, -1.0, 0.2], S, [0.4, -0.3, 0.1])
    ci = [0]
    x_c = np.array([0.7])
    cond = g.condition(ci, x_c)
    # lambda* = lambda - q/2
    assert np.isclose(cond.lambda_, lam - len(ci) / 2.0)
    # chi* = chi + d2 ; psi* = psi + g2' S22^-1 g2
    S22 = S[np.ix_(ci, ci)]
    d2 = float((x_c - g.loc_[ci]) @ np.linalg.solve(S22, (x_c - g.loc_[ci])))
    g2 = g.gamma_[ci]
    assert np.isclose(cond.chi, chi + d2)
    assert np.isclose(cond.psi, psi + float(g2 @ np.linalg.solve(S22, g2)))


def test_P6_06_marginal_keeps_lambda_chi_psi():
    S = _spd(3, 5)
    g = MultivariateGH(-0.5, 2.0, 1.3, [0.5, -1.0, 0.2], S, [0.4, -0.3, 0.1])
    m = g.marginal([0, 2])
    assert isinstance(m, MultivariateGH)
    assert (m.lambda_, m.chi, m.psi) == (g.lambda_, g.chi, g.psi)
    assert np.allclose(m.loc_, g.loc_[[0, 2]])
    assert np.allclose(m.scale_, S[np.ix_([0, 2], [0, 2])])
    assert np.allclose(m.gamma_, g.gamma_[[0, 2]])


# --------------------------------------------------------------------------
# Symmetric Student-t boundary: GH special case == direct Student-t
# --------------------------------------------------------------------------
def test_P6_07_symmetric_t_logpdf_matches_direct():
    nu = 5.0
    mu = np.array([1.0, -2.0, 0.5])
    Sig = _spd(3, 6)
    gh = MultivariateGH(
        lambda_=-nu / 2, chi=nu, psi=0.0, loc=mu, scale=Sig, gamma=[0, 0, 0]
    )
    t = MultivariateStudentT(mu, Sig, nu)
    X = np.random.default_rng(0).normal(size=(6, 3))
    assert np.allclose(gh.logpdf(X), t.logpdf(X))


def test_P6_08_symmetric_t_conditioning_matches_direct():
    """The GENERAL GH conditioning formulas, applied to the symmetric-t
    parameterization, must reproduce the direct Student-t conditional."""
    nu = 5.0
    mu = np.array([1.0, -2.0, 0.5])
    Sig = _spd(3, 7)
    gh = MultivariateGH(
        lambda_=-nu / 2, chi=nu, psi=0.0, loc=mu, scale=Sig, gamma=[0, 0, 0]
    )
    t = MultivariateStudentT(mu, Sig, nu)
    x_c = np.array([0.7])
    gc = gh.condition([0], x_c)
    tc = t.condition([0], x_c)
    # conditional df: -2 lambda* = nu + q
    assert np.isclose(-2.0 * gc.lambda_, nu + 1)
    Y = np.random.default_rng(1).normal(size=(6, 2))
    assert np.allclose(gc.logpdf(Y), tc.logpdf(Y))


# --------------------------------------------------------------------------
# Moments & sampling
# --------------------------------------------------------------------------
def test_P6_09_mean_cov_match_monte_carlo():
    S = _spd(2, 8)
    g = MultivariateGH(
        lambda_=-0.5, chi=2.0, psi=1.5, loc=[0.5, -1.0], scale=S, gamma=[0.6, -0.4]
    )
    X = g.rvs(size=400_000, random_state=0)
    assert X.shape == (400_000, 2)
    assert np.allclose(g.mean(), X.mean(axis=0), atol=0.03)
    assert np.allclose(g.cov(), np.cov(X.T), rtol=0.05, atol=0.05)


def test_P6_10_satisfies_distribution_protocol():
    g = MultivariateGH(-0.5, 1.0, 1.0, [0.0, 0.0], np.eye(2), [0.1, 0.2])
    assert isinstance(g, Distribution)


# --------------------------------------------------------------------------
# Boundary guards
# --------------------------------------------------------------------------
def test_P6_11_skew_t_boundary_not_supported():
    with pytest.raises(NotImplementedError):
        MultivariateGH(
            lambda_=-2.5, chi=5.0, psi=0.0, loc=[0.0], scale=[[1.0]], gamma=[0.5]
        )


def test_P6_12_invalid_gig_params_raise():
    with pytest.raises(ValueError):
        MultivariateGH(
            lambda_=1.0, chi=0.0, psi=0.0, loc=[0.0], scale=[[1.0]], gamma=[0.0]
        )
