"""
Phase 4 tests: generalized MixtureConditioner (GMMConditioner alias).

Numbered P4-xx. Key guarantees:
  * GMMConditioner is exactly MixtureConditioner (alias), constructor unchanged.
  * Gaussian input keeps the cached fast path and returns a
    GaussianMixtureDistribution (isinstance GaussianMixture stays True).
  * A fitted StudentTMixture conditions through the same conditioner and returns a
    MixtureDistribution of MultivariateStudentT, matching the per-component
    MultivariateStudentT.condition formulas.
  * Batch conditioning returns a list; bad inputs raise.
"""

import numpy as np
import pytest
from scipy.stats import multivariate_t
from sklearn.mixture import GaussianMixture

from cgmm import (
    GMMConditioner,
    MixtureConditioner,
    StudentTMixture,
    MixtureDistribution,
    MultivariateStudentT,
    MultivariateNormal,
    GaussianMixtureDistribution,
)


def _joint_t(df=5.0, n=400, seed=0):
    a = multivariate_t(loc=[0, 0, 0], shape=np.eye(3), df=df).rvs(n, random_state=seed)
    b = multivariate_t(loc=[6, 6, 6], shape=np.eye(3), df=df).rvs(
        n, random_state=seed + 1
    )
    return np.vstack([a, b])


def _gauss_joint(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3))


# --------------------------------------------------------------------------
def test_P4_01_gmmconditioner_is_alias():
    assert GMMConditioner is MixtureConditioner


def test_P4_02_gaussian_path_returns_gmd():
    gmm = GaussianMixture(n_components=3, random_state=0).fit(_gauss_joint())
    out = MixtureConditioner(gmm, cond_idx=[0]).precompute().condition([0.4])
    assert isinstance(out, GaussianMixtureDistribution)
    assert isinstance(out, GaussianMixture)  # keystone preserved


def test_P4_03_gaussian_path_matches_handbuilt_mixture():
    """The cached Gaussian path equals the generic MixtureDistribution path on the
    same Gaussian components."""
    gmm = GaussianMixture(n_components=3, random_state=0).fit(_gauss_joint())
    x = np.array([0.3])
    fast = MixtureConditioner(gmm, cond_idx=[0]).precompute().condition(x)
    md = MixtureDistribution(
        gmm.weights_,
        [MultivariateNormal(m, S) for m, S in zip(gmm.means_, gmm.covariances_)],
    )
    generic = md.condition([0], x)
    assert np.allclose(fast.weights_, generic.weights_)
    means_g = np.array([c.mean() for c in generic.components_])
    assert np.allclose(fast.means_, means_g)


def test_P4_04_student_t_conditioning_returns_student_t_mixture():
    st = StudentTMixture(
        n_components=2, init_params="gmm", random_state=0, max_iter=200
    ).fit(_joint_t())
    out = MixtureConditioner(st, cond_idx=[0]).precompute().condition([0.4])
    assert isinstance(out, MixtureDistribution)
    assert all(isinstance(c, MultivariateStudentT) for c in out.components_)
    # target block has dimension 2 (conditioned on 1 of 3)
    assert out.dim == 2


def test_P4_05_student_t_matches_per_component_condition():
    st = StudentTMixture(
        n_components=2, init_params="gmm", random_state=0, max_iter=200
    ).fit(_joint_t())
    x = np.array([0.4])
    out = MixtureConditioner(st, cond_idx=[0]).precompute().condition(x)
    ref = st.to_mixture_distribution().condition([0], x)
    assert np.allclose(out.weights_, ref.weights_)
    for c_out, c_ref in zip(out.components_, ref.components_):
        assert c_out.df == c_ref.df
        assert np.allclose(c_out.loc_, c_ref.loc_)
        assert np.allclose(c_out.scale_, c_ref.scale_)


def test_P4_06_student_t_conditional_heteroskedastic():
    """The Student-t conditional scale depends on the conditioning value (unlike
    the Gaussian), so conditioning on an outlier inflates the scale."""
    st = StudentTMixture(n_components=1, dof=4.0, random_state=0, max_iter=200).fit(
        _joint_t()
    )
    cond = MixtureConditioner(st, cond_idx=[0]).precompute()
    near = cond.condition([st.locations_[0, 0]])  # at the center
    far = cond.condition([st.locations_[0, 0] + 20.0])  # far outlier
    s_near = np.trace(near.components_[0].scale_)
    s_far = np.trace(far.components_[0].scale_)
    assert s_far > s_near


def test_P4_07_batch_returns_list_both_paths():
    gmm = GaussianMixture(n_components=2, random_state=0).fit(_gauss_joint())
    out_g = MixtureConditioner(gmm, cond_idx=[0]).precompute().condition([[0.1], [0.2]])
    assert isinstance(out_g, list) and len(out_g) == 2

    st = StudentTMixture(n_components=2, random_state=0, max_iter=100).fit(_joint_t())
    out_t = MixtureConditioner(st, cond_idx=[0]).precompute().condition([[0.1], [0.2]])
    assert isinstance(out_t, list) and len(out_t) == 2
    assert all(isinstance(o, MixtureDistribution) for o in out_t)


def test_P4_08_invalid_estimator_raises_typeerror():
    with pytest.raises(TypeError):
        MixtureConditioner("not a mixture", cond_idx=[0]).precompute()


def test_P4_09_wrong_x_shape_raises():
    st = StudentTMixture(n_components=2, random_state=0, max_iter=50).fit(_joint_t())
    cond = MixtureConditioner(st, cond_idx=[0]).precompute()
    with pytest.raises(ValueError):
        cond.condition([[0.1, 0.2]])  # expects 1 conditioning column, got 2


def test_P4_10_conditioning_consistent_with_score_samples():
    """For a Student-t mixture, conditioning then evaluating logpdf is consistent
    with the joint identity log p(y|x) = log p(x,y) - log p(x)."""
    st = StudentTMixture(
        n_components=2, init_params="gmm", random_state=0, max_iter=200
    ).fit(_joint_t())
    x = np.array([0.5])
    y = np.array([0.3, -0.2])
    cond = MixtureConditioner(st, cond_idx=[0]).precompute().condition(x)
    lhs = cond.logpdf(y)[0]
    joint = st.to_mixture_distribution()
    # marginal mixture over the conditioning block: p(x) = sum_k w_k f_k^(2)(x)
    marg = MixtureDistribution(
        st.weights_, [c.marginal([0]) for c in joint.components_]
    )
    rhs = joint.logpdf(np.r_[x, y])[0] - marg.logpdf(x)[0]
    assert np.isclose(lhs, rhs, atol=1e-6)
