"""
Phase 5 tests: ConditionalMixtureRegressor (family=...) and the
ConditionalStudentTRegressor preset.

Numbered P5-xx. Key guarantees:
  * Student-t regressor honors the sklearn contract (predict/sample/score/
    condition, clone, pipeline) with NO `family` leaking into get_params.
  * Conditional density uses the actual Student-t mixture (not a Gaussian
    surrogate), and predict_cov is heteroskedastic.
  * On heavy-tailed data the Student-t regressor achieves a better held-out
    log-likelihood than the Gaussian one.
  * The generic base dispatches on `family`; the legacy ConditionalGMMRegressor
    is untouched (no `family` in its params).
"""

import numpy as np
import pytest
from scipy.stats import t as student_t
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cgmm import (
    ConditionalMixtureRegressor,
    ConditionalStudentTRegressor,
    ConditionalGMMRegressor,
    MixtureDistribution,
    MultivariateStudentT,
)


def _heavy_tailed_data(n=600, df=3.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = 2.0 * X[:, 0] - X[:, 1] + student_t(df=df).rvs(n, random_state=seed + 1)
    return X, y


# --------------------------------------------------------------------------
def test_P5_01_fit_predict_single_and_multi_output():
    X, y = _heavy_tailed_data()
    m = ConditionalStudentTRegressor(n_components=2, random_state=0).fit(X, y)
    assert m.predict(X[:5]).shape == (5,)

    Y = np.c_[y, 0.5 * y - X[:, 0]]
    m2 = ConditionalStudentTRegressor(n_components=2, random_state=0).fit(X, Y)
    assert m2.predict(X[:5]).shape == (5, 2)


def test_P5_02_predict_cov_is_heteroskedastic():
    X, y = _heavy_tailed_data()
    m = ConditionalStudentTRegressor(n_components=1, dof=4.0, random_state=0).fit(X, y)
    cov = m.predict_cov(X[:20])
    assert cov.shape == (20, 1, 1)
    # Student-t conditional scale depends on x -> covariances vary across rows
    assert np.ptp(cov[:, 0, 0]) > 1e-6


def test_P5_03_sample_shapes():
    X, y = _heavy_tailed_data()
    m = ConditionalStudentTRegressor(n_components=2, random_state=0).fit(X, y)
    assert m.sample(X[0], n_samples=7).shape == (7, 1)
    assert m.sample(X[:4], n_samples=7).shape == (4, 7, 1)


def test_P5_04_score_equals_mean_log_prob_and_uses_t_density():
    X, y = _heavy_tailed_data()
    m = ConditionalStudentTRegressor(n_components=2, random_state=0).fit(X, y)
    assert np.isclose(m.score(X, y), np.mean(m.log_prob(X, y)))
    # log_prob must equal the conditioned Student-t mixture's logpdf
    cond = m.condition(X[:5])
    manual = np.array([cond[i].logpdf(y[i : i + 1])[0] for i in range(5)])
    assert np.allclose(m.log_prob(X[:5], y[:5]), manual)


def test_P5_05_condition_returns_student_t_mixture():
    X, y = _heavy_tailed_data()
    m = ConditionalStudentTRegressor(n_components=2, random_state=0).fit(X, y)
    cond = m.condition(X[:1])
    assert isinstance(cond, MixtureDistribution)
    assert all(isinstance(c, MultivariateStudentT) for c in cond.components_)


def test_P5_06_no_family_in_get_params_and_clone():
    m = ConditionalStudentTRegressor(n_components=4, dof="shared", random_state=1)
    params = m.get_params()
    assert "family" not in params
    assert params["n_components"] == 4 and params["dof"] == "shared"
    assert clone(m).get_params() == params


def test_P5_07_pipeline_with_scaler():
    X, y = _heavy_tailed_data()
    pipe = Pipeline(
        [
            ("sc", StandardScaler()),
            ("reg", ConditionalStudentTRegressor(n_components=2, random_state=0)),
        ]
    )
    pipe.fit(X, y)
    assert pipe.predict(X[:5]).shape == (5,)
    assert np.isfinite(pipe.score(X, y))


def test_P5_08_student_t_beats_gaussian_on_heavy_tails():
    Xtr, ytr = _heavy_tailed_data(n=800, df=2.5, seed=0)
    Xte, yte = _heavy_tailed_data(n=800, df=2.5, seed=99)
    st = ConditionalStudentTRegressor(n_components=1, random_state=0, max_iter=300).fit(
        Xtr, ytr
    )
    gm = ConditionalGMMRegressor(n_components=1, random_state=0).fit(Xtr, ytr)
    # higher (less negative) held-out mean log-likelihood for the t model
    assert st.score(Xte, yte) > gm.score(Xte, yte)


def test_P5_09_family_dispatch_and_validation():
    X, y = _heavy_tailed_data()
    g = ConditionalMixtureRegressor(
        family="gaussian", n_components=2, random_state=0
    ).fit(X, y)
    t = ConditionalMixtureRegressor(
        family="student_t", n_components=2, random_state=0
    ).fit(X, y)
    assert g.predict(X[:3]).shape == (3,)
    assert t.predict(X[:3]).shape == (3,)
    assert "family" in g.get_params()
    with pytest.raises(ValueError, match="family"):
        ConditionalMixtureRegressor(family="cauchy").fit(X, y)


def test_P5_10_responsibilities():
    X, y = _heavy_tailed_data()
    m = ConditionalStudentTRegressor(n_components=3, random_state=0).fit(X, y)
    W = m.responsibilities(X[:5])
    assert W.shape == (5, 3) and np.allclose(W.sum(axis=1), 1.0)
    G = m.responsibilities(X[:5], y[:5])
    assert G.shape == (5, 3) and np.allclose(G.sum(axis=1), 1.0)


def test_P5_11_legacy_gmm_regressor_unchanged():
    # regression guard: family must NOT appear on the legacy preset
    params = ConditionalGMMRegressor().get_params()
    assert "family" not in params
    assert "dof" not in params
