# tests/test_moe.py
import numpy as np
from cgmm.moe import MixtureOfExpertsRegressor


def test_moe_basic_fit_predict():
    """Test basic fit and predict functionality of MoE regressor."""
    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(size=(n, 2))
    y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + rng.normal(scale=0.1, size=n)

    model = MixtureOfExpertsRegressor(n_components=2, random_state=42)
    model.fit(X, y)

    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    assert np.isfinite(y_pred).all()

    # Test score
    score = model.score(X, y)
    assert np.isfinite(score)


def test_moe_multivariate():
    """Test MoE with multivariate targets."""
    rng = np.random.default_rng(42)
    n = 150
    X = rng.normal(size=(n, 3))
    y1 = 0.7 * X[:, 0] - 0.3 * X[:, 1] + rng.normal(scale=0.2, size=n)
    y2 = -0.5 * X[:, 2] + 0.1 * X[:, 0] + rng.normal(scale=0.3, size=n)
    Y = np.c_[y1, y2]

    model = MixtureOfExpertsRegressor(n_components=3, random_state=42)
    model.fit(X, Y)

    y_pred = model.predict(X)
    assert y_pred.shape == Y.shape
    assert np.isfinite(y_pred).all()


def test_moe_sample():
    """Test sampling from MoE."""
    rng = np.random.default_rng(42)
    n = 100
    X = rng.normal(size=(n, 2))
    y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.1, size=n)

    model = MixtureOfExpertsRegressor(n_components=2, random_state=42)
    model.fit(X, y)

    # Test single sample
    samples = model.sample(X[:5], n_samples=3)
    assert samples.shape == (5, 3, 1)
    assert np.isfinite(samples).all()

    # Test single input (1D array)
    samples_single = model.sample(X[0], n_samples=5)
    assert samples_single.shape == (5, 1)
    assert np.isfinite(samples_single).all()


def test_moe_compute_conditional_mixture():
    """Test _compute_conditional_mixture method."""
    rng = np.random.default_rng(42)
    n = 100
    X = rng.normal(size=(n, 2))
    y = X[:, 0] + rng.normal(scale=0.1, size=n)

    model = MixtureOfExpertsRegressor(n_components=2, random_state=42)
    model.fit(X, y)

    mixture = model._compute_conditional_mixture(X[:5])
    assert "weights" in mixture
    assert "means" in mixture
    assert "covariances" in mixture

    assert mixture["weights"].shape == (5, 2)
    assert mixture["means"].shape == (5, 2, 1)
    assert mixture["covariances"].shape == (2, 1, 1)

    # Weights should sum to 1
    assert np.allclose(mixture["weights"].sum(axis=1), 1.0)


def test_moe_responsibilities():
    """Test responsibilities method."""
    rng = np.random.default_rng(42)
    n = 100
    X = rng.normal(size=(n, 2))
    y = X[:, 0] + rng.normal(scale=0.1, size=n)

    model = MixtureOfExpertsRegressor(n_components=2, random_state=42)
    model.fit(X, y)

    # Test without y
    resp = model.responsibilities(X[:5])
    assert resp.shape == (5, 2)
    assert np.allclose(resp.sum(axis=1), 1.0)

    # Test with y
    resp_y = model.responsibilities(X[:5], y[:5])
    assert resp_y.shape == (5, 2)
    assert np.allclose(resp_y.sum(axis=1), 1.0)


def test_moe_different_configs():
    """Test different MoE configurations."""
    rng = np.random.default_rng(42)
    n = 100
    X = rng.normal(size=(n, 2))
    y = X[:, 0] + rng.normal(scale=0.1, size=n)

    # Test constant mean function
    model_const = MixtureOfExpertsRegressor(
        n_components=2, mean_function="constant", random_state=42
    )
    model_const.fit(X, y)
    y_pred_const = model_const.predict(X)
    assert y_pred_const.shape == y.shape

    # Test diagonal covariance
    model_diag = MixtureOfExpertsRegressor(
        n_components=2, covariance_type="diag", random_state=42
    )
    model_diag.fit(X, y)
    y_pred_diag = model_diag.predict(X)
    assert y_pred_diag.shape == y.shape

    # Test shared covariance
    model_shared = MixtureOfExpertsRegressor(
        n_components=2, shared_covariance=True, random_state=42
    )
    model_shared.fit(X, y)
    y_pred_shared = model_shared.predict(X)
    assert y_pred_shared.shape == y.shape
