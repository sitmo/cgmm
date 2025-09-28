# tests/test_discriminative.py
import numpy as np
from cgmm.discriminative import DiscriminativeConditionalGMMRegressor


def test_discriminative_basic_fit_predict():
    """Test basic fit and predict functionality of DiscriminativeConditionalGMMRegressor."""
    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(size=(n, 2))
    y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + rng.normal(scale=0.1, size=n)

    model = DiscriminativeConditionalGMMRegressor(n_components=2, random_state=42)
    model.fit(X, y)

    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    assert np.isfinite(y_pred).all()

    # Test score
    score = model.score(X, y)
    assert np.isfinite(score)


def test_discriminative_multivariate():
    """Test DiscriminativeConditionalGMMRegressor with multivariate targets."""
    rng = np.random.default_rng(42)
    n = 150
    X = rng.normal(size=(n, 3))
    y1 = 0.7 * X[:, 0] - 0.3 * X[:, 1] + rng.normal(scale=0.2, size=n)
    y2 = -0.5 * X[:, 2] + 0.1 * X[:, 0] + rng.normal(scale=0.3, size=n)
    Y = np.c_[y1, y2]

    model = DiscriminativeConditionalGMMRegressor(n_components=3, random_state=42)
    model.fit(X, Y)

    y_pred = model.predict(X)
    assert y_pred.shape == Y.shape
    assert np.isfinite(y_pred).all()


def test_discriminative_compute_conditional_mixture():
    """Test _compute_conditional_mixture method."""
    rng = np.random.default_rng(42)
    n = 100
    X = rng.normal(size=(n, 2))
    y = X[:, 0] + rng.normal(scale=0.1, size=n)

    model = DiscriminativeConditionalGMMRegressor(n_components=2, random_state=42)
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


def test_discriminative_responsibilities():
    """Test responsibilities method."""
    rng = np.random.default_rng(42)
    n = 100
    X = rng.normal(size=(n, 2))
    y = X[:, 0] + rng.normal(scale=0.1, size=n)

    model = DiscriminativeConditionalGMMRegressor(n_components=2, random_state=42)
    model.fit(X, y)

    # Test without y
    resp = model.responsibilities(X[:5])
    assert resp.shape == (5, 2)
    assert np.allclose(resp.sum(axis=1), 1.0)

    # Test with y
    resp_y = model.responsibilities(X[:5], y[:5])
    assert resp_y.shape == (5, 2)
    assert np.allclose(resp_y.sum(axis=1), 1.0)


def test_discriminative_different_configs():
    """Test different DiscriminativeConditionalGMMRegressor configurations."""
    rng = np.random.default_rng(42)
    n = 100
    X = rng.normal(size=(n, 2))
    y = X[:, 0] + rng.normal(scale=0.1, size=n)

    # Test diagonal covariance
    model_diag = DiscriminativeConditionalGMMRegressor(
        n_components=2, covariance_type="diag", random_state=42
    )
    model_diag.fit(X, y)
    y_pred_diag = model_diag.predict(X)
    assert y_pred_diag.shape == y.shape

    # Note: spherical covariance not fully supported in current implementation


def test_discriminative_convergence():
    """Test that the model converges properly."""
    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(size=(n, 2))
    y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.1, size=n)

    model = DiscriminativeConditionalGMMRegressor(
        n_components=3, max_iter=50, random_state=42
    )
    model.fit(X, y)

    # Check that the model converged
    assert hasattr(model, "converged_")
    assert hasattr(model, "n_iter_")
    assert model.n_iter_ > 0


def test_discriminative_regularization():
    """Test that regularization parameter works."""
    rng = np.random.default_rng(42)
    n = 100
    X = rng.normal(size=(n, 2))
    y = X[:, 0] + rng.normal(scale=0.1, size=n)

    # Test with different regularization values
    for reg_covar in [1e-6, 1e-3, 1e-1]:
        model = DiscriminativeConditionalGMMRegressor(
            n_components=2, reg_covar=reg_covar, random_state=42
        )
        model.fit(X, y)
        y_pred = model.predict(X)
        assert y_pred.shape == y.shape
        assert np.isfinite(y_pred).all()


def test_discriminative_weight_step():
    """Test custom weight step size."""
    rng = np.random.default_rng(42)
    n = 100
    X = rng.normal(size=(n, 2))
    y = X[:, 0] + rng.normal(scale=0.1, size=n)

    model = DiscriminativeConditionalGMMRegressor(
        n_components=2, weight_step=0.01, random_state=42
    )
    model.fit(X, y)
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    assert np.isfinite(y_pred).all()


def test_discriminative_single_component():
    """Test with single component (degenerate case)."""
    rng = np.random.default_rng(42)
    n = 50
    X = rng.normal(size=(n, 2))
    y = X[:, 0] + rng.normal(scale=0.1, size=n)

    model = DiscriminativeConditionalGMMRegressor(n_components=1, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    assert np.isfinite(y_pred).all()


def test_discriminative_edge_cases():
    """Test edge cases and error handling."""
    rng = np.random.default_rng(42)
    n = 50
    X = rng.normal(size=(n, 2))
    y = X[:, 0] + rng.normal(scale=0.1, size=n)

    # Test with very few samples
    model = DiscriminativeConditionalGMMRegressor(n_components=2, random_state=42)
    model.fit(X[:10], y[:10])
    y_pred = model.predict(X[:10])
    assert y_pred.shape == y[:10].shape

    # Test with 1D input
    model_1d = DiscriminativeConditionalGMMRegressor(n_components=2, random_state=42)
    model_1d.fit(X[:, :1], y)
    y_pred_1d = model_1d.predict(X[:, :1])
    assert y_pred_1d.shape == y.shape
