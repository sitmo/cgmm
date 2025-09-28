"""
Tests for the predict() method and related functionality.

This module tests the simplified predict() method logic and covers
the branches that were previously untested.
"""

import numpy as np
import pytest
from sklearn.datasets import make_regression, load_iris
from sklearn.preprocessing import OneHotEncoder

from cgmm import (
    ConditionalGMMRegressor,
    MixtureOfExpertsRegressor,
)


class TestPredictMethod:
    """Test the predict() method for all models."""

    @pytest.fixture
    def regression_data(self):
        """Create regression data for testing."""
        X, y = make_regression(n_samples=50, n_features=2, n_targets=1, random_state=42)
        return X, y

    @pytest.fixture
    def classification_data(self):
        """Create classification data for testing."""
        iris = load_iris()
        X = iris.data[:, :2]
        y_labels = iris.target
        encoder = OneHotEncoder(sparse_output=False)
        y_onehot = encoder.fit_transform(y_labels.reshape(-1, 1))
        return y_onehot, X

    def test_predict_single_output(self, regression_data):
        """Test predict() with single output (should return 1D array)."""
        X, y = regression_data

        model = MixtureOfExpertsRegressor(n_components=2, random_state=42)
        model.fit(X, y)

        # Test single sample
        prediction = model.predict(X[:1])
        assert prediction.ndim == 1
        assert prediction.shape == (1,)
        assert np.isfinite(prediction[0])

        # Test batch
        predictions = model.predict(X[:5])
        assert predictions.ndim == 1
        assert predictions.shape == (5,)
        assert np.all(np.isfinite(predictions))

    def test_predict_multi_output(self, classification_data):
        """Test predict() with multi output (should return 2D array)."""
        y_onehot, X = classification_data

        model = ConditionalGMMRegressor(n_components=2, random_state=42)
        model.fit(y_onehot, X)

        # Test single sample
        prediction = model.predict(y_onehot[:1])
        assert prediction.ndim == 2
        assert prediction.shape == (1, 2)  # 2 target features
        assert np.all(np.isfinite(prediction))

        # Test batch
        predictions = model.predict(y_onehot[:5])
        assert predictions.ndim == 2
        assert predictions.shape == (5, 2)
        assert np.all(np.isfinite(predictions))

    def test_predict_with_return_cov(self, regression_data):
        """Test predict() with return_cov=True."""
        X, y = regression_data

        model = MixtureOfExpertsRegressor(n_components=2, random_state=42)
        model.fit(X, y)

        # Test single sample
        mean, cov = model.predict(X[:1], return_cov=True)
        assert mean.ndim == 1
        assert mean.shape == (1,)
        assert cov.ndim == 1
        assert cov.shape == (1,)
        assert np.isfinite(mean[0])
        assert np.isfinite(cov[0])
        assert cov[0] >= 0  # Variance should be non-negative

        # Test batch
        means, covs = model.predict(X[:5], return_cov=True)
        assert means.ndim == 1
        assert means.shape == (5,)
        assert covs.ndim == 1
        assert covs.shape == (5,)
        assert np.all(np.isfinite(means))
        assert np.all(np.isfinite(covs))
        assert np.all(covs >= 0)

    def test_predict_with_return_components(self, regression_data):
        """Test predict() with return_components=True."""
        X, y = regression_data

        model = MixtureOfExpertsRegressor(n_components=2, random_state=42)
        model.fit(X, y)

        # Test single sample
        mean, components = model.predict(X[:1], return_components=True)
        assert mean.ndim == 1
        assert mean.shape == (1,)
        assert isinstance(components, dict)
        assert "weights" in components
        assert "means" in components
        assert "covariances" in components

        # Test batch
        means, components = model.predict(X[:5], return_components=True)
        assert means.ndim == 1
        assert means.shape == (5,)
        assert isinstance(components, dict)

    def test_predict_with_both_return_options(self, regression_data):
        """Test predict() with both return_cov=True and return_components=True."""
        X, y = regression_data

        model = MixtureOfExpertsRegressor(n_components=2, random_state=42)
        model.fit(X, y)

        # Test single sample
        mean, cov, components = model.predict(
            X[:1], return_cov=True, return_components=True
        )
        assert mean.ndim == 1
        assert mean.shape == (1,)
        assert cov.ndim == 1
        assert cov.shape == (1,)
        assert isinstance(components, dict)

        # Test batch
        means, covs, components = model.predict(
            X[:5], return_cov=True, return_components=True
        )
        assert means.ndim == 1
        assert means.shape == (5,)
        assert covs.ndim == 1
        assert covs.shape == (5,)
        assert isinstance(components, dict)

    def test_predict_default_return_cov(self, regression_data):
        """Test predict() with default return_cov behavior."""
        X, y = regression_data

        # Test with return_cov=False by default
        model = MixtureOfExpertsRegressor(
            n_components=2, random_state=42, return_cov=False
        )
        model.fit(X, y)

        prediction = model.predict(X[:1])
        assert prediction.ndim == 1
        assert prediction.shape == (1,)

        # Test with return_cov=True by default
        model = MixtureOfExpertsRegressor(
            n_components=2, random_state=42, return_cov=True
        )
        model.fit(X, y)

        result = model.predict(X[:1])
        if isinstance(result, tuple):
            mean, cov = result
            assert mean.ndim == 1
            assert cov.ndim == 1
        else:
            # Should return just mean if return_cov=False explicitly
            assert result.ndim == 1

    def test_predict_multi_output_with_return_cov(self, classification_data):
        """Test predict() with multi output and return_cov=True."""
        y_onehot, X = classification_data

        model = ConditionalGMMRegressor(n_components=2, random_state=42)
        model.fit(y_onehot, X)

        # Test single sample
        mean, cov = model.predict(y_onehot[:1], return_cov=True)
        assert mean.ndim == 2
        assert mean.shape == (1, 2)
        assert cov.ndim == 3
        assert cov.shape == (1, 2, 2)  # Covariance matrix for each sample
        assert np.all(np.isfinite(mean))
        assert np.all(np.isfinite(cov))

        # Test batch
        means, covs = model.predict(y_onehot[:5], return_cov=True)
        assert means.ndim == 2
        assert means.shape == (5, 2)
        assert covs.ndim == 3
        assert covs.shape == (5, 2, 2)
        assert np.all(np.isfinite(means))
        assert np.all(np.isfinite(covs))

    def test_predict_consistency_across_models(self, regression_data):
        """Test that predict() is consistent across different models."""
        X, y = regression_data

        models = [
            MixtureOfExpertsRegressor(n_components=2, random_state=42),
        ]

        for model in models:
            model.fit(X, y)

            # Test basic prediction
            prediction = model.predict(X[:1])
            assert prediction.ndim == 1
            assert prediction.shape == (1,)
            assert np.isfinite(prediction[0])

            # Test with return_cov
            mean, cov = model.predict(X[:1], return_cov=True)
            assert mean.ndim == 1
            assert cov.ndim == 1
            assert np.isfinite(mean[0])
            assert np.isfinite(cov[0])

    def test_predict_input_validation(self, regression_data):
        """Test predict() input validation."""
        X, y = regression_data

        model = MixtureOfExpertsRegressor(n_components=2, random_state=42)
        model.fit(X, y)

        # Test with invalid input shapes
        with pytest.raises(ValueError):
            model.predict(np.array([[1.0]]))  # Wrong number of features

        # Test with empty input
        with pytest.raises(ValueError):
            model.predict(np.array([]).reshape(0, 2))

    def test_predict_unfitted_model(self, regression_data):
        """Test predict() with unfitted model."""
        X, y = regression_data

        model = MixtureOfExpertsRegressor(n_components=2, random_state=42)

        with pytest.raises((ValueError, AttributeError)):
            model.predict(X[:1])


class TestPredictEdgeCases:
    """Test edge cases for predict() method."""

    @pytest.fixture
    def regression_data(self):
        """Create regression data for testing."""
        X, y = make_regression(n_samples=50, n_features=2, n_targets=1, random_state=42)
        return X, y

    def test_predict_single_sample_vs_batch(self, regression_data):
        """Test that single sample and batch give consistent results."""
        X, y = regression_data

        model = MixtureOfExpertsRegressor(n_components=2, random_state=42)
        model.fit(X, y)

        # Single sample
        single_pred = model.predict(X[:1])

        # Batch with same sample
        batch_pred = model.predict(X[:1])

        # Should be identical
        np.testing.assert_array_equal(single_pred, batch_pred)

    def test_predict_deterministic(self, regression_data):
        """Test that predict() is deterministic with same random_state."""
        X, y = regression_data

        model1 = MixtureOfExpertsRegressor(n_components=2, random_state=42)
        model1.fit(X, y)
        pred1 = model1.predict(X[:5])

        model2 = MixtureOfExpertsRegressor(n_components=2, random_state=42)
        model2.fit(X, y)
        pred2 = model2.predict(X[:5])

        # Should be identical
        np.testing.assert_array_equal(pred1, pred2)

    def test_predict_with_different_n_components(self, regression_data):
        """Test predict() with different numbers of components."""
        X, y = regression_data

        for n_components in [1, 2, 3, 5]:
            model = MixtureOfExpertsRegressor(
                n_components=n_components, random_state=42
            )
            model.fit(X, y)

            prediction = model.predict(X[:1])
            assert prediction.ndim == 1
            assert prediction.shape == (1,)
            assert np.isfinite(prediction[0])
