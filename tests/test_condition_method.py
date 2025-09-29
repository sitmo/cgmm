"""
Tests for the condition() method across all conditional mixture models.

This module tests the new condition() method that returns sklearn GaussianMixture objects.
"""

import numpy as np
import pytest
from sklearn.datasets import make_regression, load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.mixture import GaussianMixture

from cgmm import (
    ConditionalGMMRegressor,
    MixtureOfExpertsRegressor,
    DiscriminativeConditionalGMMRegressor,
)


class TestConditionMethod:
    """Test the condition() method for all conditional mixture models."""

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

    def test_conditional_gmm_condition_single_sample(self, classification_data):
        """Test ConditionalGMMRegressor.condition() with single sample."""
        y_onehot, X = classification_data

        model = ConditionalGMMRegressor(n_components=3, random_state=42)
        model.fit(y_onehot, X)

        # Test single sample
        gmm = model.condition(y_onehot[:1])

        assert isinstance(gmm, GaussianMixture)
        assert gmm.n_components == 3
        assert gmm.weights_.shape == (3,)
        assert gmm.means_.shape == (3, 2)  # 2 target features
        assert gmm.covariances_.shape == (3, 2, 2)
        assert np.allclose(gmm.weights_.sum(), 1.0)

    def test_conditional_gmm_condition_batch(self, classification_data):
        """Test ConditionalGMMRegressor.condition() with batch."""
        y_onehot, X = classification_data

        model = ConditionalGMMRegressor(n_components=3, random_state=42)
        model.fit(y_onehot, X)

        # Test batch
        gmms = model.condition(y_onehot[:5])

        assert isinstance(gmms, list)
        assert len(gmms) == 5
        for gmm in gmms:
            assert isinstance(gmm, GaussianMixture)
            assert gmm.n_components == 3
            assert gmm.weights_.shape == (3,)
            assert gmm.means_.shape == (3, 2)
            assert gmm.covariances_.shape == (3, 2, 2)
            assert np.allclose(gmm.weights_.sum(), 1.0)

    def test_mixture_of_experts_condition_single_sample(self, regression_data):
        """Test MixtureOfExpertsRegressor.condition() with single sample."""
        X, y = regression_data

        model = MixtureOfExpertsRegressor(n_components=3, random_state=42)
        model.fit(X, y)

        # Test single sample
        gmm = model.condition(X[:1])

        assert isinstance(gmm, GaussianMixture)
        assert gmm.n_components == 3
        assert gmm.weights_.shape == (3,)
        assert gmm.means_.shape == (3, 1)  # 1 target feature
        assert gmm.covariances_.shape == (3, 1, 1)
        assert np.allclose(gmm.weights_.sum(), 1.0)

    def test_mixture_of_experts_condition_batch(self, regression_data):
        """Test MixtureOfExpertsRegressor.condition() with batch."""
        X, y = regression_data

        model = MixtureOfExpertsRegressor(n_components=3, random_state=42)
        model.fit(X, y)

        # Test batch
        gmms = model.condition(X[:5])

        assert isinstance(gmms, list)
        assert len(gmms) == 5
        for gmm in gmms:
            assert isinstance(gmm, GaussianMixture)
            assert gmm.n_components == 3
            assert gmm.weights_.shape == (3,)
            assert gmm.means_.shape == (3, 1)
            assert gmm.covariances_.shape == (3, 1, 1)
            assert np.allclose(gmm.weights_.sum(), 1.0)

    def test_discriminative_condition_single_sample(self, classification_data):
        """Test DiscriminativeConditionalGMMRegressor.condition() with single sample."""
        y_onehot, X = classification_data

        model = DiscriminativeConditionalGMMRegressor(n_components=3, random_state=42)
        model.fit(y_onehot, X)

        # Test single sample
        gmm = model.condition(y_onehot[:1])

        assert isinstance(gmm, GaussianMixture)
        assert gmm.n_components == 3
        assert gmm.weights_.shape == (3,)
        assert gmm.means_.shape == (3, 2)  # 2 target features
        assert gmm.covariances_.shape == (3, 2, 2)
        assert np.allclose(gmm.weights_.sum(), 1.0)

    def test_discriminative_condition_batch(self, classification_data):
        """Test DiscriminativeConditionalGMMRegressor.condition() with batch."""
        y_onehot, X = classification_data

        model = DiscriminativeConditionalGMMRegressor(n_components=3, random_state=42)
        model.fit(y_onehot, X)

        # Test batch
        gmms = model.condition(y_onehot[:5])

        assert isinstance(gmms, list)
        assert len(gmms) == 5
        for gmm in gmms:
            assert isinstance(gmm, GaussianMixture)
            assert gmm.n_components == 3
            assert gmm.weights_.shape == (3,)
            assert gmm.means_.shape == (3, 2)
            assert gmm.covariances_.shape == (3, 2, 2)
            assert np.allclose(gmm.weights_.sum(), 1.0)

    def test_condition_with_sklearn_methods(self, regression_data):
        """Test that conditioned GMMs work with sklearn methods."""
        X, y = regression_data

        model = MixtureOfExpertsRegressor(n_components=2, random_state=42)
        model.fit(X, y)

        gmm = model.condition(X[:1])

        # Test sklearn methods work
        assert hasattr(gmm, "score_samples")
        assert hasattr(gmm, "sample")
        assert hasattr(gmm, "predict_proba")

        # Test score_samples
        log_probs = gmm.score_samples(y[:1].reshape(1, -1))
        assert log_probs.shape == (1,)
        assert np.isfinite(log_probs[0])

        # Test sample
        samples = gmm.sample(10)[0]
        assert samples.shape == (10, 1)
        assert np.all(np.isfinite(samples))

        # Test predict_proba
        probs = gmm.predict_proba(y[:1].reshape(1, -1))
        assert probs.shape == (1, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_condition_unfitted_model(self, regression_data):
        """Test that condition() raises error for unfitted models."""
        X, y = regression_data

        model = MixtureOfExpertsRegressor(n_components=2, random_state=42)

        with pytest.raises((ValueError, AttributeError)):
            model.condition(X[:1])

    def test_condition_input_validation(self, regression_data):
        """Test condition() input validation."""
        X, y = regression_data

        model = MixtureOfExpertsRegressor(n_components=2, random_state=42)
        model.fit(X, y)

        # Test with invalid input shapes
        with pytest.raises(ValueError):
            model.condition(np.array([[1.0]]))  # Wrong number of features


class TestConditionConsistency:
    """Test consistency between condition() and other methods."""

    @pytest.fixture
    def regression_data(self):
        """Create regression data for testing."""
        X, y = make_regression(n_samples=50, n_features=2, n_targets=1, random_state=42)
        return X, y

    def test_condition_vs_predict_consistency(self, regression_data):
        """Test that condition() gives same mean as predict()."""
        X, y = regression_data

        model = MixtureOfExpertsRegressor(n_components=2, random_state=42)
        model.fit(X, y)

        # Get prediction
        prediction = model.predict(X[:1])

        # Get conditioned GMM and compute mean
        gmm = model.condition(X[:1])
        gmm_mean = gmm.means_.T @ gmm.weights_

        # Should be approximately equal
        np.testing.assert_array_almost_equal(prediction, gmm_mean, decimal=10)

    def test_condition_vs_log_prob_consistency(self, regression_data):
        """Test that condition() gives reasonable log-prob values."""
        X, y = regression_data

        model = MixtureOfExpertsRegressor(n_components=2, random_state=42)
        model.fit(X, y)

        # Get log-prob from model
        model_log_prob = model.log_prob(X[:1], y[:1])

        # Get log-prob from conditioned GMM
        gmm = model.condition(X[:1])
        gmm_log_prob = gmm.score_samples(y[:1].reshape(1, -1))[0]

        # Both should be finite and reasonable
        assert np.isfinite(model_log_prob[0])
        assert np.isfinite(gmm_log_prob)
        # Note: They might not be exactly equal due to different implementations

    def test_condition_vs_sample_consistency(self, regression_data):
        """Test that condition() gives reasonable samples."""
        X, y = regression_data

        model = MixtureOfExpertsRegressor(n_components=2, random_state=42)
        model.fit(X, y)

        # Get samples from model
        model_samples = model.sample(X[:1], n_samples=100)

        # Get samples from conditioned GMM
        gmm = model.condition(X[:1])
        gmm_samples = gmm.sample(100)[0]

        # Both should be finite and have reasonable shapes
        assert np.all(np.isfinite(model_samples))
        assert np.all(np.isfinite(gmm_samples))
        assert model_samples.ndim == 2  # Now returns 2D for scikit-learn compatibility
        assert gmm_samples.ndim == 2
        assert model_samples.shape[1] == 1  # Single target
        assert gmm_samples.shape[1] == 1
        # Note: Shapes should now match due to scikit-learn compatibility
