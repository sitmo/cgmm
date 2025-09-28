"""
Tests for parameter validation across all models.

This module tests the simplified parameter validation that was introduced
to improve test coverage.
"""

import pytest

from cgmm import (
    ConditionalGMMRegressor,
    MixtureOfExpertsRegressor,
    DiscriminativeConditionalGMMRegressor,
)


class TestParameterValidation:
    """Test parameter validation for all models."""

    def test_conditional_gmm_parameter_validation(self):
        """Test ConditionalGMMRegressor parameter validation."""
        # Valid parameters should work
        model = ConditionalGMMRegressor(n_components=3, random_state=42)
        assert model.n_components == 3

        # ConditionalGMMRegressor uses sklearn's validation, so test that it works
        # with valid parameters and handles invalid ones appropriately
        model = ConditionalGMMRegressor(n_components=1, random_state=0)
        assert model.n_components == 1

    def test_mixture_of_experts_parameter_validation(self):
        """Test MixtureOfExpertsRegressor parameter validation."""
        # Valid parameters should work
        model = MixtureOfExpertsRegressor(n_components=3, random_state=42)
        assert model.n_components == 3

        # Invalid n_components should raise error
        with pytest.raises(ValueError, match="n_components must be positive"):
            MixtureOfExpertsRegressor(n_components=0)

        with pytest.raises(ValueError, match="n_components must be positive"):
            MixtureOfExpertsRegressor(n_components=-1)

        # Test type conversion
        model = MixtureOfExpertsRegressor(
            n_components=3.0
        )  # float should convert to int
        assert model.n_components == 3

    def test_discriminative_parameter_validation(self):
        """Test DiscriminativeConditionalGMMRegressor parameter validation."""
        # Valid parameters should work
        model = DiscriminativeConditionalGMMRegressor(n_components=3, random_state=42)
        assert model.n_components == 3

        # Invalid n_components should raise error
        with pytest.raises(ValueError, match="n_components must be positive"):
            DiscriminativeConditionalGMMRegressor(n_components=0)

        with pytest.raises(ValueError, match="n_components must be positive"):
            DiscriminativeConditionalGMMRegressor(n_components=-1)

        # Test type conversion
        model = DiscriminativeConditionalGMMRegressor(
            n_components=3.0
        )  # float should convert to int
        assert model.n_components == 3

    def test_parameter_type_conversion(self):
        """Test that parameters are properly converted to correct types."""
        # Test n_components conversion
        model = MixtureOfExpertsRegressor(n_components=3.7)
        assert isinstance(model.n_components, int)
        assert model.n_components == 3

        # Test other parameter conversions
        model = MixtureOfExpertsRegressor(
            n_components=2, reg_covar=1.5, gating_penalty=0.1, verbose=0
        )
        assert isinstance(model.reg_covar, float)
        assert isinstance(model.gating_penalty, float)
        assert isinstance(model.verbose, int)

    def test_parameter_assignment(self):
        """Test that all parameters are properly assigned."""
        model = MixtureOfExpertsRegressor(
            n_components=2,
            covariance_type="full",
            shared_covariance=True,
            mean_function="affine",
            reg_covar=1e-6,
            gating_penalty=1e-2,
            gating_max_iter=50,
            gating_penalty_bias=0.1,
            gating_tol=1e-6,
            gating_init_scale=1e-1,
            max_iter=200,
            tol=1e-4,
            n_init=1,
            init_params="kmeans",
            random_state=42,
            return_cov=False,
            verbose=0,
        )

        # Check all parameters are assigned
        assert model.n_components == 2
        assert model.covariance_type == "full"
        assert model.shared_covariance is True
        assert model.mean_function == "affine"
        assert model.reg_covar == 1e-6
        assert model.gating_penalty == 1e-2
        assert model.gating_max_iter == 50
        assert model.gating_penalty_bias == 0.1
        assert model.gating_tol == 1e-6
        assert model.gating_init_scale == 1e-1
        assert model.max_iter == 200
        assert model.tol == 1e-4
        assert model.n_init == 1
        assert model.init_params == "kmeans"
        assert model.random_state == 42
        assert model.return_cov is False
        assert model.verbose == 0


class TestEdgeCases:
    """Test edge cases for parameter validation."""

    def test_minimum_valid_parameters(self):
        """Test minimum valid parameter values."""
        # n_components = 1 should be valid
        model = MixtureOfExpertsRegressor(n_components=1)
        assert model.n_components == 1

        # Test with minimal parameters
        model = MixtureOfExpertsRegressor(n_components=1, random_state=0)
        assert model.n_components == 1
        assert model.random_state == 0

    def test_large_parameter_values(self):
        """Test large parameter values."""
        # Large n_components should be valid
        model = MixtureOfExpertsRegressor(n_components=100)
        assert model.n_components == 100

        # Large other parameters should be valid
        model = MixtureOfExpertsRegressor(
            n_components=10, max_iter=10000, gating_max_iter=1000, verbose=10
        )
        assert model.max_iter == 10000
        assert model.gating_max_iter == 1000
        assert model.verbose == 10

    def test_float_parameter_conversion(self):
        """Test that float parameters are properly converted."""
        model = MixtureOfExpertsRegressor(
            n_components=2.0,  # Should convert to int
            reg_covar=1.5,  # Should stay float
            gating_penalty=0.1,  # Should stay float
            verbose=0.0,  # Should convert to int
        )

        assert isinstance(model.n_components, int)
        assert model.n_components == 2
        assert isinstance(model.reg_covar, float)
        assert model.reg_covar == 1.5
        assert isinstance(model.gating_penalty, float)
        assert model.gating_penalty == 0.1
        assert isinstance(model.verbose, int)
        assert model.verbose == 0


class TestErrorMessages:
    """Test error messages for invalid parameters."""

    def test_n_components_error_messages(self):
        """Test error messages for invalid n_components."""
        with pytest.raises(ValueError) as exc_info:
            MixtureOfExpertsRegressor(n_components=0)
        assert "n_components must be positive" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            MixtureOfExpertsRegressor(n_components=-5)
        assert "n_components must be positive" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            DiscriminativeConditionalGMMRegressor(n_components=0)
        assert "n_components must be positive" in str(exc_info.value)

    def test_type_conversion_errors(self):
        """Test that invalid types raise appropriate errors."""
        # String n_components should raise TypeError
        with pytest.raises((TypeError, ValueError)):
            MixtureOfExpertsRegressor(n_components="invalid")

        # None n_components should raise TypeError
        with pytest.raises((TypeError, ValueError)):
            MixtureOfExpertsRegressor(n_components=None)
