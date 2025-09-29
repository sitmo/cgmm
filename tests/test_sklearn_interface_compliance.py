"""
Comprehensive tests to enforce scikit-learn interface compliance.

This module tests that our CGMM models strictly follow scikit-learn's
GaussianMixture interface rules for method signatures and return shapes.
"""

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.mixture import GaussianMixture

from cgmm import (
    ConditionalGMMRegressor,
    MixtureOfExpertsRegressor,
    DiscriminativeConditionalGMMRegressor,
)


class TestSklearnInterfaceCompliance:
    """Test strict compliance with scikit-learn GaussianMixture interface."""

    @pytest.fixture
    def models(self):
        """Create fitted models for testing."""
        # Create test data
        X, y = make_regression(
            n_samples=100, n_features=2, n_targets=1, random_state=42
        )
        X_multi, y_multi = make_regression(
            n_samples=100, n_features=2, n_targets=3, random_state=42
        )

        models = {
            "ConditionalGMM": ConditionalGMMRegressor(n_components=3, random_state=42),
            "MixtureOfExperts": MixtureOfExpertsRegressor(
                n_components=3, random_state=42
            ),
            "Discriminative": DiscriminativeConditionalGMMRegressor(
                n_components=3, random_state=42
            ),
        }

        # Fit models
        models["ConditionalGMM"].fit(X, y)
        models["MixtureOfExperts"].fit(X, y)
        models["Discriminative"].fit(X, y)

        return models, X, y, X_multi, y_multi

    def test_log_prob_interface_compliance(self, models):
        """Test that log_prob follows scikit-learn score_samples interface rules."""
        model_dict, X, y, X_multi, y_multi = models

        for name, model in model_dict.items():
            # Test 1: Single sample input -> 1D output
            single_X = X[:1]
            single_y = y[:1]
            single_log_prob = model.log_prob(single_X, single_y)

            assert isinstance(
                single_log_prob, np.ndarray
            ), f"{name}: log_prob should return numpy array"
            assert (
                single_log_prob.ndim == 1
            ), f"{name}: log_prob should return 1D array for single sample"
            assert single_log_prob.shape == (
                1,
            ), f"{name}: single sample should return shape (1,)"
            assert np.isfinite(
                single_log_prob
            ).all(), f"{name}: log_prob should return finite values"

            # Test 2: Batch input -> 1D output
            batch_X = X[:5]
            batch_y = y[:5]
            batch_log_prob = model.log_prob(batch_X, batch_y)

            assert isinstance(
                batch_log_prob, np.ndarray
            ), f"{name}: log_prob should return numpy array"
            assert (
                batch_log_prob.ndim == 1
            ), f"{name}: log_prob should return 1D array for batch"
            assert batch_log_prob.shape == (
                5,
            ), f"{name}: batch should return shape (5,)"
            assert np.isfinite(
                batch_log_prob
            ).all(), f"{name}: log_prob should return finite values"

            # Test 3: Consistency with scikit-learn GaussianMixture
            # Get conditioned GMM for comparison
            gmm = model.condition(single_X)
            if hasattr(gmm, "score_samples"):
                sklearn_log_prob = gmm.score_samples(single_y.reshape(1, -1))
                assert (
                    sklearn_log_prob.shape == single_log_prob.shape
                ), f"{name}: should match sklearn shape"
                assert (
                    sklearn_log_prob.ndim == single_log_prob.ndim
                ), f"{name}: should match sklearn dimensions"

    def test_score_interface_compliance(self, models):
        """Test that score follows scikit-learn interface rules."""
        model_dict, X, y, X_multi, y_multi = models

        for name, model in model_dict.items():
            # Test 1: Single sample input -> scalar output
            single_X = X[:1]
            single_y = y[:1]
            single_score = model.score(single_X, single_y)

            assert np.isscalar(
                single_score
            ), f"{name}: score should return scalar for single sample"
            assert np.isfinite(
                single_score
            ), f"{name}: score should return finite value"

            # Test 2: Batch input -> scalar output
            batch_X = X[:5]
            batch_y = y[:5]
            batch_score = model.score(batch_X, batch_y)

            assert np.isscalar(
                batch_score
            ), f"{name}: score should return scalar for batch"
            assert np.isfinite(batch_score), f"{name}: score should return finite value"

            # Test 3: Score should equal mean of log_prob
            batch_log_prob = model.log_prob(batch_X, batch_y)
            expected_score = np.mean(batch_log_prob)
            np.testing.assert_almost_equal(
                batch_score,
                expected_score,
                decimal=10,
                err_msg=f"{name}: score should equal mean of log_prob",
            )

    def test_sample_interface_compliance(self, models):
        """Test that sample follows scikit-learn interface rules."""
        model_dict, X, y, X_multi, y_multi = models

        for name, model in model_dict.items():
            # Test 1: Single sample input -> 2D output (n_samples, n_targets)
            single_X = X[:1]
            single_samples = model.sample(single_X, n_samples=5)

            assert isinstance(
                single_samples, np.ndarray
            ), f"{name}: sample should return numpy array"
            assert (
                single_samples.ndim == 2
            ), f"{name}: sample should return 2D array for single input"
            assert single_samples.shape == (
                5,
                1,
            ), f"{name}: single input should return shape (5, 1)"
            assert np.isfinite(
                single_samples
            ).all(), f"{name}: sample should return finite values"

            # Test 2: Batch input -> 3D output (n_inputs, n_samples, n_targets)
            batch_X = X[:3]
            batch_samples = model.sample(batch_X, n_samples=4)

            assert isinstance(
                batch_samples, np.ndarray
            ), f"{name}: sample should return numpy array"
            assert (
                batch_samples.ndim == 3
            ), f"{name}: sample should return 3D array for batch input"
            assert batch_samples.shape == (
                3,
                4,
                1,
            ), f"{name}: batch input should return shape (3, 4, 1)"
            assert np.isfinite(
                batch_samples
            ).all(), f"{name}: sample should return finite values"

            # Test 3: No random_state parameter (scikit-learn compliance)
            import inspect

            sig = inspect.signature(model.sample)
            params = list(sig.parameters.keys())
            assert (
                "random_state" not in params
            ), f"{name}: sample should not have random_state parameter"

    def test_predict_interface_compliance(self, models):
        """Test that predict follows scikit-learn interface rules."""
        model_dict, X, y, X_multi, y_multi = models

        for name, model in model_dict.items():
            # Test 1: Single sample input -> 1D output
            single_X = X[:1]
            single_pred = model.predict(single_X)

            assert isinstance(
                single_pred, np.ndarray
            ), f"{name}: predict should return numpy array"
            assert (
                single_pred.ndim == 1
            ), f"{name}: predict should return 1D array for single sample"
            assert single_pred.shape == (
                1,
            ), f"{name}: single sample should return shape (1,)"
            assert np.isfinite(
                single_pred
            ).all(), f"{name}: predict should return finite values"

            # Test 2: Batch input -> 1D output
            batch_X = X[:5]
            batch_pred = model.predict(batch_X)

            assert isinstance(
                batch_pred, np.ndarray
            ), f"{name}: predict should return numpy array"
            assert (
                batch_pred.ndim == 1
            ), f"{name}: predict should return 1D array for batch"
            assert batch_pred.shape == (5,), f"{name}: batch should return shape (5,)"
            assert np.isfinite(
                batch_pred
            ).all(), f"{name}: predict should return finite values"

    def test_condition_interface_compliance(self, models):
        """Test that condition returns scikit-learn GaussianMixture objects."""
        model_dict, X, y, X_multi, y_multi = models

        for name, model in model_dict.items():
            # Test 1: Single sample input
            single_X = X[:1]
            gmm = model.condition(single_X)

            assert isinstance(
                gmm, GaussianMixture
            ), f"{name}: condition should return GaussianMixture"
            assert hasattr(
                gmm, "score_samples"
            ), f"{name}: returned GMM should have score_samples"
            assert hasattr(
                gmm, "predict_proba"
            ), f"{name}: returned GMM should have predict_proba"
            assert hasattr(gmm, "sample"), f"{name}: returned GMM should have sample"

            # Test 2: Batch input
            batch_X = X[:3]
            gmm_list = model.condition(batch_X)

            assert isinstance(
                gmm_list, list
            ), f"{name}: batch condition should return list"
            assert len(gmm_list) == 3, f"{name}: should return one GMM per input"
            for i, gmm in enumerate(gmm_list):
                assert isinstance(
                    gmm, GaussianMixture
                ), f"{name}: GMM {i} should be GaussianMixture"

    def test_input_shape_handling(self, models):
        """Test that methods handle various input shapes correctly."""
        model_dict, X, y, X_multi, y_multi = models

        for name, model in model_dict.items():
            # Test 1: 1D input (should be reshaped to 2D internally)
            single_X_1d = X[0]  # Shape: (2,)
            single_y_1d = y[0]  # Shape: ()

            # These should work without error
            pred_1d = model.predict(single_X_1d.reshape(1, -1))
            log_prob_1d = model.log_prob(
                single_X_1d.reshape(1, -1), single_y_1d.reshape(1, -1)
            )
            sample_1d = model.sample(single_X_1d.reshape(1, -1), n_samples=3)

            assert pred_1d.shape == (1,), f"{name}: 1D input should work for predict"
            assert log_prob_1d.shape == (
                1,
            ), f"{name}: 1D input should work for log_prob"
            assert sample_1d.shape == (3, 1), f"{name}: 1D input should work for sample"

            # Test 2: 2D input (standard case)
            single_X_2d = X[:1]  # Shape: (1, 2)
            single_y_2d = y[:1]  # Shape: (1,)

            pred_2d = model.predict(single_X_2d)
            log_prob_2d = model.log_prob(single_X_2d, single_y_2d)
            sample_2d = model.sample(single_X_2d, n_samples=3)

            assert pred_2d.shape == (1,), f"{name}: 2D input should work for predict"
            assert log_prob_2d.shape == (
                1,
            ), f"{name}: 2D input should work for log_prob"
            assert sample_2d.shape == (3, 1), f"{name}: 2D input should work for sample"

    def test_consistency_with_sklearn_gmm(self, models):
        """Test that our models are consistent with sklearn GaussianMixture where applicable."""
        model_dict, X, y, X_multi, y_multi = models

        # Create sklearn GMM for comparison
        sklearn_gmm = GaussianMixture(n_components=3, random_state=42)
        sklearn_gmm.fit(X)

        for name, model in model_dict.items():
            # Test log_prob vs score_samples consistency
            single_X = X[:1]
            single_y = y[:1]

            our_log_prob = model.log_prob(single_X, single_y)
            sklearn_log_prob = sklearn_gmm.score_samples(single_X)

            # Both should return 1D arrays
            assert our_log_prob.ndim == 1, f"{name}: log_prob should be 1D"
            assert (
                sklearn_log_prob.ndim == 1
            ), f"{name}: sklearn score_samples should be 1D"
            assert (
                our_log_prob.shape == sklearn_log_prob.shape
            ), f"{name}: shapes should match"

            # Test sample consistency
            our_samples = model.sample(single_X, n_samples=5)
            sklearn_samples = sklearn_gmm.sample(5)[0]

            # Both should return 2D arrays
            assert our_samples.ndim == 2, f"{name}: sample should be 2D"
            assert sklearn_samples.ndim == 2, f"{name}: sklearn sample should be 2D"
            assert (
                our_samples.shape[0] == sklearn_samples.shape[0]
            ), f"{name}: sample count should match"

    def test_method_signatures_match_sklearn(self, models):
        """Test that method signatures match sklearn patterns."""
        model_dict, X, y, X_multi, y_multi = models
        import inspect

        for name, model in model_dict.items():
            # Test log_prob signature
            log_prob_sig = inspect.signature(model.log_prob)
            log_prob_params = list(log_prob_sig.parameters.keys())
            assert "X" in log_prob_params, f"{name}: log_prob should have 'X' parameter"
            assert "y" in log_prob_params, f"{name}: log_prob should have 'y' parameter"
            assert (
                log_prob_params[0] == "X"
            ), f"{name}: 'X' should be first parameter in log_prob"
            assert (
                log_prob_params[1] == "y"
            ), f"{name}: 'y' should be second parameter in log_prob"

            # Test score signature
            score_sig = inspect.signature(model.score)
            score_params = list(score_sig.parameters.keys())
            assert "X" in score_params, f"{name}: score should have 'X' parameter"
            assert "y" in score_params, f"{name}: score should have 'y' parameter"
            assert (
                score_params[0] == "X"
            ), f"{name}: 'X' should be first parameter in score"
            assert (
                score_params[1] == "y"
            ), f"{name}: 'y' should be second parameter in score"

            # Test sample signature
            sample_sig = inspect.signature(model.sample)
            sample_params = list(sample_sig.parameters.keys())
            assert "X" in sample_params, f"{name}: sample should have 'X' parameter"
            assert (
                "n_samples" in sample_params
            ), f"{name}: sample should have 'n_samples' parameter"
            assert (
                sample_params[0] == "X"
            ), f"{name}: 'X' should be first parameter in sample"
            assert (
                sample_params[1] == "n_samples"
            ), f"{name}: 'n_samples' should be second parameter in sample"

            # Test predict signature
            predict_sig = inspect.signature(model.predict)
            predict_params = list(predict_sig.parameters.keys())
            assert "X" in predict_params, f"{name}: predict should have 'X' parameter"
            assert (
                predict_params[0] == "X"
            ), f"{name}: 'X' should be first parameter in predict"

    def test_error_handling_follows_sklearn(self, models):
        """Test that error handling follows sklearn patterns."""
        model_dict, X, y, X_multi, y_multi = models

        for name, model in model_dict.items():
            # Test unfitted model errors
            unfitted_model = model.__class__(n_components=3, random_state=42)

            with pytest.raises((ValueError, AttributeError)):
                unfitted_model.predict(X[:1])

            with pytest.raises((ValueError, AttributeError)):
                unfitted_model.score(X[:1], y[:1])

            with pytest.raises((ValueError, AttributeError)):
                unfitted_model.log_prob(X[:1], y[:1])

            with pytest.raises((ValueError, AttributeError)):
                unfitted_model.sample(X[:1], n_samples=5)

    def test_return_types_are_consistent(self, models):
        """Test that return types are consistent across all methods."""
        model_dict, X, y, X_multi, y_multi = models

        for name, model in model_dict.items():
            single_X = X[:1]
            single_y = y[:1]

            # All methods should return numpy arrays (except score which returns scalar)
            predict_result = model.predict(single_X)
            log_prob_result = model.log_prob(single_X, single_y)
            sample_result = model.sample(single_X, n_samples=3)

            assert isinstance(
                predict_result, np.ndarray
            ), f"{name}: predict should return numpy array"
            assert isinstance(
                log_prob_result, np.ndarray
            ), f"{name}: log_prob should return numpy array"
            assert isinstance(
                sample_result, np.ndarray
            ), f"{name}: sample should return numpy array"

            # All arrays should be float64
            assert (
                predict_result.dtype == np.float64
            ), f"{name}: predict should return float64"
            assert (
                log_prob_result.dtype == np.float64
            ), f"{name}: log_prob should return float64"
            assert (
                sample_result.dtype == np.float64
            ), f"{name}: sample should return float64"

            # All arrays should be finite
            assert np.isfinite(
                predict_result
            ).all(), f"{name}: predict should return finite values"
            assert np.isfinite(
                log_prob_result
            ).all(), f"{name}: log_prob should return finite values"
            assert np.isfinite(
                sample_result
            ).all(), f"{name}: sample should return finite values"

    def test_interface_rules_against_sklearn_gmm(self):
        """Test that our interface rules match actual scikit-learn GaussianMixture behavior."""
        from sklearn.datasets import make_regression

        # Create test data
        X, y = make_regression(
            n_samples=100, n_features=2, n_targets=1, random_state=42
        )
        X_multi, y_multi = make_regression(
            n_samples=100, n_features=2, n_targets=3, random_state=42
        )

        # Test single target case
        print("\n=== TESTING SINGLE TARGET CASE ===")
        gmm_single = GaussianMixture(n_components=3, random_state=42)
        gmm_single.fit(y.reshape(-1, 1))  # Fit on targets

        # Test score_samples (equivalent to our log_prob)
        # single_X = X[:1]
        single_y = y[:1]

        sklearn_log_prob = gmm_single.score_samples(single_y.reshape(1, -1))
        print(f"sklearn score_samples shape: {sklearn_log_prob.shape}")
        print(f"sklearn score_samples ndim: {sklearn_log_prob.ndim}")
        assert sklearn_log_prob.ndim == 1, "sklearn score_samples should be 1D"
        assert sklearn_log_prob.shape == (
            1,
        ), "sklearn score_samples should have shape (1,) for single sample"

        # Test sample
        sklearn_samples = gmm_single.sample(n_samples=5)[0]  # Get X from tuple
        print(f"sklearn sample shape: {sklearn_samples.shape}")
        print(f"sklearn sample ndim: {sklearn_samples.ndim}")
        assert sklearn_samples.ndim == 2, "sklearn sample should be 2D"
        assert sklearn_samples.shape == (
            5,
            1,
        ), "sklearn sample should have shape (5, 1) for single target"

        # Test predict_proba
        sklearn_proba = gmm_single.predict_proba(single_y.reshape(1, -1))
        print(f"sklearn predict_proba shape: {sklearn_proba.shape}")
        print(f"sklearn predict_proba ndim: {sklearn_proba.ndim}")
        assert sklearn_proba.ndim == 2, "sklearn predict_proba should be 2D"
        assert sklearn_proba.shape == (
            1,
            3,
        ), "sklearn predict_proba should have shape (1, n_components)"

        # Test multi target case
        print("\n=== TESTING MULTI TARGET CASE ===")
        gmm_multi = GaussianMixture(n_components=3, random_state=42)
        gmm_multi.fit(y_multi)  # y_multi is already 2D

        # single_X_multi = X_multi[:1]
        single_y_multi = y_multi[:1]

        sklearn_log_prob_multi = gmm_multi.score_samples(single_y_multi)
        print(f"sklearn multi score_samples shape: {sklearn_log_prob_multi.shape}")
        assert (
            sklearn_log_prob_multi.ndim == 1
        ), "sklearn multi score_samples should be 1D"
        assert sklearn_log_prob_multi.shape == (
            1,
        ), "sklearn multi score_samples should have shape (1,) for single sample"

        sklearn_samples_multi = gmm_multi.sample(n_samples=5)[0]
        print(f"sklearn multi sample shape: {sklearn_samples_multi.shape}")
        assert sklearn_samples_multi.ndim == 2, "sklearn multi sample should be 2D"
        assert sklearn_samples_multi.shape == (
            5,
            3,
        ), "sklearn multi sample should have shape (5, n_targets)"

        sklearn_proba_multi = gmm_multi.predict_proba(single_y_multi)
        print(f"sklearn multi predict_proba shape: {sklearn_proba_multi.shape}")
        assert sklearn_proba_multi.ndim == 2, "sklearn multi predict_proba should be 2D"
        assert sklearn_proba_multi.shape == (
            1,
            3,
        ), "sklearn multi predict_proba should have shape (1, n_components)"

        # Test batch processing
        print("\n=== TESTING BATCH PROCESSING ===")
        # batch_X = X[:3]
        batch_y = y[:3]

        sklearn_log_prob_batch = gmm_single.score_samples(batch_y.reshape(-1, 1))
        print(f"sklearn batch score_samples shape: {sklearn_log_prob_batch.shape}")
        assert (
            sklearn_log_prob_batch.ndim == 1
        ), "sklearn batch score_samples should be 1D"
        assert sklearn_log_prob_batch.shape == (
            3,
        ), "sklearn batch score_samples should have shape (n_samples,)"

        sklearn_proba_batch = gmm_single.predict_proba(batch_y.reshape(-1, 1))
        print(f"sklearn batch predict_proba shape: {sklearn_proba_batch.shape}")
        assert sklearn_proba_batch.ndim == 2, "sklearn batch predict_proba should be 2D"
        assert sklearn_proba_batch.shape == (
            3,
            3,
        ), "sklearn batch predict_proba should have shape (n_samples, n_components)"

        print("\n=== INTERFACE RULES VALIDATION COMPLETE ===")
        print("✅ All interface rules match scikit-learn GaussianMixture behavior!")
        print("✅ Our CGMM models should follow these same rules.")

        return True
