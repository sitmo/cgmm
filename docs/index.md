# cgmm: Conditional Gaussian Mixture Models

[![CI](https://github.com/sitmo/cgmm/actions/workflows/tests.yml/badge.svg)](https://github.com/sitmo/cgmm/actions/workflows/tests.yml)
[![Coverage](https://codecov.io/gh/sitmo/cgmm/branch/main/graph/badge.svg)](https://codecov.io/gh/sitmo/cgmm)
[![PyPI version](https://img.shields.io/pypi/v/cgmm.svg)](https://pypi.org/project/cgmm/)
[![Python versions](https://img.shields.io/pypi/pyversions/cgmm.svg)](https://pypi.org/project/cgmm/)
[![License](https://img.shields.io/pypi/l/cgmm.svg)](https://github.com/sitmo/cgmm/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://sitmo.github.io/cgmm/)

**cgmm** is a Python library for **Conditional Gaussian Mixture Models** that seamlessly integrates with scikit-learn. It enables you to fit a joint Gaussian mixture on your data and then condition on a subset of variables to obtain the posterior distribution of the remaining ones.


## Install

```bash
pip install cgmm
```

## Requirements

- Python 3.9–3.12
- NumPy, SciPy, scikit‑learn (installed automatically)
- For the example plots: Matplotlib


## Quick Start

```python
import numpy as np
from sklearn.mixture import GaussianMixture
from cgmm import ConditionalGMMRegressor

# Generate sample data
rng = np.random.default_rng(42)
X = rng.normal(size=(200, 2))
y = 2 * X[:, 0] + X[:, 1] + 0.1 * rng.normal(size=200)

# Fit conditional GMM regressor
model = ConditionalGMMRegressor(n_components=3, random_state=42)
model.fit(X, y)

# Make predictions with uncertainty
X_new = np.array([[1.0, 0.5]])
y_pred = model.predict(X_new)  # Mean prediction
y_cov = model.predict_cov(X_new)  # Covariance matrix

print(f"Prediction: {y_pred[0]:.3f}")
print(f"Uncertainty: {np.sqrt(y_cov[0, 0]):.3f}")
```

## Key Features

- **🔗 Scikit-learn Compatible**: Drop-in replacement for regression tasks
- **📊 Multimodal Predictions**: Capture complex, multi-peaked distributions
- **⚡ Multiple Algorithms**: Conditional GMM, Mixture of Experts, and Discriminative approaches
- **🎯 Uncertainty Quantification**: Full covariance matrices, not just point estimates
- **🔧 Production Ready**: Well-tested, documented, and actively maintained

## Installation

```bash
pip install cgmm
```

**Requirements**: Python 3.9+, NumPy, SciPy, scikit-learn

## Use Cases

- **Multimodal Regression**: Predict complex, multi-peaked target distributions
- **Scenario Simulation**: Generate realistic synthetic data for forecasting
- **Missing Data Imputation**: Fill gaps using learned conditional distributions
- **Inverse Problems**: Solve kinematics, finance, and volatility modeling tasks
- **Uncertainty Quantification**: Provide confidence intervals and risk measures

## Models Available

- **`ConditionalGMMRegressor`**: Joint GMM with analytical conditioning
- **`MixtureOfExpertsRegressor`**: Softmax-gated experts with linear mean functions  
- **`DiscriminativeConditionalGMMRegressor`**: Direct conditional likelihood optimization
- **`GMMConditioner`**: Low-level API for custom conditioning workflows

## Why cgmm?

Traditional regression methods assume unimodal, normally distributed residuals. Real-world data often exhibits:

- **Multiple modes** in the target distribution
- **Complex dependencies** between features and targets  
- **Heteroscedastic noise** that varies with input
- **Non-linear relationships** that linear models miss

cgmm addresses these challenges by modeling the **full conditional distribution** rather than just the mean, enabling:

- More accurate predictions in complex scenarios
- Proper uncertainty quantification
- Generation of realistic synthetic data
- Better handling of multimodal target distributions

## Performance

cgmm is optimized for both accuracy and speed:

- **Fast training** with efficient EM algorithms
- **Scalable** to thousands of samples and moderate dimensions
- **Memory efficient** with sparse covariance representations
- **Parallel execution** where possible

## Citation

If you use cgmm in your research, please cite:

```bibtex
@software{cgmm2024,
  title={cgmm: Conditional Gaussian Mixture Models for Python},
  author={van den Berg, Thijs},
  year={2024},
  url={https://github.com/sitmo/cgmm},
  version={0.3.2}
}
```

## Contributing

We welcome contributions!

- 🐛 **Bug reports**: [GitHub Issues](https://github.com/sitmo/cgmm/issues)
- 💡 **Feature requests**: [GitHub Discussions](https://github.com/sitmo/cgmm/discussions)
- 📖 **Documentation**: [GitHub Wiki](https://github.com/sitmo/cgmm/wiki)

## License

BSD 3-Clause License - see [LICENSE](https://github.com/sitmo/cgmm/blob/main/LICENSE) for details.

---

```{toctree}
:maxdepth: 2
:hidden:

api
examples/conditional_2d_gmm
examples/digits_conditional_modeling
examples/iris_conditional_gmm
examples/regression_models_and_datasets
examples/vix_predictor
