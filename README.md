# cgmm

[![PyPI version](https://img.shields.io/pypi/v/cgmm.svg)](https://pypi.org/project/cgmm/)
[![Documentation Status](https://readthedocs.org/projects/cgmm/badge/?version=latest)](https://cgmm.readthedocs.io/en/latest/?badge=latest)
![Tests](https://github.com/sitmo/cgmm/actions/workflows/tests.yml/badge.svg)
[![codecov](https://codecov.io/gh/sitmo/cgmm/branch/main/graph/badge.svg)](https://codecov.io/gh/sitmo/cgmm)


**cgmm** provides **Conditional Gaussian Mixture Models** that are fully compatible with scikit-learn.
It lets you fit a standard `GaussianMixture` on your data, then **condition** on a subset of variables
to obtain the posterior distribution of the remaining ones.

Typical applications:
- Multimodal regression (`E[y | X=x]` and predictive bands)
- Scenario simulation and stochastic forecasting
- Imputation of missing values
- Inverse problems (e.g. kinematics, finance, volatility)

---

## Features

- `GMMConditioner` — take a fitted `GaussianMixture`, choose conditioning indices, and get a new mixture over the target block.
- `ConditionalGMMRegressor` — sklearn-style regressor wrapper, exposes `.predict` and `.predict_cov`.
- Compatible with scikit-learn pipelines & tools.
- Support for multi-modal posteriors (mixtures, not just means).
- Well-tested, BSD-3 licensed.

---

## Installation

```bash
pip install cgmm
```

---

## Quick Start

```python
import numpy as np
from sklearn.mixture import GaussianMixture
from cgmm import GMMConditioner, ConditionalGMMRegressor

# Example data
rng = np.random.default_rng(0)
X = rng.normal(size=(1000, 3))

# Fit a scikit-learn GMM on all features
gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=0).fit(X)

# Condition on the first coordinate (X), predict the rest (y)
cond = GMMConditioner(gmm, cond_idx=[0]).precompute()
gmmy = cond.condition([0.5])        # returns a GaussianMixture over y

samples = gmmy.sample(5)[0]         # sample from p(y | X=0.5)

# Regressor interface for posterior mean
reg = ConditionalGMMRegressor(gmm, cond_idx=[0]).fit(X=np.zeros((1, 1)))
y_mean = reg.predict([[0.5]])       # E[y | X=0.5]
```

---

## Examples

- 2D conditional demo: scatter + contours + conditional slice.  
  https://cgmm.readthedocs.io/en/latest/examples/conditional_2d_gmm.html
- VIX scenario generation: model daily log-VIX changes and simulate stochastic futures.  
  https://cgmm.readthedocs.io/en/latest/examples/vix_predictor.html

See the documentation for full tutorials: https://cgmm.readthedocs.io/

---

## Documentation

Full documentation is hosted on Read the Docs: https://cgmm.readthedocs.io/

Includes:
- API reference
- Tutorials and examples
- Background on conditional GMMs

---

## Contributing

Contributions are welcome! Typical workflow:

```bash
git clone https://github.com/your-org/cgmm.git
cd cgmm
poetry install -E docs --with dev
pre-commit install
```

- Format & lint: `make precommit`
- Build docs locally: `make docs`
- Bump version: `make bump-patch` (or minor/major)
- Push tags → GitHub Actions → PyPI release

Please open issues or pull requests on GitHub.

---

## License

BSD-3-Clause License © 2025 Thijs van den Berg
