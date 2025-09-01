# Conditional Gaussian Mixture Models

[![CI](https://github.com/sitmo/cgmm/actions/workflows/tests.yml/badge.svg)](https://github.com/sitmo/cgmm/actions/workflows/tests.yml)
[![Coverage](https://codecov.io/gh/sitmo/cgmm/branch/main/graph/badge.svg)](https://codecov.io/gh/sitmo/cgmm)
[![PyPI version](https://img.shields.io/pypi/v/cgmm.svg)](https://pypi.org/project/cgmm/)
[![Python versions](https://img.shields.io/pypi/pyversions/cgmm.svg)](https://pypi.org/project/cgmm/)
[![License](https://img.shields.io/pypi/l/cgmm.svg)](https://github.com/sitmo/cgmm/blob/main/LICENSE)


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


```{toctree}
:maxdepth: 2
:hidden:

getting-started
api
examples/conditional_2d_gmm
examples/vix_predictor
