
# Getting started

This page shows how to install **cgmm**, fit a Gaussian Mixture with scikit‑learn, and use `GMMConditioner` / `ConditionalGMMRegressor`.

## Install

```bash
pip install cgmm
```

## Requirements

- Python 3.9–3.12
- NumPy, SciPy, scikit‑learn (installed automatically)
- For the example plots: Matplotlib

## Quick start

```python
import numpy as np
from sklearn.mixture import GaussianMixture
from cgmm import GMMConditioner, ConditionalGMMRegressor

# Example data: shape (n_samples, d)
rng = np.random.default_rng(0)
X = rng.normal(size=(1000, 3))

# Fit a scikit‑learn GMM on the full feature space
gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=0).fit(X)

# We will condition on the first coordinate (X block), predict the rest (y block)
cond_idx = [0]

# --- Conditioner API: returns a posterior GMM over y given X=x ---
cond = GMMConditioner(gmm, cond_idx=cond_idx, reg_covar=1e-9).precompute()

x0 = np.array([0.5])                  # value for the conditioning block X
gmmy = cond.condition(x=x0)           # GaussianMixture over y (d_y = 2)
samples_y = gmmy.sample(5)[0]         # sample from p(y | X=0.5)

# --- Regressor API: posterior mean (and optionally covariance) ---
reg = ConditionalGMMRegressor(gmm, cond_idx=cond_idx).fit(X=np.zeros((1, len(cond_idx))))
y_mean = reg.predict(x0)              # shape (d_y,)
```

### What gets returned?

- `GMMConditioner.condition(x)` returns a **new** `sklearn.mixture.GaussianMixture`
  defined over the **target block** (y) only. Component count matches the original GMM.
- `ConditionalGMMRegressor.predict(x)` returns the posterior mean \(\mathbb{E}[y\,|\,X=x]\).
- `ConditionalGMMRegressor.predict_cov(x)` returns the total posterior covariance
  \(\mathrm{Var}[y\,|\,X=x]\) (mixture of covariances plus between‑component term).

## 2D fitting and conditioning example

See: {doc}`examples/conditional_2d_gmm` for a fully executable page that:

- draws samples from a random 3‑component 2D GMM,
- fits a scikit‑learn `GaussianMixture`,
- shows the scatter and fitted iso‑density contours,
- conditions on \(X=1\),
- overlays the conditional density \(p(y\mid X=1)\) as an offset curve along the vertical line \(x=1\).


## VIX example

See: {doc}`examples/vix_predictor` for a fully executable notebook that:

- loads daily VIX data and forms \(x_t=\log(\mathrm{CLOSE}_t)\),
- fits a 2D `GaussianMixture` on \((x_t,\ \Delta_{t+1})\) with \(\Delta_{t+1}=x_{t+1}-x_t\),
- builds the conditional model \(p(\Delta_{t+1}\mid x_t)\),
- plots the posterior mean \(\mathbb{E}[\Delta_{t+1}\mid x_t]\) as a function of \(x_t\),
- generates multi-step scenarios by iteratively sampling \(\Delta_{t+1}\) and updating \(x_{t+1}=x_t+\Delta_{t+1}\), then converts back to VIX levels with \(\exp(x)\) to show median and 10-90% bands.


## Notes

- Currently, `covariance_type="full"` is required for the input GMM.
- The conditioned mixture is returned in the **reduced** space (y only), avoiding singular covariances.
- Numerical stability: a small `reg_covar` is added to the \(\Sigma_{XX}\) block before inversion.

## Troubleshooting

- **ImportError: cannot import name 'GMMConditioner'** — ensure `cgmm >= 0.1.0` is installed and your environment is clean (`pip show cgmm`).
- **ValueError: cond_idx out of range** — `cond_idx` must reference existing columns of the original data used to fit the GMM.
- **NotImplementedError: covariance_type != 'full'** — at present only the `full` covariance is supported; fit your GMM with `covariance_type="full"`.

## Changelog and versioning

The project follows semantic versioning. See the repository README for release notes.

## Citation

If this library helps your work, please cite it as:

```
Your Name. cgmm: Conditional Gaussian Mixture Models (scikit‑learn compatible). Zenodo (year). DOI: TBD
```
