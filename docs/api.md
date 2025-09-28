# API Reference

The `cgmm` library provides conditional Gaussian mixture models compatible with scikit-learn. These models learn joint distributions over input variables X and target variables y, then condition on X to predict y or generate samples from p(y|X).

## Quick Start

```python
from cgmm import ConditionalGMMRegressor, MixtureOfExpertsRegressor, DiscriminativeConditionalGMMRegressor
from sklearn.datasets import make_regression

# Generate sample data
X, y = make_regression(n_samples=100, n_features=2, n_targets=1, random_state=42)

# Fit a conditional GMM
model = ConditionalGMMRegressor(n_components=3, random_state=42)
model.fit(X, y)

# Predict mean, generate samples, and calculate PDF
y_pred = model.predict(X[:5])           # E[y|X] for each input
y_samples = model.sample(X[:5], n_samples=3)  # Sample from p(y|X)
y_pdf = np.exp(model.log_prob(X[:5], y_pred))  # p(y|X) density values
```

## Core Classes

### ConditionalGMMRegressor

A regressor that learns a joint Gaussian mixture model over [X, y] and analytically conditions to produce p(y|X). This approach is computationally efficient and provides exact conditional distributions.

**Key Features:**
- Analytical conditioning using matrix operations
- Supports both single and multi-output regression
- Compatible with scikit-learn pipelines and cross-validation
- Provides exact conditional means, covariances, and samples

```python
# Basic usage
model = ConditionalGMMRegressor(n_components=5, random_state=42)
model.fit(X, y)
y_pred = model.predict(X_test)  # Mean predictions

# Get conditional mixture and generate samples
gmm = model.condition(X_test)  # Returns sklearn GaussianMixture
y_samples = model.sample(X_test, n_samples=100)  # Generate samples
```

**Constructor Parameters:**
- `n_components` (int): Number of mixture components
- `covariance_type` (str): Type of covariance matrix ("full", "diag", "spherical")
- `reg_covar` (float): Regularization for numerical stability
- `random_state` (int): Random seed for reproducibility

**Main Methods:**

- **`fit(X, y)`** → `self`: Learn the joint GMM from training data
  - `X`: Input features of shape (n_samples, n_features)
  - `y`: Target values of shape (n_samples, n_targets)
  - Returns fitted model for method chaining

- **`predict(X)`** → `np.ndarray`: Predict conditional mean E[y|X]
  - `X`: Input features of shape (n_samples, n_features) or (n_features,)
  - Returns: Predicted means of shape (n_samples, n_targets) or (n_targets,)

- **`sample(X, n_samples=1, random_state=None)`** → `np.ndarray`: Generate samples from p(y|X)
  - `X`: Input features of shape (n_samples, n_features) or (n_features,)
  - `n_samples`: Number of samples to generate per input
  - Returns: Samples of shape (n_samples, n_targets) or (n_inputs, n_samples, n_targets)

- **`condition(X)`** → `GaussianMixture` or `List[GaussianMixture]`: Get conditional mixture as sklearn object
  - Returns sklearn GaussianMixture with standard attributes: `weights_`, `means_`, `covariances_`
  - Enables use of sklearn methods: `sample()`, `score_samples()`, `predict_proba()`

- **`score(X, y)`** → `float`: Compute mean conditional log-likelihood
  - `X`: Input features
  - `y`: Target values
  - Returns: Mean log p(y|X) across all samples

- **`condition(X)`** → `GaussianMixture` or `List[GaussianMixture]`: Return scikit-learn GMM objects
  - Enables use of standard sklearn methods like `score_samples()`, `sample()`, `predict_proba()`

### MixtureOfExpertsRegressor

A regressor that uses softmax gating to weight Gaussian experts with affine mean functions. Each expert learns a linear relationship between inputs and targets, with the gating network determining expert weights.

**Key Features:**
- Softmax gating network for expert selection
- Linear mean functions for each expert
- Supports different covariance types
- Good for modeling complex, multi-modal relationships

```python
# Mixture of experts with linear mean functions
model = MixtureOfExpertsRegressor(
    n_components=4, 
    covariance_type='full',
    random_state=42
)
model.fit(X, y)
y_pred = model.predict(X_test)  # Weighted combination of expert predictions
```

**Constructor Parameters:**
- `n_components` (int): Number of expert components
- `covariance_type` (str): Covariance structure ("full", "diag", "spherical")
- `mean_function` (str): Type of mean function ("linear", "constant")
- `reg_covar` (float): Regularization for numerical stability
- `gating_penalty` (float): L2 penalty on gating network weights
- `max_iter` (int): Maximum EM iterations
- `random_state` (int): Random seed

**Main Methods:**
- Same interface as `ConditionalGMMRegressor`: `fit()`, `predict()`, `sample()`, `score()`, `condition()`

### DiscriminativeConditionalGMMRegressor

A regressor that directly optimizes conditional likelihood using a discriminative EM algorithm. This approach focuses on learning p(y|X) without modeling the full joint distribution.

**Key Features:**
- Discriminative training focused on conditional likelihood
- Direct optimization of p(y|X) without joint modeling
- Supports different covariance types
- Good for cases where joint modeling is difficult

```python
# Discriminative conditional GMM
model = DiscriminativeConditionalGMMRegressor(
    n_components=3,
    covariance_type='full',
    reg_covar=1e-2,
    random_state=42
)
model.fit(X, y)
y_pred = model.predict(X_test)  # Discriminatively learned predictions
```

**Constructor Parameters:**
- `n_components` (int): Number of mixture components
- `covariance_type` (str): Covariance structure ("full", "diag", "spherical")
- `reg_covar` (float): Regularization for numerical stability
- `max_iter` (int): Maximum EM iterations
- `weight_step` (float): Step size for weight updates
- `random_state` (int): Random seed

**Main Methods:**
- Same interface as other regressors: `fit()`, `predict()`, `sample()`, `score()`, `condition()`

### GMMConditioner

A utility class for conditioning pre-fitted Gaussian mixture models. This is the core implementation used by `ConditionalGMMRegressor` for analytical conditioning.

**Key Features:**
- Precomputes conditioning matrices for efficiency
- Supports batch conditioning
- Returns scikit-learn `GaussianMixture` objects
- Used internally by `ConditionalGMMRegressor`

```python
from sklearn.mixture import GaussianMixture
from cgmm import GMMConditioner

# Fit a joint GMM
gmm = GaussianMixture(n_components=3, covariance_type='full')
gmm.fit(np.column_stack([X, y]))

# Create conditioner for analytical conditioning
conditioner = GMMConditioner(gmm, cond_idx=[0, 1])  # First 2 dims are X
conditioner.precompute()

# Condition on specific X values
conditioned_gmm = conditioner.condition(X_test)  # Returns GaussianMixture
```

**Constructor Parameters:**
- `mixture_estimator` (GaussianMixture): Pre-fitted GMM
- `cond_idx` (Sequence[int]): Indices of conditioning variables X
- `reg_covar` (float): Regularization for numerical stability

**Main Methods:**
- **`precompute()`** → `self`: Precompute conditioning matrices
- **`condition(x)`** → `GaussianMixture` or `List[GaussianMixture]`: Condition on X=x

## Common Usage Patterns

### Basic Regression

```python
from cgmm import ConditionalGMMRegressor
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit model
model = ConditionalGMMRegressor(n_components=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
r2_score = model.score(X_test, y_test)
```

### Uncertainty Quantification

```python
# Get conditional mixture parameters using sklearn interface
gmm = model.condition(X_test)  # Returns sklearn GaussianMixture
weights = gmm.weights_         # (n_components,)
means = gmm.means_             # (n_components, n_targets)
covariances = gmm.covariances_ # (n_components, n_targets, n_targets)

# Generate samples for uncertainty analysis
samples = model.sample(X_test, n_samples=1000)
confidence_interval = np.percentile(samples, [2.5, 97.5], axis=1)
```

### Model Comparison

```python
from cgmm import ConditionalGMMRegressor, MixtureOfExpertsRegressor, DiscriminativeConditionalGMMRegressor

models = {
    'ConditionalGMM': ConditionalGMMRegressor(n_components=3, random_state=42),
    'MixtureOfExperts': MixtureOfExpertsRegressor(n_components=3, random_state=42),
    'Discriminative': DiscriminativeConditionalGMMRegressor(n_components=3, random_state=42)
}

# Compare models
for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"{name}: {score:.3f}")
```

### Integration with Scikit-learn

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', ConditionalGMMRegressor(random_state=42))
])

# Grid search
param_grid = {'model__n_components': [2, 3, 5, 8]}
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X, y)
```

## Performance Considerations

- **ConditionalGMMRegressor**: Fastest for prediction, good for most use cases
- **MixtureOfExpertsRegressor**: Good for complex relationships, moderate speed
- **DiscriminativeConditionalGMMRegressor**: Best when joint modeling is difficult, slower training

## Compatibility

All models are fully compatible with scikit-learn:
- Support `fit()`, `predict()`, `score()` methods
- Work with `Pipeline`, `GridSearchCV`, `cross_val_score`
- Follow scikit-learn conventions for input/output shapes
- Support both single and multi-output regression

## Error Handling

Common errors and solutions:
- `ValueError: n_components must be positive`: Use valid number of components
- `NotFittedError`: Call `fit()` before using other methods
- `ValueError: Input must be 1- or 2-d`: Ensure input arrays have correct dimensions
- `ValueError: cov must be 2 dimensional and square`: Check covariance type compatibility