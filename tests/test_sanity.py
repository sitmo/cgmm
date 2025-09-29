import numpy as np

from cgmm import ConditionalGMMRegressor


def test_fit_predict_basic(RS):
    # Simple conditional relation: y = 2*x + noise
    rng = np.random.default_rng(RS)
    n = 400
    X = rng.normal(size=(n, 1))
    y = 2.0 * X[:, 0] + rng.normal(scale=0.1, size=n)

    model = ConditionalGMMRegressor(n_components=2, random_state=RS)  # adapt params
    model.fit(X, y)

    y_pred = model.predict(X)
    assert y_pred.shape == y.shape

    # sanity: correlation should be positive and reasonably high
    corr = np.corrcoef(y, y_pred)[0, 1]
    assert corr > 0.8


def test_sample_shapes(RS):
    rng = np.random.default_rng(RS)
    X = rng.normal(size=(50, 3))
    y = rng.normal(size=50)

    model = ConditionalGMMRegressor(n_components=3, random_state=RS)
    model.fit(X, y)

    samples = model.sample(X[:10], n_samples=5)  # shape (10, 5, 1)
    assert samples.shape == (10, 5, 1)
