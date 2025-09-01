import numpy as np
from cgmm import ConditionalGMMRegressor


def test_predict_one_row_return_cov_true():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    y = X[:, 0] - X[:, 1] + rng.normal(scale=0.1, size=100)
    m = ConditionalGMMRegressor(n_components=2, random_state=0, return_cov=True)
    m.fit(X, y)
    mu, var = m.predict(X[:1])  # still returns (mean, var) with shapes (1,), (1,)
    assert mu.shape == (1,)
    assert var.shape == (1,)
