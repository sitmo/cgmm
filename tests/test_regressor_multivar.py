import numpy as np
from cgmm import ConditionalGMMRegressor

RS = 321


def test_predict_and_sample_multivariate_y_with_cov_and_batch(RS=RS):
    rng = np.random.default_rng(RS)
    n = 220
    X = rng.normal(size=(n, 3))

    # 2-D target
    y1 = 0.7 * X[:, 0] - 0.3 * X[:, 1] + rng.normal(scale=0.2, size=n)
    y2 = -0.5 * X[:, 2] + 0.1 * X[:, 0] + rng.normal(scale=0.3, size=n)
    Y = np.c_[y1, y2]

    model = ConditionalGMMRegressor(n_components=3, random_state=RS, return_cov=True)
    model.fit(X, Y)

    # batch predict with cov
    mean, cov = model.predict(X[:5])
    assert mean.shape == (5, 2)
    assert cov.shape == (5, 2, 2)
    # covariance must be symmetric positive-semi-definite
    for i in range(5):
        assert np.allclose(cov[i], cov[i].T, atol=1e-10)
        eig = np.linalg.eigvalsh(cov[i])
        assert np.all(eig >= -1e-8)

    # sampling shape for multivariate target
    S = model.sample(X[:4], n_samples=7)
    assert S.shape == (4, 7, 2)
