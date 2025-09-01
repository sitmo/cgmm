import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
from sklearn.base import clone

from cgmm import ConditionalGMMRegressor


def test_pipeline_and_gridsearch(RS):
    rng = np.random.default_rng(RS)
    n = 300
    X = np.c_[rng.normal(size=n), rng.normal(size=n), rng.integers(0, 3, size=n)]
    y = 1.5 * X[:, 0] - 0.5 * X[:, 1] + 0.2 * X[:, 2] + rng.normal(scale=0.2, size=n)

    num_cols = [0, 1]
    cat_cols = [2]

    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("sc", StandardScaler()),
                        ("poly", PolynomialFeatures(2, include_bias=False)),
                    ]
                ),
                num_cols,
            ),
            ("cat", "passthrough", cat_cols),
        ]
    )

    reg = ConditionalGMMRegressor(random_state=RS)

    pipe = Pipeline([("pre", pre), ("model", reg)])

    grid = {"model__n_components": [1, 2, 3]}
    cv = KFold(n_splits=3, shuffle=True, random_state=RS)

    gs = GridSearchCV(pipe, grid, cv=cv, scoring="r2")
    gs.fit(X, y)

    assert gs.best_estimator_ is not None
    assert "model__n_components" in gs.best_params_

    yhat = gs.predict(X)
    assert yhat.shape == y.shape
    assert r2_score(y, yhat) > 0.7


def test_clone_and_params():
    reg = ConditionalGMMRegressor(n_components=2, random_state=123)
    reg2 = clone(reg).set_params(n_components=3)
    assert reg2.get_params()["n_components"] == 3
