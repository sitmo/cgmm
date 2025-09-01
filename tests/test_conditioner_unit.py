import numpy as np
import pytest
from sklearn.mixture import GaussianMixture

from cgmm.conditioner import GMMConditioner, _as_1d

RS = 123


def _fit_joint_gmm(dx=2, dy=1, n=200, rs=RS, cov_type="full"):
    rng = np.random.default_rng(rs)
    X = rng.normal(size=(n, dx))
    Y = rng.normal(size=(n, dy))
    Z = np.concatenate([X, Y], axis=1)
    gmm = GaussianMixture(
        n_components=2, covariance_type=cov_type, random_state=rs, max_iter=50
    ).fit(Z)
    return gmm, dx, dy


def test__as_1d_variants():
    assert _as_1d(5.0).shape == (1,)
    assert _as_1d(np.array([1.0, 2.0])).shape == (2,)
    assert _as_1d(np.array([[3.0, 4.0]])).shape == (2,)
    with pytest.raises(ValueError):
        _as_1d(np.ones((2, 2)))  # not a single x


def test_validate_and_prepare_errors_unfitted_and_types():
    # wrong type
    with pytest.raises(TypeError):
        GMMConditioner(mixture_estimator="not a gmm", cond_idx=[0]).precompute()
    # unfitted gmm
    with pytest.raises(ValueError):
        GMMConditioner(
            mixture_estimator=GaussianMixture(n_components=1), cond_idx=[0]
        ).precompute()
    # non-full covariance
    gmm, dx, dy = _fit_joint_gmm(dx=2, dy=1, cov_type="full")
    gmm_diag, *_ = _fit_joint_gmm(
        dx=2, dy=1, cov_type="diag"
    )  # fitted but not supported
    with pytest.raises(NotImplementedError):
        GMMConditioner(mixture_estimator=gmm_diag, cond_idx=[0, 1]).precompute()


def test_validate_and_prepare_errors_indices():
    gmm, dx, dy = _fit_joint_gmm(dx=3, dy=2)
    # duplicates
    with pytest.raises(ValueError):
        GMMConditioner(gmm, cond_idx=[0, 0, 1]).precompute()
    # out of range
    with pytest.raises(ValueError):
        GMMConditioner(gmm, cond_idx=[0, 10]).precompute()


def test_condition_single_vs_batch_and_precisions():
    gmm, dx, dy = _fit_joint_gmm(dx=2, dy=2)
    cond = GMMConditioner(gmm, cond_idx=[0, 1]).precompute()

    # ----- single-sample (1D) path returns a single GaussianMixture -----
    x1 = np.array([0.1, -0.2])  # shape (dx,)
    gm1 = cond.condition(x1)
    from sklearn.mixture import GaussianMixture as GM

    assert isinstance(gm1, GM)
    assert gm1.means_.shape[1] == dy

    # precisions_cholesky_ should correspond to inv(cov)
    for k in range(gm1.n_components):
        L = gm1.precisions_cholesky_[k]
        P = L @ L.T
        C = gm1.covariances_[k]
        Pinv = np.linalg.inv(C)
        assert np.allclose(P, Pinv, atol=1e-6, rtol=1e-6)

    # ----- batch (2D) path returns list[GaussianMixture] -----
    Xb = np.stack([x1, np.array([0.0, 0.0])], axis=0)  # (2, dx)
    gms = cond.condition(Xb)
    assert isinstance(gms, list) and len(gms) == 2


def test_condition_wrong_x_shape_raises():
    gmm, dx, dy = _fit_joint_gmm(dx=2, dy=1)
    cond = GMMConditioner(gmm, cond_idx=[0, 1]).precompute()
    with pytest.raises(ValueError):
        cond.condition(np.array([[1.0, 2.0, 3.0]]))  # wrong feature count
