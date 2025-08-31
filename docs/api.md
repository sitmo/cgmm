# API Reference

Public API of `cgmm` for conditional Gaussian mixture modeling compatible with scikit-learn.

---

## Modules

- cgmm.conditioner — core implementations.
- Import shortcut: `from cgmm import GMMConditioner, ConditionalGMMRegressor`.

---

## Classes

### GMMConditioner

    class GMMConditioner(BaseEstimator):
        """
        Precompute reusable terms to condition a fitted GaussianMixture on a fixed
        set of variables X (conditioning block). Returns a new GaussianMixture over
        the y-block (targets).

        Parameters
        ----------
        mixture_estimator : sklearn.mixture.GaussianMixture
            Fitted GMM with covariance_type="full".
        cond_idx : Sequence[int]
            Indices of conditioning variables X in the original feature order.
            The complement indices constitute y (targets).
        reg_covar : float, default=1e-9
            Diagonal regularization added to the Sigma_XX block per component.

        Notes
        -----
        - Only covariance_type="full" is currently supported.
        - The returned conditioned mixture is defined in the reduced y-space
          (no singular covariances).
        """

        def __init__(
            self,
            mixture_estimator: GaussianMixture,
            cond_idx: Sequence[int],
            reg_covar: float = 1e-9
        ): ...

        def precompute(self) -> "GMMConditioner":
            """
            Validate inputs and compute per-component reusable matrices:
              A_k = Sigma_yX,k * inv(Sigma_XX,k)
              C_k = Sigma_yy,k - A_k * Sigma_Xy,k
            Also precomputes Cholesky(Sigma_XX,k) and log-normalizers for weight updates.
            Returns self.
            """

        def condition(self, x: ArrayLike) -> Union[GaussianMixture, list[GaussianMixture]]:
            """
            Condition on X=x and return a new GaussianMixture over y.

            Parameters
            ----------
            x : array-like
                Shape (d_X,) for a single point or (n_samples, d_X) for a batch.

            Returns
            -------
            gmmy : GaussianMixture or list[GaussianMixture]
                Posterior mixture(s) p(y | X=x). For a batch input, a list is returned.

            Details
            -------
            For each component k:
              mu_{y|x,k} = mu_y,k + A_k * (x - mu_X,k)
              Sigma_{y|x,k} = C_k
              w'_k(x) proportional to w_k * N(x | mu_X,k, Sigma_XX,k), normalized across k.

            The returned object(s) expose:
              - weights_ : (K,)
              - means_ : (K, d_y)
              - covariances_ : (K, d_y, d_y)
              - precisions_cholesky_ : (K, d_y, d_y)
            """

Attributes (after `precompute`)
- cond_idx: tuple[int, ...]  (conditioning indices X)
- _target_idx: tuple[int, ...]  (target indices y)
- _d_X: int, _d_y: int  (block sizes)

Errors
- TypeError if mixture_estimator is not a fitted GaussianMixture
- NotImplementedError if covariance_type != "full"
- ValueError for invalid/duplicate/out-of-range cond_idx or shape mismatch in x

Complexity
- Precompute: O(K * d_X^3) due to Cholesky per component
- Per query: O(K * d_X^2 + K) for means/weights; covariance is reused

Minimal usage

    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=5, covariance_type="full").fit(X)
    cond = GMMConditioner(gmm, cond_idx=[0]).precompute()
    gmmy = cond.condition([1.0])  # mixture over remaining dimensions (y)

---

### ConditionalGMMRegressor

    class ConditionalGMMRegressor(BaseEstimator, RegressorMixin):
        """
        Regressor-style interface for posterior mean and covariance under a fitted GMM.

        Computes E[y | X=x] and (optionally) Var[y | X=x] with the same conditioning
        semantics as GMMConditioner.

        Parameters
        ----------
        mixture_estimator : sklearn.mixture.GaussianMixture
            Fitted GMM with covariance_type="full".
        cond_idx : Sequence[int]
            Conditioning indices (X).
        return_cov : bool, default=False
            If True, enables predict_cov.
        reg_covar : float, default=1e-9
            Diagonal regularization added to Sigma_XX.
        """

        def __init__(
            self,
            mixture_estimator: GaussianMixture,
            cond_idx: Sequence[int],
            return_cov: bool = False,
            reg_covar: float = 1e-9
        ): ...

        def fit(self, X: ArrayLike, y=None) -> "ConditionalGMMRegressor":
            """
            No training of the mixture occurs; this validates and prepares
            internal caches. X is accepted for sklearn API compatibility.
            """

        def predict(self, X: ArrayLike) -> np.ndarray:
            """
            Posterior mean E[y | X=x] for each row in X.

            Parameters
            ----------
            X : array-like, shape (n_samples, d_X) or (d_X,)

            Returns
            -------
            y_mean : ndarray, shape (n_samples, d_y) or (d_y,)
            """

        def predict_cov(self, X: ArrayLike) -> Union[np.ndarray, list[np.ndarray]]:
            """
            Total posterior covariance Var[y | X=x] (mixture-of-Gaussians formula):
              Var[y|x] = sum_k w_k * (Sigma_k + mu_k mu_k^T) - m m^T,
            where m = sum_k w_k * mu_k.

            Returns
            -------
            cov : (d_y, d_y) for a single x, or list of (d_y, d_y) per sample.
            """

Notes
- Works in Pipeline, GridSearchCV, etc. (array-in -> array-out)
- Uses GMMConditioner internally; same assumptions and safeguards

Example

    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(n_components=4, covariance_type="full").fit(X)
    reg = ConditionalGMMRegressor(gmm, cond_idx=[0]).fit(X=np.zeros((1,1)))
    y_mean = reg.predict([[1.0]])      # E[y | X=1]
    y_cov  = reg.predict_cov([[1.0]])  # Var[y | X=1]

---

## Behaviors and conventions

- Space of result: conditioned mixtures live in y-space (targets only).
- Indexing: cond_idx references columns of the data used to fit the GMM.
- Covariance type: only "full" supported at present.
- Numerical stability: reg_covar adds eps * I to Sigma_XX before solves.
- Randomness: conditioning is deterministic; sampling uses GaussianMixture.sample.

---

## Error messages (selected)

- ValueError: Expected x with shape (*, d_X)  (condition(x) shape mismatch)
- NotImplementedError: Only covariance_type='full' is supported currently.
- ValueError: cond_idx contains duplicates / out-of-range

---

## Typical patterns

Posterior mean curve (1D X, 1D y)

    x_grid = np.linspace(xmin, xmax, 200)[:, None]
    y_mean = reg.predict(x_grid).ravel()

Posterior mixture sampling (1D y)

    gmms = cond.condition(x_grid)  # list of mixtures
    samples = [g.sample(1000)[0].ravel() for g in gmms]

Scenario simulation with additive target (Delta)

    def step_mixture(cond, xt, rng):
        gmmd = cond.condition([xt])
        # manual 1D draw:
        k = rng.choice(gmmd.weights_.size, p=gmmd.weights_)
        mu = gmmd.means_[k, 0]
        sd = np.sqrt(gmmd.covariances_[k, 0, 0])
        return float(rng.normal(mu, sd))

    rng = np.random.default_rng(0)
    xt = x0
    for t in range(H):
        d = step_mixture(cond, xt, rng)  # draw Delta
        xt = xt + d

---

## Versioning

- API stability starts at >= 0.1.0. Backwards-compatible changes bump the minor version; breaking changes bump the major version.
