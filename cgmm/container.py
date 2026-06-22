# cgmm/container.py
# Layer 2 — family-agnostic mixture value object + the back-compatible
# GaussianMixtureDistribution.
#
# The only public surface that needs care is the *return type* of conditioning.
# Historically the conditioner returned a plain sklearn GaussianMixture. We keep
# that contract by returning `GaussianMixtureDistribution`, a thin subclass of
# sklearn `GaussianMixture` that ALSO satisfies the structural `Mixture` protocol
# (`isinstance(result, GaussianMixture)` stays True). Heavy-tailed families return
# `MixtureDistribution`, which implements the same protocol with no prior contract.
from __future__ import annotations

from typing import List, Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture

from .distributions import Distribution, MultivariateNormal, _as_generator

Array = np.ndarray


@runtime_checkable
class Mixture(Protocol):
    """Structural protocol for a (frozen) mixture distribution."""

    weights_: Array
    components_: List[Distribution]

    def logpdf(self, x: ArrayLike) -> Array: ...
    def rvs(self, size: int = 1, random_state=None) -> Array: ...
    def mean(self) -> Array: ...
    def cov(self) -> Array: ...
    def condition(self, cond_idx: Sequence[int], x_cond: ArrayLike) -> "Mixture": ...


def _mixture_mean(weights: Array, components: List[Distribution]) -> Array:
    means = np.array([c.mean() for c in components])  # (K, d)
    return weights @ means


def _mixture_cov(weights: Array, components: List[Distribution]) -> Array:
    """Var[X] = sum_k w_k (cov_k + mu_k mu_k^T) - mu mu^T (law of total variance)."""
    mu = _mixture_mean(weights, components)
    d = mu.size
    C = np.zeros((d, d))
    for w, c in zip(weights, components):
        m = c.mean()
        C += w * (c.cov() + np.outer(m, m))
    C -= np.outer(mu, mu)
    return 0.5 * (C + C.T)


def _mixture_logpdf(
    weights: Array, components: List[Distribution], x: ArrayLike
) -> Array:
    X = np.atleast_2d(np.asarray(x, dtype=float))
    log_w = np.log(weights)[:, None]  # (K, 1)
    comp = np.array([c.logpdf(X) for c in components])  # (K, n)
    return logsumexp(log_w + comp, axis=0)  # (n,)


def _condition_weights(
    weights: Array,
    components: List[Distribution],
    cond_idx: Sequence[int],
    x_cond: Array,
) -> Array:
    """Posterior mixing weights after conditioning: w_k' ∝ w_k f_k^cond(x_cond)."""
    log_w = np.log(weights) + np.array(
        [float(c.marginal(cond_idx).logpdf(x_cond)[0]) for c in components]
    )
    w = np.exp(log_w - logsumexp(log_w))
    return w / w.sum()


class MixtureDistribution:
    """A frozen mixture over components of a single distribution family.

    Parameters
    ----------
    weights : array-like of shape (K,)
        Mixing weights (normalized internally).
    components : sequence of Distribution
        Length-K list of components, all the same family and dimension.
    """

    def __init__(self, weights: ArrayLike, components: Sequence[Distribution]):
        self.components_ = list(components)
        w = np.asarray(weights, dtype=float).ravel()
        if w.size != len(self.components_):
            raise ValueError(
                f"weights has length {w.size} but there are {len(self.components_)} components."
            )
        self.weights_ = w / w.sum()

    @property
    def dim(self) -> int:
        return self.components_[0].dim

    def logpdf(self, x: ArrayLike) -> Array:
        return _mixture_logpdf(self.weights_, self.components_, x)

    def mean(self) -> Array:
        return _mixture_mean(self.weights_, self.components_)

    def cov(self) -> Array:
        return _mixture_cov(self.weights_, self.components_)

    def rvs(self, size: int = 1, random_state=None) -> Array:
        rng = _as_generator(random_state)
        ks = rng.choice(len(self.components_), size=int(size), p=self.weights_)
        out = np.empty((int(size), self.dim))
        for i, k in enumerate(ks):
            out[i] = self.components_[k].rvs(size=1, random_state=rng)[0]
        return out

    def condition(
        self, cond_idx: Sequence[int], x_cond: ArrayLike
    ) -> "MixtureDistribution":
        x_cond = np.asarray(x_cond, dtype=float).ravel()
        w = _condition_weights(self.weights_, self.components_, cond_idx, x_cond)
        comps = [c.condition(cond_idx, x_cond) for c in self.components_]
        return MixtureDistribution(w, comps)


class GaussianMixtureDistribution(GaussianMixture):
    """A fitted sklearn ``GaussianMixture`` that also satisfies the ``Mixture``
    protocol.

    ``isinstance(x, GaussianMixture)`` remains True and every sklearn method
    (``sample``, ``score_samples``, ``predict_proba``, ``bic``, ``aic`` …) keeps
    working, because this is a real GaussianMixture with the usual fitted
    attributes (``weights_``, ``means_``, ``covariances_``, ``precisions_cholesky_``)
    hand-set by the conditioner. On top of that it exposes the protocol methods
    (``logpdf``/``mean``/``cov``/``condition``) and a ``components_`` view.
    """

    def _full_covariances(self) -> Array:
        # Reuse the conditioner's covariance-type expansion (handles full/tied/
        # diag/spherical). Lazy import avoids an import cycle at module load.
        from .conditioner import _full_covariances

        return _full_covariances(self)

    def logpdf(self, x: ArrayLike) -> Array:
        X = np.atleast_2d(np.asarray(x, dtype=float))
        return self.score_samples(X)

    def rvs(self, size: int = 1, random_state=None) -> Array:
        """Draw ``size`` samples; returns shape (size, d).

        Protocol-conformant counterpart to sklearn's ``sample`` (which returns a
        ``(X, labels)`` tuple and uses the estimator's own ``random_state``).
        """
        rng = _as_generator(random_state)
        covs = self._full_covariances()
        ks = rng.choice(self.weights_.size, size=int(size), p=self.weights_)
        out = np.empty((int(size), self.means_.shape[1]))
        for i, k in enumerate(ks):
            out[i] = MultivariateNormal(self.means_[k], covs[k]).rvs(
                size=1, random_state=rng
            )[0]
        return out

    def mean(self) -> Array:
        return self.weights_ @ self.means_

    def cov(self) -> Array:
        covs = self._full_covariances()
        mu = self.mean()
        d = self.means_.shape[1]
        C = np.zeros((d, d))
        for w, m, S in zip(self.weights_, self.means_, covs):
            C += w * (S + np.outer(m, m))
        C -= np.outer(mu, mu)
        return 0.5 * (C + C.T)

    @property
    def components_(self) -> List[MultivariateNormal]:
        """View of the components as MultivariateNormal distributions."""
        covs = self._full_covariances()
        return [MultivariateNormal(m, S) for m, S in zip(self.means_, covs)]

    def condition(
        self, cond_idx: Sequence[int], x_cond: ArrayLike
    ) -> "GaussianMixtureDistribution":
        """Condition on ``cond_idx = x_cond``; returns a GaussianMixtureDistribution
        over the complementary block (delegates to the shared conditioning math)."""
        from .conditioner import GMMConditioner

        return GMMConditioner(self, cond_idx).precompute().condition(x_cond)
