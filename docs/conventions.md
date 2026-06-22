# Distributions & Conditioning: conventions and equations

This page documents the mathematical conventions used throughout `cgmm`: the
parameter names, the probability density functions, and the closed-form
marginalization and conditioning rules for each distribution family. The code
objects ([`MultivariateNormal`](api.md), `MultivariateStudentT`,
`MixtureDistribution`, `GaussianMixtureDistribution`) implement exactly these
formulas, so this is the reference for *why a returned object has the parameters
it has*.

## Notation

We split the variables of a $d$-dimensional distribution into two blocks: a
**conditioning** block $X_2$ (the variables we observe, selected by `cond_idx`)
and the **target** block $X_1$ (the complement). Let $q = \dim(X_2)$.

For a scale/covariance matrix $S$ we write the corresponding block partition

$$
S = \begin{pmatrix} S_{11} & S_{12} \\ S_{21} & S_{22} \end{pmatrix},
\qquad S_{21} = S_{12}^{\mathsf T},
$$

where index `1` is the target block and index `2` is the conditioning block.
Three quantities recur in every family:

$$
\textbf{affine mean:}\quad \mu_{1\mid 2} = \mu_1 + S_{12} S_{22}^{-1} (x_2 - \mu_2)
$$

$$
\textbf{Schur complement:}\quad S_{11\cdot 2} = S_{11} - S_{12} S_{22}^{-1} S_{21}
$$

$$
\textbf{Mahalanobis term:}\quad d^2 = (x_2 - \mu_2)^{\mathsf T} S_{22}^{-1} (x_2 - \mu_2)
$$

```{note}
In the API, `cond_idx` selects the conditioning block $X_2$; the target block
$X_1$ is the complement, returned in increasing index order. `condition(cond_idx,
x_cond)` returns a distribution of the **same family** over $X_1$.
```

## Multivariate Normal

Object: `MultivariateNormal(mean, cov)`. Parameters: `mean` $=\mu$ (shape `(d,)`)
and `cov` $=\Sigma$ (shape `(d, d)`, symmetric positive-definite). For the
Gaussian family the scale matrix *equals* the covariance, so `scale_` is an alias
of `cov()`.

**Density**

$$
f(x) = (2\pi)^{-d/2}\,|\Sigma|^{-1/2}
       \exp\!\Big(-\tfrac12 (x-\mu)^{\mathsf T}\Sigma^{-1}(x-\mu)\Big).
$$

**Marginal** (`marginal(idx)`): a sub-block is Gaussian with the sub-vector of
$\mu$ and the sub-matrix of $\Sigma$.

**Conditional** (`condition(cond_idx, x_cond)`):

$$
X_1 \mid X_2 = x_2 \;\sim\; \mathcal N\big(\mu_{1\mid 2},\; S_{11\cdot 2}\big).
$$

The conditional covariance does **not** depend on $x_2$ (homoskedastic) - this is
the special property the heavy-tailed families relax.

## Multivariate Student-t

Object: `MultivariateStudentT(loc, scale, df)`. Parameters follow
`scipy.stats.multivariate_t`: location `loc` $=\mu$, scale (dispersion) matrix
`scale` $=\Sigma$, and degrees of freedom `df` $=\nu > 0$.

```{important}
`scale` is the **dispersion matrix, not the covariance**. They differ by a
factor depending on $\nu$ (see moments below), which is why `scale_` and `cov()`
are exposed separately.
```

**Density** in dimension $p$:

$$
f(x) = \frac{\Gamma\!\big(\tfrac{\nu+p}{2}\big)}
            {\Gamma\!\big(\tfrac{\nu}{2}\big)\,(\nu\pi)^{p/2}\,|\Sigma|^{1/2}}
       \left[1 + \tfrac{1}{\nu}(x-\mu)^{\mathsf T}\Sigma^{-1}(x-\mu)\right]^{-(\nu+p)/2}.
$$

**Moments**

$$
\mathbb E[X] = \mu \;\;(\nu > 1), \qquad
\operatorname{Cov}[X] = \frac{\nu}{\nu-2}\,\Sigma \;\;(\nu > 2).
$$

As $\nu \to \infty$ the Student-t converges to $\mathcal N(\mu, \Sigma)$ and
$\operatorname{Cov}[X] \to \Sigma$, recovering the Gaussian.

**Marginal** (`marginal(idx)`): same $\nu$, sub-location and sub-scale.

**Conditional** (`condition(cond_idx, x_cond)`):

$$
X_1 \mid X_2 = x_2 \;\sim\;
t_{\nu + q}\!\left(\; \mu_{1\mid 2},\;
\frac{\nu + d^2}{\nu + q}\, S_{11\cdot 2} \;\right).
$$

Two things change relative to the Gaussian, and both are data-dependent:

- the degrees of freedom rise to $\nu + q$ (thinner conditional tails), and
- the scale is the Schur complement inflated by the factor $(\nu + d^2)/(\nu + q)$,
  which **grows with the Mahalanobis distance** $d^2$ of the observed $x_2$ - i.e.
  conditioning on an outlier widens the predictive distribution
  (conditional heteroskedasticity).

Letting $\nu \to \infty$ sends the inflation factor to $1$ and the degrees of
freedom to infinity, so the Student-t conditional reduces to the Gaussian one.

## Multivariate Generalized Hyperbolic

Object: `MultivariateGH(lambda_, chi, psi, loc, scale, gamma)`. The GH is a normal
mean-variance mixture (NMVM): with a Generalized Inverse Gaussian mixing scalar
$W \sim \mathrm{GIG}(\lambda, \chi, \psi)$,

$$
X = \mu + W\,\gamma + \sqrt{W}\, A Z,
\qquad A A^{\mathsf T} = \Sigma,\;\; Z \sim \mathcal N(0, I).
$$

Parameters: GIG index $\lambda$ (`lambda_`), GIG parameters $\chi, \psi \ge 0$
(not both zero), location $\mu$ (`loc`), dispersion $\Sigma$ (`scale`), and the
**skewness vector** $\gamma$ (`gamma`); $\gamma = 0$ is symmetric. The family
subsumes the Gaussian, Student-t, variance-gamma, NIG and hyperbolic laws.

**Density** in dimension $d$, with $Q(x) = (x-\mu)^{\mathsf T}\Sigma^{-1}(x-\mu)$
and $p = \gamma^{\mathsf T}\Sigma^{-1}\gamma$:

$$
f(x) = c\;
\frac{K_{\lambda - d/2}\!\big(\sqrt{(\chi + Q)(\psi + p)}\big)\,
      e^{(x-\mu)^{\mathsf T}\Sigma^{-1}\gamma}}
     {\big(\sqrt{(\chi + Q)(\psi + p)}\big)^{d/2 - \lambda}},
\qquad
c = \frac{(\psi/\chi)^{\lambda/2}\,(\psi + p)^{d/2 - \lambda}}
         {(2\pi)^{d/2}\,|\Sigma|^{1/2}\,K_\lambda(\sqrt{\chi\psi})},
$$

where $K$ is the modified Bessel function of the second kind (evaluated stably via
the exponentially-scaled `scipy.special.kve`).

**Moments** use the GIG moments of $W$ (Bessel-K ratios at $\eta=\sqrt{\chi\psi}$):

$$
\mathbb E[X] = \mu + \mathbb E[W]\,\gamma,
\qquad
\operatorname{Cov}[X] = \mathbb E[W]\,\Sigma + \operatorname{Var}[W]\,\gamma\gamma^{\mathsf T},
$$

so `cov()` differs from `scale` both by the scalar $\mathbb E[W]$ and by a
rank-contribution along $\gamma$.

**Marginal** (`marginal(idx)`): keeps $(\lambda, \chi, \psi)$ and takes the
sub-vectors of $\mu, \gamma$ and the sub-matrix of $\Sigma$.

**Conditional** (`condition(cond_idx, x_cond)`): the GIG is conjugate to
conditioning, so $X_1 \mid X_2 = x_2$ is again GH with

$$
\lambda^\* = \lambda - q/2,\qquad
\chi^\* = \chi + d^2,\qquad
\psi^\* = \psi + \gamma_2^{\mathsf T} S_{22}^{-1}\gamma_2,
$$

$$
\mu^\* = \mu_{1\mid 2},\qquad
\Sigma^\* = S_{11\cdot 2},\qquad
\gamma^\* = \gamma_1 - S_{12} S_{22}^{-1}\gamma_2 .
$$

These same formulas specialize to every sub-family. In particular the symmetric
Student-t is the GH point $\gamma = 0,\ \psi = 0,\ \lambda = -\nu/2,\ \chi = \nu$;
applying the rules above gives $\lambda^\* = -(\nu+q)/2$ and $\chi^\* = \nu + d^2$,
which is exactly the Student-t conditional $t_{\nu+q}(\mu_{1\mid 2},
\frac{\nu+d^2}{\nu+q} S_{11\cdot 2})$ from the section above.

```{important}
$\psi = 0$ is a degenerate boundary of the GIG (it collapses to an
inverse-gamma), where the generic Bessel evaluations break down. `MultivariateGH`
handles the symmetric-t boundary ($\psi=0,\ \gamma=0$) with a dedicated branch that
delegates to `MultivariateStudentT`; the skew-t boundary ($\psi=0,\ \gamma\neq 0$)
is not yet supported - use a small $\psi>0$.
```

## Mixtures

Objects: `MixtureDistribution(weights, components)` (family-agnostic) and
`GaussianMixtureDistribution` (a back-compatible subclass of scikit-learn's
`GaussianMixture`). A mixture has weights $w_k \ge 0$, $\sum_k w_k = 1$, and
components $f_k$ of one family:

$$
f(x) = \sum_{k} w_k\, f_k(x).
$$

**Conditioning** reweights the components by how well each one explains the
observed $x_2$, then conditions each component. With $f_k^{(2)}$ the marginal of
component $k$ over the conditioning block,

$$
w_k' \;\propto\; w_k\, f_k^{(2)}(x_2),
\qquad \sum_k w_k' = 1,
$$

$$
X_1 \mid X_2 = x_2 \;\sim\; \sum_k w_k'\, \big[\,f_k \mid X_2 = x_2\,\big],
$$

where each conditioned component $f_k \mid X_2 = x_2$ follows the per-family rule
above. This single reweight-then-condition scheme is what lets the same
conditioner serve Gaussian and Student-t (and, later, Generalized Hyperbolic)
mixtures unchanged.

```{note}
`MixtureConditioner` (aliased as `GMMConditioner`) implements this scheme.
A fitted sklearn `GaussianMixture` takes a cached fast path and returns a
`GaussianMixtureDistribution`; a `MixtureDistribution` or any estimator exposing
`to_mixture_distribution()` (such as `StudentTMixture`) takes the generic path and
returns a `MixtureDistribution` whose components remain in the original family.
```

## Estimation - EM for the Student-t mixture

`StudentTMixture` fits weights $\{w_k\}$, locations $\{\mu_k\}$, scales
$\{\Sigma_k\}$ and degrees of freedom $\{\nu_k\}$ by maximum likelihood with the
EM algorithm (McLachlan & Peel; ECME for $\nu$). It subclasses scikit-learn's
`BaseMixture`, so `n_init`, `tol`, `max_iter`, k-means initialization and
convergence bookkeeping are inherited; the parameter `dof` selects whether $\nu$
is `"free"` (one per component), `"shared"` (a single $\nu$) or a fixed float.

### Latent-variable form

The Student-t is a *scale mixture of normals*. Introduce, per observation $x_i$
and component $k$, a latent positive scalar $W_{ik}$:

$$
x_i \mid (z_i = k,\, W_{ik}) \sim \mathcal N(\mu_k,\; W_{ik}\,\Sigma_k),
\qquad
W_{ik} \sim \text{InverseGamma}\!\big(\tfrac{\nu_k}{2}, \tfrac{\nu_k}{2}\big),
$$

with $z_i$ the component label. Marginalizing $W_{ik}$ recovers
$x_i \mid z_i = k \sim t_{\nu_k}(\mu_k, \Sigma_k)$. Working with the precision
$u = 1/W$ (which is Gamma-distributed) makes the updates closed-form.

### E-step

With $p$ the dimension and $\delta_{ik} = (x_i-\mu_k)^{\mathsf T}\Sigma_k^{-1}(x_i-\mu_k)$
the squared Mahalanobis distance, the E-step computes three quantities at the
*current* parameters:

$$
\tau_{ik} = \frac{w_k\, t_{\nu_k}(x_i \mid \mu_k, \Sigma_k)}
                 {\sum_j w_j\, t_{\nu_j}(x_i \mid \mu_j, \Sigma_j)}
\qquad\text{(responsibility)}
$$

$$
u_{ik} \equiv \mathbb E[\,1/W_{ik} \mid x_i, z_i=k\,]
       = \frac{\nu_k + p}{\nu_k + \delta_{ik}}
$$

$$
\mathbb E[\log u_{ik} \mid x_i, z_i=k]
   = \psi\!\Big(\tfrac{\nu_k + p}{2}\Big) - \log\!\Big(\tfrac{\nu_k + \delta_{ik}}{2}\Big),
$$

where $\psi$ is the digamma function. Note $u_{ik}$ *down-weights* outliers (large
$\delta_{ik}$) - the mechanism behind the t's robustness.

### M-step

Let $n_k = \sum_i \tau_{ik}$. The weights and the weighted-Gaussian
location/scale updates are

$$
w_k = \frac{n_k}{n},
\qquad
\mu_k = \frac{\sum_i \tau_{ik}\, u_{ik}\, x_i}{\sum_i \tau_{ik}\, u_{ik}},
\qquad
\Sigma_k = \frac{1}{n_k}\sum_i \tau_{ik}\, u_{ik}\,(x_i-\mu_k)(x_i-\mu_k)^{\mathsf T}.
$$

(`reg_covar` is added to the diagonal of $\Sigma_k$ for numerical stability.)
The degrees of freedom $\nu_k$ are updated by solving the 1-D equation

$$
\log\!\Big(\tfrac{\nu_k}{2}\Big) - \psi\!\Big(\tfrac{\nu_k}{2}\Big) + 1
 + \frac{1}{n_k}\sum_i \tau_{ik}\big(\mathbb E[\log u_{ik}] - u_{ik}\big) = 0
$$

for $\nu_k$ (Brent's method). The left side is strictly decreasing in $\nu_k$, so
the root is unique when it exists; when the data are effectively Gaussian no
finite root exists and $\nu_k$ is driven to a large cap (the $\nu \to \infty$
limit). For `dof="shared"` the same equation is solved once with the sums pooled
over all components; for a fixed float `dof` the step is skipped.

```{note}
Because the $\nu$ update uses $u_{ik}$ and $\mathbb E[\log u_{ik}]$ evaluated at
the **old** parameters, the procedure is a genuine EM and the observed-data
log-likelihood is non-decreasing every iteration. The per-iteration values are
recorded in `lower_bounds_`.
```

### Initialization

`init_params` controls the starting responsibilities: the scikit-learn options
(`"kmeans"`, `"k-means++"`, `"random"`, `"random_from_data"`) or `"gmm"`, which
warm-starts from a fitted `GaussianMixture` and tends to converge fastest. The
degrees of freedom start large (near-Gaussian) and are pulled down by the data.

## Top-level regressors

Two ways to model $p(y \mid X)$ with these families:

- **Joint + condition** - `ConditionalMixtureRegressor(family=...)` fits a joint
  mixture over $[X, y]$ and conditions. `family="gaussian"` is the classic
  conditional GMM (also available as the back-compatible `ConditionalGMMRegressor`
  preset); `family="student_t"` is exposed as `ConditionalStudentTRegressor`. The
  Student-t conditional is heteroskedastic, so `predict_cov` returns per-sample
  covariances (requires conditional $\nu + q > 2$).
- **Mixture of experts** - `MixtureOfExpertsRegressor(expert=...)` keeps the
  softmax gating and affine expert means; `expert="student_t"` swaps the Gaussian
  emission for a Student-t. Its EM reuses the same latent-scale weights $u_{ik}$ as
  above: each expert's weighted least squares and scale update are reweighted by
  $\tau_{ik} u_{ik}$ (iteratively-reweighted least squares - outliers are
  down-weighted), and per-expert $\nu_k$ solves the same digamma equation. Here the
  scale $\Sigma_k$ is constant in $x$ (only the gating and mean depend on $x$),
  unlike the joint-conditioning route above.

The Generalized Hyperbolic family is fitted by `GeneralizedHyperbolicMixture` and
exposed for regression as `ConditionalGHRegressor` (also `family="gh"`).

### Estimation - EM for the GH mixture (MCECM)

The GH mixture is fitted by an EM that augments the data with both the component
label and the latent GIG scale $W$. Writing $Q_{ik} = (x_i - \mu_k)^{\mathsf T}
\Sigma_k^{-1}(x_i - \mu_k)$ and $p_k = \gamma_k^{\mathsf T}\Sigma_k^{-1}\gamma_k$,
the posterior of the mixing scalar is again GIG,

$$
W \mid (x_i, z_i = k) \sim \mathrm{GIG}\big(\lambda_k - \tfrac d2,\; \chi_k + Q_{ik},\; \psi_k + p_k\big),
$$

so the **E-step** computes the responsibilities $\tau_{ik}$ together with
$a_{ik}=\mathbb E[W]$, $b_{ik}=\mathbb E[1/W]$ and $c_{ik}=\mathbb E[\log W]$ from
that GIG (Bessel-K ratios; $\mathbb E[\log W]$ by differentiating
$\log K_\lambda$). With $n_k = \sum_i \tau_{ik}$ and bars denoting
$\tau$-weighted averages, the **M-step** has closed forms

$$
\gamma_k = \frac{\overline{b x} - \bar b\,\bar x}{1 - \bar a\,\bar b},
\qquad
\mu_k = \bar x - \bar a\,\gamma_k,
\qquad
\Sigma_k = \frac1{n_k}\sum_i \tau_{ik}\, b_{ik}\,(x_i-\mu_k)(x_i-\mu_k)^{\mathsf T}
           - \bar a\,\gamma_k\gamma_k^{\mathsf T},
$$

and the GIG parameters $(\lambda_k, \chi_k, \psi_k)$ are updated by maximizing the
GIG expected complete-data log-likelihood under the **identifiability constraint
$\mathbb E[W]=1$** (the GH parameterization is otherwise redundant: $W$ can be
rescaled against $\Sigma,\gamma$). Imposing $\mathbb E[W]=1$ reduces the GIG update
to a 2-D maximization over $(\lambda, \omega)$ with $\omega=\sqrt{\chi\psi}$,
$\chi=\omega/R_\lambda(\omega)$, $\psi=\omega R_\lambda(\omega)$, and
$R_\lambda(\omega)=K_{\lambda+1}(\omega)/K_\lambda(\omega)$. The observed-data
log-likelihood is non-decreasing each iteration (recorded in `lower_bounds_`).
