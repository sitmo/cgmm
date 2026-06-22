"""Generate docs/examples/heavy_tailed_regression.ipynb.

Kept in the repo so the example notebook is reproducible. Run with:
    poetry run python tools/make_example_notebook.py
"""

import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
cells = []

cells.append(
    new_markdown_cell(
        "# Heavy-tailed & skewed conditional regression\n"
        "\n"
        "Real residuals are often **heavy-tailed** (occasional large errors) and "
        "**skewed** (asymmetric). A Gaussian conditional model under-states the tails "
        "and cannot represent asymmetry. `cgmm` provides three drop-in conditional "
        "regressors that share the same API but differ in the family of the joint "
        "mixture they fit and then condition:\n"
        "\n"
        "| Regressor | Family | Captures |\n"
        "|---|---|---|\n"
        "| `ConditionalGMMRegressor` | Gaussian | mean, (co)variance |\n"
        "| `ConditionalStudentTRegressor` | Student-t | + heavy tails |\n"
        "| `ConditionalGHRegressor` | Generalized Hyperbolic | + skewness |\n"
        "\n"
        "See [Distributions & Conditioning](../conventions.md) for the equations and "
        "the EM algorithms. Below we fit all three on synthetic data whose noise is "
        "both heavy-tailed and right-skewed, and compare their conditional densities "
        "and held-out log-likelihood."
    )
)

cells.append(
    new_code_cell(
        "import matplotlib\n"
        "matplotlib.use('module://matplotlib_inline.backend_inline')  # capture figures "
        "inline (must be before pyplot import)\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from scipy.stats import skewnorm, chi2\n"
        "\n"
        "from cgmm import (\n"
        "    ConditionalGMMRegressor,\n"
        "    ConditionalStudentTRegressor,\n"
        "    ConditionalGHRegressor,\n"
        ")\n"
        "\n"
        "rng = np.random.default_rng(7)\n"
        "\n"
        "def make_data(n, seed, df=4, skew=6.0, scale=0.55):\n"
        "    rs = np.random.default_rng(seed)\n"
        "    x = rs.uniform(-3, 3, size=n)\n"
        "    # skew-t noise: skew-normal divided by sqrt(chi2/df) -> heavy-tailed + skewed\n"
        "    z = skewnorm(a=skew).rvs(n, random_state=seed)\n"
        "    v = chi2(df=df).rvs(n, random_state=seed + 1)\n"
        "    noise = scale * z / np.sqrt(v / df)\n"
        "    y = 1.2 * x + noise\n"
        "    return x.reshape(-1, 1), y\n"
        "\n"
        "X_train, y_train = make_data(1200, seed=0)\n"
        "X_test, y_test = make_data(1200, seed=100)\n"
        "print('train:', X_train.shape, 'test:', X_test.shape)"
    )
)

cells.append(
    new_markdown_cell(
        "## Fit the three regressors\n"
        "\n"
        "Identical call pattern - only the class differs. We use a single mixture "
        "component so the comparison isolates the *noise family* rather than "
        "multi-modality."
    )
)

cells.append(
    new_code_cell(
        "models = {\n"
        "    'Gaussian': ConditionalGMMRegressor(n_components=1, random_state=0),\n"
        "    'Student-t': ConditionalStudentTRegressor(n_components=1, random_state=0,\n"
        "                                              max_iter=300),\n"
        "    'GH': ConditionalGHRegressor(n_components=1, random_state=0, max_iter=150),\n"
        "}\n"
        "COLORS = {'Gaussian': 'C3', 'Student-t': 'C0', 'GH': 'C2'}\n"
        "for m in models.values():\n"
        "    m.fit(X_train, y_train)\n"
        "\n"
        "print('Held-out mean log-likelihood  (higher is better):')\n"
        "for name, m in models.items():\n"
        "    print(f'  {name:10s}: {m.score(X_test, y_test):+.4f}')"
    )
)

cells.append(
    new_markdown_cell(
        "The Student-t and GH models assign higher likelihood to the held-out data: "
        "they do not waste probability mass trying to explain the heavy, asymmetric "
        "tails with a Gaussian.\n"
        "\n"
        "## How well does each model fit the noise?\n"
        "\n"
        "Because the noise is i.i.d., the residuals $r = y - 1.2x$ are draws from one "
        "skewed, heavy-tailed distribution. Each model's *fitted* conditional density "
        "(evaluated at $x=0$, where the mean is just the intercept) should match the "
        "shape of those residuals. We overlay them on the residual histogram twice:\n"
        "\n"
        "- **linear scale** - exposes the asymmetry (skew) near the peak;\n"
        "- **log scale** - exposes the tail decay (fat tails)."
    )
)

cells.append(
    new_code_cell(
        "r = y_test - 1.2 * X_test[:, 0]                      # i.i.d. noise residuals\n"
        "# focus the view on the central 99% (a few extreme outliers would otherwise\n"
        "# stretch the axes and hide the shape -- the heavy tail is still visible here)\n"
        "qlo, qhi = np.percentile(r, [0.5, 99.5])\n"
        "grid = np.linspace(qlo - 0.5, qhi + 1.0, 600)\n"
        "rng_plot = (grid.min(), grid.max())\n"
        "# each model's fitted noise density = conditional density at x = 0\n"
        "dens = {name: np.exp(m.condition(np.array([[0.0]])).logpdf(grid.reshape(-1, 1)))\n"
        "        for name, m in models.items()}\n"
        "\n"
        "fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.5))\n"
        "for ax, yscale in [(axL, 'linear'), (axR, 'log')]:\n"
        "    ax.hist(r, bins=80, range=rng_plot, density=True, color='lightgray',\n"
        "            edgecolor='none', label='residuals')\n"
        "    for name, d in dens.items():\n"
        "        ax.plot(grid, d, color=COLORS[name], lw=2, label=name)\n"
        "    ax.set_xlabel('residual  r = y - 1.2x'); ax.set_yscale(yscale)\n"
        "    ax.set_xlim(*rng_plot)\n"
        "    ax.set_title(f'fitted noise density ({yscale} scale)')\n"
        "axL.set_ylabel('density'); axL.legend()\n"
        "axR.set_ylim(1e-4, None); axR.legend()\n"
        "fig.tight_layout()"
    )
)

cells.append(
    new_markdown_cell(
        "On the **linear** panel the Gaussian (red) is symmetric and too flat - it "
        "misses the tall, asymmetric peak. The Student-t (blue) is more peaked and a "
        "little heavier-tailed, but still symmetric. The GH (green) leans into the "
        "skew and matches the asymmetric peak best. On the **log** panel the Gaussian "
        "and Student-t right tails fall away while the GH tracks the long, heavy right "
        "tail; and where the noise is skewed the differences are starkest - note the "
        "GH's short *left* tail versus the Gaussian's symmetric, data-free left tail.\n"
        "\n"
        "## Predictive mean and 5-95% bands\n"
        "\n"
        "Back in data space: we sample $p(y \\mid x)$ on a grid and take percentiles. "
        "Heavy-tailed / skewed models produce wider, asymmetric predictive bands."
    )
)

cells.append(
    new_code_cell(
        "xg = np.linspace(-3, 3, 60).reshape(-1, 1)\n"
        "fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)\n"
        "for ax, (name, m) in zip(axes, models.items()):\n"
        "    samples = m.sample(xg, n_samples=4000)        # (n, n_samples, 1)\n"
        "    lo, hi = np.percentile(samples[:, :, 0], [5, 95], axis=1)\n"
        "    ax.scatter(X_train[:, 0], y_train, s=6, alpha=0.2, color='gray')\n"
        "    ax.fill_between(xg[:, 0], lo, hi, alpha=0.30, color=COLORS[name],\n"
        "                    label='5-95%')\n"
        "    ax.plot(xg[:, 0], m.predict(xg), color='k', lw=2, label='mean')\n"
        "    ax.set_title(name); ax.set_xlabel('x'); ax.set_ylim(-6, 9)\n"
        "axes[0].set_ylabel('y'); axes[0].legend(loc='upper left')\n"
        "fig.tight_layout()"
    )
)

cells.append(
    new_markdown_cell(
        "## Takeaways\n"
        "\n"
        "- The three regressors are interchangeable in code (`fit`/`predict`/`sample`/"
        "`score`/`condition`); pick the family that matches your residuals.\n"
        "- `ConditionalStudentTRegressor` is the robust choice for heavy tails; "
        "`ConditionalGHRegressor` adds skewness.\n"
        "- All return a conditional mixture you can sample from or evaluate, and they "
        "integrate with scikit-learn pipelines and `GridSearchCV` like the Gaussian "
        "model.\n"
        "\n"
        "For the mathematics - the densities, the closed-form conditioning, and the EM "
        "algorithms - see [Distributions & Conditioning](../conventions.md)."
    )
)

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

# Assign stable cell ids (nbformat >= 4.5 requires them; avoids MissingIDFieldWarning)
for i, cell in enumerate(nb["cells"]):
    cell["id"] = f"cell-{i:02d}"

nbf.validate(nb)
out = "docs/examples/heavy_tailed_regression.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print("wrote", out)
