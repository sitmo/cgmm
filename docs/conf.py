import os
import sys
from datetime import datetime

# Ensure package import works in executed notebooks
sys.path.insert(0, os.path.abspath(".."))

project = "cgmm"
author = "Thijs van den Berg"
copyright = f"{datetime.now():%Y}, {author}"

extensions = [
    "myst_nb",  # executable Markdown notebooks
    "sphinx_copybutton",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}
# Execute code cells during build so plots are generated
nb_execution_mode = "cache"  # speed up subsequent builds
nb_execution_timeout = 600  # 10 minutes for hyperparameter optimization
nb_execution_raise_on_error = True

# MyST configuration
myst_enable_extensions = ["deflist", "colon_fence"]

# MyST-NB configuration
nb_render_code_prompt_show = "Show code cell {type}"
nb_render_code_prompt_hide = "Hide code cell {type}"

html_theme = "furo"
html_title = "cgmm"
html_static_path = []

# We want warnings if execution fails
nb_execution_raise_on_error = True

# Keep the matplotlib backend non-interactive for headless builds
os.environ.setdefault("MPLBACKEND", "Agg")
