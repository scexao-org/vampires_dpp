from pkg_resources import DistributionNotFound, get_distribution
from datetime import date

# -- Project information -----------------------------------------------------
try:
    __version__ = get_distribution("exoplanet").version
except DistributionNotFound:
    __version__ = "unknown version"

# The full version, including alpha/beta/rc tags
version = __version__
release = __version__

project = "vampires_dpp"
author = "Miles Lucas"
# get current year
current_year = date.today().year
years = range(2022, current_year + 1)
copyright = f"{', '.join(map(str, years))}, {author}"


# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "myst_nb",
]
myst_enable_extensions = [
    "dollarmath",
]
myst_heading_anchors = 2
source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_typehints = "description"
autodoc_typehints_format = "short"

# -- Options for HTML output -------------------------------------------------

html_theme = "alabaster"
html_static_path = ["_static"]
html_title = "VAMPIRES DPP"
html_theme = "sphinx_book_theme"
html_theme_options = {
    "github_url": "https://github.com/scexao-org/vampires_dpp",
    "repository_url": "https://github.com/scexao-org/vampires_dpp",
    "use_repository_button": True,
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_fullscreen_button": False,
}
