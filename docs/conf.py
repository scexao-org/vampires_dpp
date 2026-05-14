import os
from datetime import date
from importlib.metadata import PackageNotFoundError, version

# -- Project information -----------------------------------------------------
try:
    __version__ = version("vampires_dpp")
except PackageNotFoundError:
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
    "sphinx_autodoc_typehints",
    "sphinx_click",
    "sphinxcontrib.autodoc_pydantic",
    "sphinxcontrib.eval",
]
myst_enable_extensions = ["dollarmath", "substitution"]
myst_heading_anchors = 2
source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}
nb_execution_mode = "cache"
nb_execution_show_tb = os.environ.get("CI", "false") == "true"
nb_execution_timeout = 600

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_typehints = "description"
autodoc_typehints_format = "short"

# -- autodoc_pydantic --------------------------------------------------------
# Render Pydantic ``Field(description=...)`` values as field docstrings
# without cluttering the page with JSON schema or validator summaries.
autodoc_pydantic_model_show_json = False
autodoc_pydantic_model_show_config_summary = False
autodoc_pydantic_model_show_config_member = False
autodoc_pydantic_model_show_validator_summary = False
autodoc_pydantic_model_show_validator_members = False
autodoc_pydantic_model_show_field_summary = False
autodoc_pydantic_model_member_order = "bysource"
autodoc_pydantic_field_list_validators = False
autodoc_pydantic_field_show_constraints = False
autodoc_pydantic_field_doc_policy = "description"

# -- Options for HTML output -------------------------------------------------

html_static_path = ["_static"]
html_title = "VAMPIRES DPP"
html_theme = "sphinx_book_theme"
html_logo = "scexao_logo.svg"
html_theme_options = {
    "github_url": "https://github.com/scexao-org/vampires_dpp",
    "repository_url": "https://github.com/scexao-org/vampires_dpp",
    "use_repository_button": True,
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_fullscreen_button": False,
}
