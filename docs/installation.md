(installation)=
# Installation

## Python Versions

```{admonition} Managing python versions
:class: tip

Subaru's archive interface depends on python 2, while this package requires at least python 3.10. You will inevitably need to manage different python versions. We recommend using [uv](https://docs.astral.sh/uv/) to manage Python installations and virtual environments.
```

## From GitHub

Install directly from GitHub using `uv` (recommended) or `pip`:

```bash
uv add git+https://github.com/scexao-org/vampires_dpp
# or
pip install git+https://github.com/scexao-org/vampires_dpp
```

## From Source

Clone the repository and install in editable mode:

```bash
git clone https://github.com/scexao-org/vampires_dpp
cd vampires_dpp
uv sync
# or
pip install -e .
```

## Testing

You can quickly check if you've installed the pipeline correctly by calling the `dpp` command

```bash
dpp --version
```

To run the unit tests:

```bash
uv run --extra test pytest
# or
pip install -e ".[test]" && pytest
```

This will automatically run the tests with plugins enabled. All of the tests should (of course) pass. If any of the tests don't pass and if you can't sort out why, [open an issue on GitHub](https://github.com/scexao-org/vampires_dpp/issues).

## Contributing

If you would like to contribute, first off, thank you! To get started, clone the repo, sync the environment, and install the [pre-commit](https://pre-commit.com/) hooks:

```bash
git clone https://github.com/scexao-org/vampires_dpp
cd vampires_dpp
uv sync --extra dev
pre-commit install
```

`uv sync --extra dev` installs all development tools (ruff, ty) into the project venv. `pre-commit install` wires up the git hooks, which call those tools via `uv run` — so there is no separate tool version to manage.

Now whenever you `git commit`, source files will be linted and formatted using [`ruff`](https://docs.astral.sh/ruff/) and the `uv.lock` file will be kept in sync. Any linting errors that cannot be auto-fixed will block `git commit` unless overridden with `git commit -n`.

Any contributions should be submitted as [pull requests](https://github.com/scexao-org/vampires_dpp/pulls). Feel free to reach out ahead of time about questions or ambitions about contributing.

## Documentation

To build these docs locally, install the documentation dependencies:

```bash
uv sync --extra docs
# or
pip install -e ".[docs]"
```

Then run the [sphinx](https://www.sphinx-doc.org/en/master/) build:

```bash
sphinx-build docs docs/_build
```

```{admonition} Local docs viewer
To quickly serve the generated HTML files you can use

    python -m http.server -d docs/_build/html 8000

(or any port you'd like) and view them in an internet browser at the url `localhost:8000/index.html`
```