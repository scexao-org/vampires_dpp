# Installation


## Using `pip`

```{margin} Python Version
`vampires_dpp` requires at least python 3.7
```

For now, install directly from GitHub

```bash
pip install -U git+https://github.com/scexao-org/vampires_dpp.git
```

## From Source

The source code for `vampires_dpp` can be downloaded and installed [from GitHub](https://github.com/scexao-org/vampires_dpp) by running

```bash
git clone https://github.com/scexao-org/vampires_dpp
cd vampires_dpp
pip install -e .
```

## Testing

To run the unit tests, install the development dependencies using pip:

```bash
pip install -e .[test]
```

and then execute:

```bash
pytest
```

This will automatically run the tests with plugins enabled. All of the tests should (of course) pass. If any of the tests don't pass and if you can't sort out why, [open an issue on GitHub](https://github.com/scexao-org/vampires_dpp/issues).

## Contributing

If you would like to contribute, first off, thank you! To get started, you should install our git [pre-commit hooks](https://pre-commit.com/) which autoformat the repository. After you have `pre-commit` installed, run

```bash
cd vampires_dpp
pre-commit install
```

and you're all set! Now whenever you `git commit` the source files will be autoformatted using [`black`](https://github.com/psf/black).

Any contributions should be submitted as [pull requests](https://github.com/scexao-org/vampires_dpp/pulls). Feel free to reach out ahead of time about questions or ambitions about contributing.

## Documentation

To build these docs locally, first install the documentation dependencies

```bash
pip install -e .[docs]
```

Then, run the [sphinx](https://www.sphinx-doc.org/en/master/) make script

```bash
sphinx-build docs docs/_build
```
