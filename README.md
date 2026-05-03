# VAMPIRES Data Processing Pipeline

[![CI tests](https://github.com/scexao-org/vampires_dpp/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/scexao-org/vampires_dpp/actions/workflows/CI.yml)
[![Docs](https://github.com/scexao-org/vampires_dpp/actions/workflows/docs.yml/badge.svg?branch=main)](https://scexao-org.github.io/vampires_dpp)
[![Coverage](https://codecov.io/gh/scexao-org/vampires_dpp/branch/main/graph/badge.svg)](https://codecov.io/gh/scexao-org/vampires_dpp)
[![License](https://img.shields.io/github/license/scexao-org/vampires_dpp?color=yellow)](LICENSE)
[![Python versions](https://img.shields.io/badge/python-%3E%3D3.10-3776AB?logo=python&logoColor=white)](https://github.com/scexao-org/vampires_dpp/actions/workflows/CI.yml)

 `vampires_dpp` is still under development, and the API can change without notice. Use with your own caution (and please report any bugs you find).

**Primary maintainer:** [Miles Lucas](https://github.com/scexao-org)

(Experimental) Tools for processing VAMPIRES data

WARNING: version 0.13.0 has breaking changes that have not been fully documented! Contact Miles for more details.

## Installation

Install directly from GitHub using `uv` (recommended) or `pip`:

```sh
uv add git+https://github.com/scexao-org/vampires_dpp
# or
pip install git+https://github.com/scexao-org/vampires_dpp
```

For local development, clone the repository and sync the environment:

```sh
git clone https://github.com/scexao-org/vampires_dpp
cd vampires_dpp
uv sync --extra dev
pre-commit install
```

Run the test suite with:

```sh
uv run --extra test pytest
```

## Citing

If you use `vampires_dpp` in your research, please consider citing it as software with the following DOI: [TODO](https://github.com/scexao-org/vampires_dpp/blob/main/CITAIONS.bib)

## License

`vampires_dpp` is licensed under the MIT open-source license. See [LICENSE](LICENSE) for more details.

## Contributing and Support

If you would like to contribute, feel free to open a [pull request](https://github.com/scexao-org/vampires_dpp/pulls). If you want to discuss something before contributing, head over to [discussions](https://github.com/scexao-org/vampires_dpp/discussions) and join or open a new topic. If you're having problems with something, please open an [issue](https://github.com/scexao-org/vampires_dpp/issues).
