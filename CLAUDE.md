# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VAMPIRES DPP is an astronomical data processing pipeline for the VAMPIRES instrument (Subaru Telescope). It handles calibration, frame selection, image registration, polarimetric differential imaging (PDI), and non-redundant masking (NRM) interferometry. APIs are experimental and subject to change.

## Commands

```bash
# Install for development
pip install -e ".[dev,test]"

# Run tests
pytest

# Run a single test file
pytest tests/test_registration.py

# Lint and format
ruff check src/ --fix
ruff format src/

# Pre-commit (runs ruff)
pre-commit run --all-files

# CLI entry point
dpp --help
```

## Architecture

### Configuration System
All pipeline configuration is TOML-based and validated with Pydantic v2. The main config model is `PipelineConfig` in `src/vampires_dpp/pipeline/config.py`, which composes sub-configs for each processing stage (`CombineConfig`, `SpecphotConfig`, `NRMConfig`, etc.). Users create configs via `dpp new` and execute via `dpp run <config.toml>`.

### Pipeline Execution
`Pipeline` in `src/vampires_dpp/pipeline/pipeline.py` takes a `PipelineConfig`, a working directory, and an optional `num_proc` for multiprocessing. It orchestrates the full processing sequence: calibration → frame selection → registration → coadding → PDI/NRM.

### Processing Modes
- **PDI** (`src/vampires_dpp/pdi/`): Stokes parameter computation via single/double/triple difference imaging; Mueller matrix calibration for instrumental polarization.
- **NRM** (`src/vampires_dpp/nrm/`): Non-redundant masking interferometry — phase alignment, closure phase extraction, and NRM+PDI hybrid analysis.
- **Specphot** (`src/vampires_dpp/specphot/`): Spectrophotometric calibration using synthetic photometry (`synphot`).

### Instrument Constants
`src/vampires_dpp/constants.py` defines instrument info dataclasses (`EMCCDVAMPIRES`, `CMOSVAMPIRES`, `CMOSVAMPIRESPostNBS`). `NBS_INSTALL_MJD = 60949` (2025-10-03) marks a major optical change; code uses this date to branch behavior for pre/post-NBS data.

### Key Patterns
- **Error handling:** Uses the `result` library (`Ok`/`Err` types) in several modules.
- **Logging:** `loguru` throughout; configured in `src/vampires_dpp/logging_utils.py`.
- **CLI:** `click` with subcommand groups; entry point is `dpp` → `vampires_dpp.cli.main:main`.
- **FITS I/O:** `astropy.io.fits`; WCS via `astropy.wcs`; header utilities in `src/vampires_dpp/headers.py`.

## Code Style

- Line length: 100 characters
- Docstrings: NumPy convention
- Linter/formatter: `ruff` (configured in `pyproject.toml`)
- No relative imports from parent packages (enforced by ruff)
