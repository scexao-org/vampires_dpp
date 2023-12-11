# Configuration

The configuration file uses the [TOML](https://toml.io) format. There are many options that have defaults, sometimes even sensible ones. In general, if an entire section is missing, the operation will be excluded. Note that sections for the configuration (e.g., `[calibration]`, `[collapsing]`) can be in any order, although keeping them in the same order as the pipeline execution may be clearer.

We use [semantic versioning](https://semver.org/), so there are certain guarantees about the backwards compatibility of the pipeline, which means you don't have to have the exact same version of `vampires_dpp` as in the configuration- merely a version that is compatible.

## General Options

```{eval-rst}
.. autoclass:: vampires_dpp.pipeline.config.PipelineConfig
    :members:
.. autoclass:: vampires_dpp.pipeline.config.ObjectConfig
```

## Calibration Options

```{eval-rst}
.. autoclass:: vampires_dpp.pipeline.config.CalibrateConfig
```

## Frame-collapsing Options

```{eval-rst}
.. autoclass:: vampires_dpp.pipeline.config.CollapseConfig
```

## Spectrophotometric Calibration Options

```{eval-rst}
.. autoclass:: vampires_dpp.pipeline.config.SpecphotConfig
```

## Polarimetry Options

```{eval-rst}
.. autoclass:: vampires_dpp.pipeline.config.PolarimetryConfig
```
