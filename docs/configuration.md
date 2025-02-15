(configuration)=
# Configuration

The configuration file uses the [TOML](https://toml.io) format. There are many options that have defaults, sometimes even sensible ones. In general, if an entire section is missing, the operation will be excluded. Note that sections for the configuration (e.g., `[calibration]`, `[polarimetry]`) can be in any order, although keeping them in the same order as the pipeline execution may be clearer.

We use [semantic versioning](https://semver.org/), so there are certain guarantees about the backwards compatibility of the pipeline, which means you don't have to have the exact same version of `vampires_dpp` as in the configuration- merely a version that is compatible.

```{admonition} File Outputs
We notate where the pipeline ends up saving data files with the "💾" emoji.
```

```{contents}
:local:
:depth: 2
```

## Pipeline
```{margin} 💾 File Output
‎
```
```{eval-rst}
.. autoclass:: vampires_dpp.pipeline.config.PipelineConfig
    :members: from_file, to_toml, save
```

### Target Information

```{eval-rst}
.. autoclass:: vampires_dpp.pipeline.config.TargetConfig
```

### File combination
```{margin} 💾 File Output
‎
```
```{eval-rst}
.. autoclass:: vampires_dpp.pipeline.config.CombineConfig
```

### Calibration
```{margin} 💾 File Output
‎
```
```{eval-rst}
.. autoclass:: vampires_dpp.pipeline.config.CalibrateConfig
```

### Analysis
```{margin} 💾 File Output
‎
```
```{eval-rst}
.. autoclass:: vampires_dpp.pipeline.config.AnalysisConfig
```

### Frame selection
```{margin} 💾 File Output
‎
```
```{eval-rst}
.. autoclass:: vampires_dpp.pipeline.config.FrameSelectConfig
```

### Frame alignment
```{margin} 💾 File Output
‎
```
```{eval-rst}
.. autoclass:: vampires_dpp.pipeline.config.AlignmentConfig
```


### Cube coadding
```{margin} 💾 File Output
‎
```
```{eval-rst}
.. autoclass:: vampires_dpp.pipeline.config.CoaddConfig
```

### Spectrophotometric Calibration

```{eval-rst}
.. autoclass:: vampires_dpp.pipeline.config.SpecphotConfig
```

### Polarimetry
```{margin} 💾 File Output
‎
```
```{eval-rst}
.. autoclass:: vampires_dpp.pipeline.config.PolarimetryConfig
```

