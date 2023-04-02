# Pipeline

The VAMPIRES data processing pipeline uses a configuration file to automate the bulk reduction of VAMPIRES data. To run the pipeline, use the `dpp run` script

{{dpprun_help}}

or create a `Pipeline` object and call the `run` method.

## Processing steps

The pipeline will reduce the data in the following order
1. Calibration
2. Frame Selection
3. Image Registration
4. Collapsing
6. Polarimetric differential imaging

```{admonition} Warning: Large data volume
:class: warning
This pipeline tries to minimize the number of FITS files saved to disk due to the massive volume of VAMPIRES data. To accomplish this, we skip saving intermediate files when possible. Still, you should expect your data volume to increase by a factor of \~2.5. If saving intermediate products, this factor increases to \~6.5 times the raw data size. It is strongly recommended to work with a large attached storage.
```

```{admonition} Troubleshooting
:class: tip
If you run into problems, take a look at the debug file, which will be saved to the same directory as the input config file with `_debug.log` appended to the file name as well as in the output directory.

    dpp run config.toml VMPA*.fits
    tail config_debug.log
```


## Configuration

The configuration file uses the [TOML](https://toml.io) format. There are many options that have defaults, sometimes even sensible ones. In general, if an entire section is missing, the operation will be excluded. Note that sections for the configuration (e.g., `[calibration]`, `[frame_selection]`) can be in any order, although keeping them in the same order as the pipeline execution may be clearer.

We use [semantic versioning](https://semver.org/), so there are certain guarantees about the backwards compatibility of the pipeline, which means you don't have to have the exact same version of `vampires_dpp` as in the configuration- merely a version that is compatible.

### General Options


```{eval-rst}
.. autoclass:: vampires_dpp.pipeline.pipeline.Pipeline
    :members:
.. autoclass:: vampires_dpp.pipeline.config.CoordinateOptions
.. autoclass:: vampires_dpp.pipeline.config.ProductOptions
```

### Coronagraph Options

```{eval-rst}
.. autoclass:: vampires_dpp.pipeline.config.CoronagraphOptions
.. autoclass:: vampires_dpp.pipeline.config.SatspotOptions
```

### Calibration Options

```{eval-rst}
.. autoclass:: vampires_dpp.pipeline.config.CalibrateOptions
.. autoclass:: vampires_dpp.pipeline.config.DistortionOptions
```

### Frame Selection Options

```{eval-rst}
.. autoclass:: vampires_dpp.pipeline.config.FrameSelectOptions
```

### Image Registration Options

```{eval-rst}
.. autoclass:: vampires_dpp.pipeline.config.RegisterOptions
```
### Collapsing Options

```{eval-rst}
.. autoclass:: vampires_dpp.pipeline.config.CollapseOptions
```

### Polarimetry Options

```{eval-rst}
.. autoclass:: vampires_dpp.pipeline.config.PolarimetryOptions
.. autoclass:: vampires_dpp.pipeline.config.IPOptions
```


## Templates

### Polarimetric Differential Imaging (PDI)

{{pdi_toml}}

### Single-cam

For single-cam data you can elude all of the `cam2` keys in the configuration.

{{singlecam_toml}}

### Spectral Differential Imaging (SDI)

```{admonition} Warning: WIP
:class: warning

The full functionality for SDI is not implemented, but the collapsed data can be used as inputs to manual reductions, for now.
```

SDI reduction looks a lot like PDI reduction without the polarimetry.

{{sdi_toml}}
