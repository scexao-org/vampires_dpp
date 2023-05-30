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
5. Make difference images
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

### Difference Image Options

```{eval-rst}
.. autoclass:: vampires_dpp.pipeline.config.DiffOptions
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

The full functionality for SDI is not implemented, but the difference images can be used as inputs to manual reductions, for now.
```

SDI reduction looks a lot like PDI reduction without the polarimetry.

{{sdi_toml}}


## Frequently Asked Questions (FAQ)

### Pipeline and Configuration version mismatch

> I can't run the pipeline because it says my TOML file has a version mismatch?

In order to try and manage compatibility for the pipeline, your configuration file has a `version` key in it. This key must be compatible (within SemVer) with the installed version of `vampires_dpp`. There are two approaches to fixing this:

1. (Recommended) Call `dpp upgrade` to try to automatically upgrade your configuration
2. Downgrade `vampires_dpp` to match the version in your configuration

### Older data and PDI

> I downloaded some archival VAMPIRES data and I keep getting errors during PDI that say the HWP indices can't be ordered.

Some antique VAMPIRES data uses a different order for the HWP cycles than modern VAMPIRES data. In other words, instead of iterating between 0° (Q), 45° (-Q), 22.5° (U), 67.5° (-U), in this data the order is 0° (Q), 22.5° (U), 45° (-Q), 67.5° (-U). To fix this behavior, you can use the least-squares polarimetry method which does not need ordering, or you can set the following in your configuration-

```toml
[polarimetry]
method = "difference"
order = "QUQU"
```

### Performance

> It's slow. It's so, so slow. Help.

It's hard to process data in the volumes that VAMPIRES produces, but there are some tips for speeding it up.
1. Use an SSD (over USB 3 or thunderbolt)

Faster storage media reduces slowdowns from opening and closing files, which happens *a lot* throughout the pipeline

2. Don't save intermediate files

The time it takes to open a file, write to disk, and close it will add a lot to your overheads, in addition to the huge increase in data volume

3. Use multi-processing

Using more processes should improve some parts of the pipeline, but don't expect multiplicative increases in speed since most operations are limited by the storage IO speed.