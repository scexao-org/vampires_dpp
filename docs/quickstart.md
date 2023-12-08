# Quick-start guide

We strive to make the VAMPIRES DPP as automated as possible- the following setup should get you 90% of the way towards successful data reduction, depending on the complexity of your dataset!

We assume that you have already successfully installed the `vampires_dpp` package. If not, see [Installation](). You can quickly check by calling the `dpp` command

```
dpp --version
```

## Sorting raw data

After downloading your data, you may want to sort it into subfolders based on the data type and object observed.

```{admonition} Pre-S23A Data types
:class: tip

Data types are not appropriately populated into headers for data taken before the S23A semester. For these data, we will try and parse a file type from the OG filename. In other words, if the VAMPIRES operator took data labeled "darks_*.fits", we will put that into the darks folder. This automation is ad-hoc and should be verified by a human after sorting.
```

After downloading your data, you can run `dpp sort` to automatically sort your data into human-readable folders, ready for further processing.

```
dpp sort VMPA*.fits
```

The prescribed folder structure for this sorting is
```
.
├── ABAUR
│   └── 750-50_em300_00100ms_512x512
├── HD31233
│   └── 750-50_em300_00100ms_512x512
├── darks
│   └── em300_00100ms_512x512
├── flats
│   └── 750-50_em300_00100ms_512x512
├── pinholes
│   └── 750-50_em300_00100ms_512x512
└── skies
    └── em300_00100ms_512x512
```
after sorting this folders can be changed or rearranged as much as you'd like. The configuration for the pipeline is flexible, so you don't have to sort your files at all if you prefer a different method.

## Normalizing Data

There are a few recognized data formats the processing pipeline recognizes. Depending on which data format you have, additional steps may need to be taken to normalize data to prepare it for the pipeline.

```{admonition} What data needs normalized?

Any EMCCD VAMPIRES format data needs normalized- at minimum it will removed the detector readout and empty frames.
```

### EMCCD Formats

These formats are used for VAMPIRES data prior to the June 2023 upgrades. They come in two distinct formats
1. STARS archive format (**default for open-use observers**)
    - Files are given frame id names or standardized archive names, e.g. "VMPA00023445.fits"
    - Each file corresponds to a single camera, FLC state, and HWP angle
2. VAMPIRES format
    - Files have names as set by controller, e.g. "HD141569A_00_750-50_LyotStop_00_cam1.fits"
    - Each file is a raw cube from the camera, which includes bad readout frame in the first frame
    - FLC states interleaved in every other frame for PDI mode

If your data is in the STARS format, *no further action is needed*- `dpp norm` can be skipped. If your data is in the VAMPIRES format, you'll need to run `dpp norm` to run some essential pre-processing steps.

At minimum, you'll need to discard the initial frames which have readout artifacts. If your data is polarimetric you will also need to deinterleave the files (add the `-d` flag). This process will also cut out any frames which are all zeros- a common occurrence when aborting exposure sequences before completion.

```
dpp norm [-d] -o normed 750-50_em300_00100ms_512x512/*.fits
```

```{admonition} What data needs deinterleaved?

Not all data needs deinterleaved- calibration files (darks, flats, pinholes, skies, etc.) typically do not need deinterleaved. If you do not plan to do polarimetry (e.g., speckle imaging, ADI-only) you can skip deinterleaving entirely, effectively averaging together data between the two FLC states. If you prefer to model each state separately then deinterleave away.
```

### CMOS Format

Any data taken after the June 2023 upgrades has the same format regardless if data is downloaded from the archive or from the SCExAO computers directly. It is assumed that any FLC deinterleaving has been done before hand, which is expected to be done by the support astronomer.

No further action is required for the CMOS format- `dpp norm` can be skipped.

## Quick look and filter

Before running files through the pipeline, it is recommended to inspect your raw data and discard errant cubes and cubes with poor seeing. Doing this ahead of time saves on processing time and avoids errors. 

## Create calibration files

Next, you'll want to create your master backgrounds and flats. This can be accomplished in one command using `dpp prep`.

If you use the prescribed folder structure above, creating your files can be done like so
```
dpp prep -o master_cals back darks/**/*.fits
dpp prep -o master_cals flat flats/**/*.fits
```

This will produce a series of calibration files in the `master_cals/` folder. We do not combine individual flat or dark frames- instead we match each science file to the closest matching available calibration file.

## Set up configuration files

After your data has been downloaded and sorted, you'll want to create configuration files for the data you want to process. To get started quickly, we provide templates for common observing scenarios that can be produced interactively with `dpp new`. In the example below, we are creating a PDI template with the 55 mas Lyot coronagraph.

```
dpp new 20230101_ABAur.toml
```

At this point, we highly recommend viewing the [pipeline options]() and making adjustments to your TOML file for your object and observation. The processing pipeline is not a panacea- the defaults in the templates are best guesses in ideal situations.

## Prepare image centroids

To allow efficient data analysis we only measure the PSF centroid and statistics in a window around an initial guess of the PSF centroid. Due to the complexity of input data types and observing scenarios we cannot easily prescribe an automated centroiding scheme. We provide tools for quick-looking data and getting precise guesses so that analysis is much more likely to succeed.

### Interactive mode

If the terminal you are connected to has working graphics (i.e., X-forwarding for SSH connections, not inside a tmux sesssion) and you have a copy of matplotlib installed you can run

```
dpp centroid 20230101_ABAur.toml 750-50_em300_00100ms_512x512/*.fits
```
and you will be prompted with matplotlib figures to click on the PSFs you want to centroid. For non-coronagraphic data you should click on the central star, and for coronagraphic data you can choose any set of four satellite spots. Multi-band imaging data will show you one field at a time.

```{admonition} Coronagraphic cam 1 orientation
:class: warning

Because camera 1 gets flipped along the y-axis during calibration, if you care about having consistently labeled PSF statistics (i.e., PSF sum for field 1 on cam 1 matches PSF sum for field 1 on cam 2) you should start with the bottom satellite PSF and go clockwise for camera 1 and use the top satellite PSF and proceed counter-clockwise for comera2
```

### Manual mode

If you do not have a graphics-ready terminal, or you want to skip the interactive process and use your own methods, we provide a manual method for entering centroids. You will be prompted for the centroid of each PSF- enter the centroid as an `x, y` comma-separated tuple. 

```{admonition} Pro-tip: DS9
:class: tip

If previewing data with DS9, you can copy the x, y coordinates directly after subtracting 1 (because python starts indexing at 0). This allows you to use region centroids or the cross for assistance with manual entry.
```


## Running the pipeline

After you've selected your configuration options, you can run the pipeline from the command line with `dpp run`

```
dpp run 20230101_ABAur.toml 750-50_em300_00010ms_512x512/*
```

```{admonition} Warning: multiprocessing with large data
:class: warning

If you have large data, you need to avoid multi-processing or you are at risk of running out of memory. This can happen with high framerate MBI data, in particular. Make sure you use the `-j` flag to limit processes to avoid running out of memory

    dpp run -j 1 20230101_ABAur.toml 750-50_em300_00010ms_512x512/*

```

### PDI

If you have polarimetry enabled in your configuration, it will run automatically with `dpp run`, but if you want to rerun the polarimetric analysis *only*, you can use

```
dpp pdi 20230101_ABAur.toml 750-50_em300_00010ms_512x512/*
```

## Re-running the pipeline

If you want to rerun the pipeline TODO
