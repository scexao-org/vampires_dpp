# Quick start guide

We strive to make the VAMPIRES DPP as automated as possible- the following setup should get you 90% of the way towards successful data reduction, depending on the complexity of your dataset!

We assume that you have already successfully installed the `vampires_dpp` package. If not, see [Installation](_). You can quickly check by calling the `dpp` command

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
│   └── 750-50_em300_00100ms
├── HD31233
│   └── raw
├── darks
│   └── raw
├── flats
│   └── raw
├── pinholes
│   └── raw
└── skies
│   └── raw
```
after sorting this folders can be changed or rearranged as much as you'd like. The configuration for the pipeline is flexible, so you don't have to sort your files at all if you prefer a different method.

### Reference

```
dpp sort --help
usage: dpp sort [-h] [-o OUTPUT] [-c] [-e EXT] [-j NUM_PROC] [-q] filenames [filenames ...]

Sorts raw data based on the data type. This will either use the `DATA-TYP` header value or the `U_OGFNAM` header, depending on when
your data was taken.

positional arguments:
  filenames             FITS files to sort

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        output directory, if not specified will use current working directory
  -c, --copy            copy files instead of moving them
  -e EXT, --ext EXT     FITS extension/HDU to use
  -j NUM_PROC, --num-proc NUM_PROC
                        number of processors to use for multiprocessing (default is 8)
  -q, --quiet           silence the progress bar
```

## Create master calibration files

Next, you'll want to create your master darks and flats. This can be accomplished in one command using `dpp calib`.

```{admonition} Matching calibration settings
Since VAMPIRES uses EM-CCDs, the camera gain and exposure settings change the noise properties. Therefore, we automatically sort all calibration files by camera, EM gain, and exposure time. We will automatically try and match darks to flats, but if the settings are not equal the flat will not be dark-subtracted.
```

If you use the prescribed folder structure above, creating your files can be done like so
```
dpp calib --darks=darks/raw/*.fits --flats=flats/raw/*.fits -o master_cals
```

This will produce a series of calibration files in the `master_cals/` folder

```
master_cals
├── master_dark_em300_000050ms_cam1.fits
├── master_dark_em300_000050ms_cam2.fits
├── master_dark_em300_000080ms_cam1.fits
├── master_dark_em300_000080ms_cam2.fits
├── master_dark_em300_000100ms_cam1.fits
├── master_dark_em300_000100ms_cam2.fits
├── master_flat_em300_001000ms_cam1.fits
└── master_flat_em300_001000ms_cam2.fits
```

### Reference

```
dpp calib -h
usage: dpp calib [-h] [--darks [DARKS ...]] [--flats [FLATS ...]] [-c {median,mean,varmean,biweight}] [-o OUTPUT] [-f] [-j NUM_PROC]
                 [-q]

Create calibration files from darks and flats.

options:
  -h, --help            show this help message and exit
  --darks [DARKS ...]   FITS files to use as dark frames
  --flats [FLATS ...]   FITS files to use as flat frames
  -c {median,mean,varmean,biweight}, --collapse {median,mean,varmean,biweight}
  -o OUTPUT, --output OUTPUT
                        output directory, if not specified will use current working directory
  -f, --force           Force recomputation and overwrite existing files.
  -j NUM_PROC, --num-proc NUM_PROC
                        number of processors to use for multiprocessing (default is 8)
  -q, --quiet           silence the progress bar
```

## Set up configuration files

After your data has been downloaded and sorted, you'll want to create configuration files for the data you want to process. To get started quickly, we provide templates for common observing scenarios that can be produced with `dpp new`. In the example below, we are creating a PDI template with the 55 mas Lyot coronagraph.

```
dpp new 20230101_ABAur_VAMPIRES.toml -n 20230101_ABAur -t pdi --preview
```

At this point, we highly recommend viewing the [pipeline options]() and making adjustments to your TOML file for your object and observation. The processing pipeline is not a panacea- the defaults in the templates are best guesses in ideal situations.

### Reference

```
dpp new --help
usage: dpp new [-h] -t {singlecam,pdi,halpha,all} [-o OBJECT] [-c IWA] [-p] config

positional arguments:
  config                path to configuration file

options:
  -h, --help            show this help message and exit
  -t {singlecam,pdi,halpha,all}, --template {singlecam,pdi,halpha,all}
                        template configuration to make
  -o OBJECT, --object OBJECT
                        SIMBAD-compatible target name
  -c IWA, --coronagraph IWA
                        if coronagraphic, specify IWA (mas)
  -p, --preview         preview generated TOML
```

## Running the pipeline

After you've selected your configuration options, you can run the pipeline from the command line with `dpp run`

```
dpp run 20230101_ABAur_VAMPIRES.toml
```

### References

```
dpp run --help 
```
```
usage: dpp run [-h] [-j NUM_PROC] config

positional arguments:
  config                path to configuration file

options:
  -h, --help            show this help message and exit
  -j NUM_PROC, --num-proc NUM_PROC
                        number of processors to use for multiprocessing (default is 8)
```

