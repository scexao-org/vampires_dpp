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
│   └── raw
├── HD31233
│   └── raw
├── darks
├── flats
├── pinholes
└── skies
```
after sorting this folders can be changed or rearranged as much as you'd like. The configuration for the pipeline is flexible, so you don't have to sort your files at all if you prefer a different method.

### Reference

```
dpp sort --help
usage: dpp sort [-h] [-o OUTPUT] [-c] filenames [filenames ...]

positional arguments:
  filenames             FITS files to sort

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        output directory, if not specified will use current
                        working directory
  -c, --copy            copy files instead of moving them
```

## Set up configuration files

After your data has been downloaded and sorted, you'll want to create configuration files for the data you want to process. To get started quickly, we provide templates for common observing scenarios that can be produced with `dpp new`. In the example below, we are creating a PDI template with the 55 mas Lyot coronagraph.

```
dpp new 20230101_ABAur_VAMPIRES.toml -n 20230101_ABAur -t pdi -c 55 --show
```
```
filenames = "raw/VMPA*.fits"
name = "20230101_ABAur"
target = ""

[frame_centers]
cam1 = []
cam2 = []

[coronagraph]
iwa = 55.0

[satspot]
radius = 15.9
angle = -4
amp = 50

[calibrate]
output_directory = "calibrated"

[calibrate.master_darks]
cam1 = "../darks/master_dark_cam1.fits"
cam2 = "../darks/master_dark_cam2.fits"

[calibrate.master_flats]
cam1 = "../flats/master_flat_cam1.fits"
cam2 = "../flats/master_flat_cam2.fits"

[coregister]
output_directory = "coregistered"

[collapse]
output_directory = "collapsed"

[polarimetry]
output_directory = "pdi"
```

At this point, we highly recommend viewing the [pipeline options]() and making adjustments to your TOML file for your object and observation. The processing pipeline is not a panacea- the defaults in the templates are best guesses in ideal situations.

### Reference

```
dpp new --help
usage: dpp new [-h] -t {singlecam,pdi,all} [-n NAME] [-c IWA] [-p] config

positional arguments:
  config                path to configuration file

options:
  -h, --help            show this help message and exit
  -t {singlecam,pdi,all}, --template {singlecam,pdi,all}
                        template configuration to make
  -n NAME, --name NAME  name of configuration
  -c IWA, --coronagraph IWA
                        if coronagraphic, specify IWA (mas)
  -p, --preview         display generated TOML
```

## Running the pipeline

After you've selected your configuration options, you can run the pipeline from the command line with `dpp run`

```
dpp run 20230101_ABAur_VAMPIRES.toml
```

