# Examples

The following sections contain configuration files and example command usage for full processing of different VAMPIRES data.

## 1. VAMPIRES Multiband Imaging

This example will reduce coronagraphic multiband imaging data and prepare it for further ADI/SDI post-processing. This data has satellite spots for alignment and photometry, and we'll calibrate to units of contrast.

TODO get config file!


    dpp sort *.fits
    dpp calib -o master_cals back dark/**/*.fits skies/**/*.fits
    cd <data folder>
    dpp centroid <config> Open*/*.fits
    OMP_NUM_THREADS=1 dpp run -j4 <config> Open*/*.fits

## 2. (Old) VAMPIRES PDI

## 3. VAMPIRES Halpha 