# Getting started

## VAMPIRES Observation Modes

VAMPIRES is an incredibly diverse instrument in its usage with SCExAO. It can be used for studying evolved stellar physics, protoplanetary disks, polarimetry, H-ɑ emission, in addition to an interferometric mode via sparse aperture masking (SAM). VAMPIRES is also used for focal-plane wavefront sensing, for example using differential focus for [phase diversity](), as well as complementary data for infrared observations (i.e., telemetry). `vampires_dpp` provides tooling primarily for imaging modes, although the image-processing tools provide building blocks for a variety of applications of VAMPIRES data (and PRs are welcome for interferometric modes!). The following observation modes have well-defined tutorials and workflows, and represent the majority of open-use observations

1. Speckle imaging / telemetry (TODO)
2. Polarimetric differential imaging (TODO)
3. Coronagraphic polarimetric differential imaging (TODO)
4. H-ɑ differential imaging (TODO)

## VAMPIRES Data Formats

Depending on who you are, you may have to deal with a variety of VAMPIRES data formats. For open-use observers, this is the STARS gen-2 archive format, which is different from the format saved by VAMPIRES during acquisition, and is different from internal telemetry stream data. The `vampires_dpp` prioritizes the gen-2 format highest, but the API should be flexible enough to accommadate any user of VAMPIRES- observer or engineer.

### Conventions

The data follow these general specifications

- Parallactic angles are specified as degrees CCW from the North-celestial pole
```{margin} SAOImage DS9
These coordinates are the transpose/reverse of DS9 and minus one.
```
- Image coordinates follow *python* conventions
    - The bottom-left pixel center has coordinates $(0, 0)$
    - The natural axes for images in numpy arrays are $(y, x)$
    - The image center is therefore $(N_y - 1)/2, (N_x - 1)/2$
- Camera 1 is flipped on its y-axis compared to sky
    - The beam-splitter will naturally mirror the beam along this axis so camera 2 doesn't need flipped
    

### Formats for imaging and PDI modes

- STARS archive format (**default for open-use observers**)
    - Files are given frame id names, e.g. "VMPA0023445.fits"
    - Each file corresponds to a single camera, FLC state, and HWP angle
    - Rich FITS headers
- VAMPIRES format
    - Files have names as set by controller, e.g. "HD141569A_00_750-50_LyotStop_00_cam1.fits"
    - Each file is a raw cube from the camera, which includes bad readout frame in the first frame
    - FLC states interleaved in every other frame for PDI mode
    - Rich FITS headers
- Stream format
    - Files are given stream names, e.g. "vcamim1_12:04:34.234412598.fits"
    - Camera 1 and 2 may or may not be synced, depending on observer usage
    - No meaningful metadata present in the FITS headers (e.g., gain, HWP, camera, etc.)
    - Extremely precise frame-grabber timing

```{admonition} Multiple HDU FITS files
:class: danger 
Some old VAMPIRES data (circa 2019 and earlier) had multiple FITS header data units (HDUs) to store header information. `vampires_dpp` is not built to support these data, but a conversion script could be written if requested.
```



## Processing Workflows

`vampires_dpp` is built with multiple processing workflows in mind to support the large diversity of observing modes and usages of VAMPIRES. Typical VAMPIRES observations produce many gigabytes of data across hundreds of files, which means processing data must occur on a file-by-file basis, with the potential for multi-processing. A large portion of the `vampires_dpp` API is available in a command line interface (CLI) for quick processing of files and various other helpers. Despite the attractive ease of using the scripting interface, if you plan to publish or share the data you have reduced, please consider saving your commands in bash scripts or using the python interface and a python script so that your reductions can be shared with and reproduced by future investigators.

As alluded to above, the most direct interface with the API is, well, the Python API itself! Any script you find will be composed of functions in the `vampires_dpp` python module, which gets installed when you `pip install` this code. If you are someone who likes working in Jupyter notebooks, or in interactive scripting environments like iPython, this will give you not only the most direct access to the API, but will allow you to customize your process however you please. The python methods even include the same progress bars and file-by-file processing of the command line tools!


## Getting Help

Hopefully the tutorials we provide give enough explanation for the common use-cases of VAMPIRES, but if you ran into issues or have suggestions please let us know in the [issues](https://github.com/scexao-org/vampires_dpp/issue/new). If you have more complicated questions or data issues contact the SCExAO team directly (c.c. [Barnaby Norris](mailto:barnaby.norris@sydney.edu.au) and [Miles Lucas](mailto:mdlucas@hawaii.edu)).