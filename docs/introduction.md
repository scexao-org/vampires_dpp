# Introduction

## VAMPIRES Observation Modes

VAMPIRES is an incredibly diverse instrument in its usage with SCExAO. It can be used for studying evolved stellar physics, protoplanetary disks, polarimetry, H-ɑ emission, in addition to an interferometric mode via sparse aperture masking (SAM). VAMPIRES is also used for focal-plane wavefront sensing, for example using differential focus for [phase diversity](), as well as complementary data for infrared observations (i.e., telemetry). `vampires_dpp` provides tooling primarily for imaging modes, although the image-processing tools provide building blocks for a variety of applications of VAMPIRES data (and PRs are welcome for interferometric modes!).

1. Speckle imaging / telemetry
2. Polarimetric differential imaging (PDI)
3. Narrowband spectral differential imaging (SDI)

These modes also support the use of the SCExAO's [visible coronagraph](https://www.naoj.org/Projects/SCEXAO/scexaoWEB/030openuse.web/040vampires.web/100vampcoronagraph.web/indexm.html).

## VAMPIRES Data Formats

Depending on who you are, you may have to deal with a variety of VAMPIRES data formats. For open-use observers, this is the STARS gen-2 archive format, which is different from the format saved by VAMPIRES during acquisition. The `vampires_dpp` prioritizes the gen-2 format highest, but the API should be flexible enough to accommadate any user of VAMPIRES- observer or engineer.

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
    - Files are given frame id names, e.g. "VMPA00023445.fits"
    - Each file corresponds to a single camera, FLC state, and HWP angle
- VAMPIRES format
    - Files have names as set by controller, e.g. "HD141569A_00_750-50_LyotStop_00_cam1.fits"
    - Each file is a raw cube from the camera, which includes bad readout frame in the first frame
    - FLC states interleaved in every other frame for PDI mode

## Processing workflows

`vampires_dpp` is built with multiple processing workflows in mind to support the large diversity of observing modes and usages of VAMPIRES. Typical VAMPIRES observations produce many gigabytes of data across hundreds of files, which means processing data must occur on a file-by-file basis, with the potential for multi-processing. A large portion of the `vampires_dpp` API is available in a command line interface (CLI) for quick processing of files and various other helpers.

If you're an advanced python user, any script you find will be composed of functions in the `vampires_dpp` python module, which gets installed when you `pip install` this code. If you are someone who likes working in Jupyter notebooks, or in interactive scripting environments like iPython, this will give you not only the most direct access to the API, but will allow you to customize your process to some extent. The python methods even include the same progress bars and file-by-file processing of the command line tools!


## Getting Help

Hopefully the tutorials we provide give enough explanation for the common use-cases of VAMPIRES, but if you ran into issues or have suggestions please let us know in the [issues](https://github.com/scexao-org/vampires_dpp/issue/new). If you have more complicated questions or data issues contact the SCExAO team directly (c.c. [Miles Lucas](mailto:mdlucas@hawaii.edu)). If you are in the SCExAO slack, feel free to ask a question in the `#vampires-dpp` channel.
