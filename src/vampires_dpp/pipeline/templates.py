from .config import *

__all__ = ["VAMPIRES_SINGLECAM", "VAMPIRES_PDI", "VAMPIRES_HALPHA", "VAMPIRES_MAXIMAL"]

DEFAULT_DIRS = {
    PipelineOptions: "products",
    CalibrateOptions: "calibrated",
    FrameSelectOptions: "selected",
    CoregisterOptions: "coregistered",
    CollapseOptions: "collapsed",
    PolarimetryOptions: "pdi",
}

DARK_CAM1 = "../darks/master_dark_cam1.fits"
DARK_CAM2 = "../darks/master_dark_cam2.fits"
FLAT_CAM1 = "../flats/master_flat_cam1.fits"
FLAT_CAM2 = "../flats/master_flat_cam2.fits"

VAMPIRES_SINGLECAM = PipelineOptions(
    name="",
    target="",
    filenames="raw/VMPA*.fits",
    output_directory=DEFAULT_DIRS[PipelineOptions],
    frame_centers=CamCtrOption(cam1=[]),
    master_dark=MasterDarkOptions(filenames="../darks/raw/VMPA*.fits", cam1=DARK_CAM1),
    master_flat=MasterFlatOptions(
        filenames="../flats/raw/VMPA*.fits", cam1=FLAT_CAM1, cam1_dark=DARK_CAM1
    ),
    calibrate=CalibrateOptions(
        master_darks=CamFileInput(cam1=DARK_CAM1),
        master_flats=CamFileInput(cam1=FLAT_CAM1),
        output_directory=DEFAULT_DIRS[CalibrateOptions],
    ),
    coregister=CoregisterOptions(method="peak", output_directory=DEFAULT_DIRS[CoregisterOptions]),
    collapse=CollapseOptions(method="median", output_directory=DEFAULT_DIRS[CollapseOptions]),
)


VAMPIRES_PDI = PipelineOptions(
    name="",
    target="",
    filenames="raw/VMPA*.fits",
    output_directory=DEFAULT_DIRS[PipelineOptions],
    frame_centers=CamCtrOption(cam1=[], cam2=[]),
    master_dark=MasterDarkOptions(
        filenames="../darks/raw/VMPA*.fits", cam1=DARK_CAM1, cam2=DARK_CAM2
    ),
    master_flat=MasterFlatOptions(
        filenames="../flats/raw/VMPA*.fits",
        cam1=FLAT_CAM1,
        cam2=FLAT_CAM2,
        cam1_dark=DARK_CAM1,
        cam2_dark=DARK_CAM2,
    ),
    calibrate=CalibrateOptions(
        master_darks=CamFileInput(cam1=DARK_CAM1, cam2=DARK_CAM2),
        master_flats=CamFileInput(cam1=FLAT_CAM1, cam2=FLAT_CAM2),
        output_directory=DEFAULT_DIRS[CalibrateOptions],
    ),
    coregister=CoregisterOptions(method="peak", output_directory=DEFAULT_DIRS[CoregisterOptions]),
    collapse=CollapseOptions(method="median", output_directory=DEFAULT_DIRS[CollapseOptions]),
    polarimetry=PolarimetryOptions(
        output_directory=DEFAULT_DIRS[PolarimetryOptions], ip=IPOptions()
    ),
)


VAMPIRES_HALPHA = PipelineOptions(
    name="",
    target="",
    filenames="raw/VMPA*.fits",
    output_directory=DEFAULT_DIRS[PipelineOptions],
    frame_centers=dict(cam1=[], cam2=[]),
    master_dark=MasterDarkOptions(
        filenames="../darks/raw/VMPA*.fits", cam1=DARK_CAM1, cam2=DARK_CAM2
    ),
    master_flat=MasterFlatOptions(
        filenames="../flats/raw/VMPA*.fits",
        cam1=FLAT_CAM1,
        cam2=FLAT_CAM2,
        cam1_dark=DARK_CAM1,
        cam2_dark=DARK_CAM2,
    ),
    calibrate=CalibrateOptions(
        master_darks=CamFileInput(cam1=DARK_CAM1, cam2=DARK_CAM2),
        master_flats=CamFileInput(cam1=FLAT_CAM1, cam2=FLAT_CAM2),
        output_directory=DEFAULT_DIRS[CalibrateOptions],
    ),
    coregister=CoregisterOptions(method="peak", output_directory=DEFAULT_DIRS[CoregisterOptions]),
    collapse=CollapseOptions(method="median", output_directory=DEFAULT_DIRS[CollapseOptions]),
)

VAMPIRES_MAXIMAL = PipelineOptions(
    name="",
    target="",
    filenames="raw/VMPA*.fits",
    output_directory=DEFAULT_DIRS[PipelineOptions],
    frame_centers=dict(cam1=[], cam2=[]),
    master_dark=MasterDarkOptions(
        filenames="../darks/raw/VMPA*.fits", cam1=DARK_CAM1, cam2=DARK_CAM2
    ),
    master_flat=MasterFlatOptions(
        filenames="../flats/raw/VMPA*.fits",
        cam1=FLAT_CAM1,
        cam2=FLAT_CAM2,
        cam1_dark=DARK_CAM1,
        cam2_dark=DARK_CAM2,
    ),
    calibrate=CalibrateOptions(
        master_darks=CamFileInput(cam1=DARK_CAM1, cam2=DARK_CAM2),
        master_flats=CamFileInput(cam1=FLAT_CAM1, cam2=FLAT_CAM2),
        output_directory=DEFAULT_DIRS[CalibrateOptions],
    ),
    frame_select=FrameSelectOptions(
        cutoff=0.3, metric="normvar", output_directory=DEFAULT_DIRS[FrameSelectOptions]
    ),
    coregister=CoregisterOptions(method="peak", output_directory=DEFAULT_DIRS[CoregisterOptions]),
    collapse=CollapseOptions(method="median", output_directory=DEFAULT_DIRS[CollapseOptions]),
    polarimetry=PolarimetryOptions(output_directory=DEFAULT_DIRS[PolarimetryOptions]),
)
