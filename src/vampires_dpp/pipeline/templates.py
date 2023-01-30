from .config import *

__all__ = ["VAMPIRES_SINGLECAM", "VAMPIRES_PDI", "VAMPIRES_HALPHA"]

DEFAULT_DIRS = {
    CalibrateOptions: "calibrated",
    FrameSelectOptions: "selected",
    CoregisterOptions: "coregistered",
    CollapseOptions: "collapsed",
    PolarimetryOptions: "pdi",
}


VAMPIRES_SINGLECAM = PipelineOptions(
    name="",
    target="",
    filenames="raw/VMPA*.fits",
    frame_centers=CamCtrOption(cam1=[]),
    calibrate=CalibrateOptions(
        master_darks=CamFileInput(cam1="master_dark_cam1.fits"),
        master_flats=CamFileInput(cam1="master_flat_cam1.fits"),
        output_directory=DEFAULT_DIRS[CalibrateOptions],
    ),
    coregister=CoregisterOptions(method="peak", output_directory=DEFAULT_DIRS[CoregisterOptions]),
    collapse=CollapseOptions(method="median", output_directory=DEFAULT_DIRS[CollapseOptions]),
)


VAMPIRES_PDI = PipelineOptions(
    name="",
    target="",
    filenames="raw/VMPA*.fits",
    frame_centers=CamCtrOption(cam1=[], cam2=[]),
    calibrate=CalibrateOptions(
        master_darks=CamFileInput(cam1="master_dark_cam1.fits", cam2="master_dark_cam2.fits"),
        master_flats=CamFileInput(cam1="master_flat_cam1.fits", cam2="master_flat_cam2.fits"),
        output_directory=DEFAULT_DIRS[CalibrateOptions],
    ),
    coregister=CoregisterOptions(method="peak", output_directory=DEFAULT_DIRS[CoregisterOptions]),
    collapse=CollapseOptions(method="median", output_directory=DEFAULT_DIRS[CollapseOptions]),
    polarimetry=PolarimetryOptions(output_directory=DEFAULT_DIRS[PolarimetryOptions]),
)


VAMPIRES_HALPHA = PipelineOptions(
    name="",
    target="",
    filenames="raw/VMPA*.fits",
    frame_centers=dict(cam1=[], cam2=[]),
    calibrate=CalibrateOptions(
        master_darks=CamFileInput(cam1="master_dark_cam1.fits", cam2="master_dark_cam2.fits"),
        master_flats=CamFileInput(cam1="master_flat_cam1.fits", cam2="master_flat_cam2.fits"),
        output_directory=DEFAULT_DIRS[CalibrateOptions],
    ),
    coregister=CoregisterOptions(method="peak", output_directory=DEFAULT_DIRS[CoregisterOptions]),
    collapse=CollapseOptions(method="median", output_directory=DEFAULT_DIRS[CollapseOptions]),
)
