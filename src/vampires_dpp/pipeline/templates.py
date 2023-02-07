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

DEFAULT_FILENAME = "./**/VMPA*.fits"

VAMPIRES_SINGLECAM = PipelineOptions(
    name="",
    target="",
    filenames=DEFAULT_FILENAME,
    output_directory=DEFAULT_DIRS[PipelineOptions],
    frame_centers=CamCtrOption(cam1=[]),
    calibrate=CalibrateOptions(
        master_darks=CamFileInput(cam1=""),
        master_flats=CamFileInput(cam1=""),
        output_directory=DEFAULT_DIRS[CalibrateOptions],
    ),
    coregister=CoregisterOptions(method="peak", output_directory=DEFAULT_DIRS[CoregisterOptions]),
    collapse=CollapseOptions(method="median", output_directory=DEFAULT_DIRS[CollapseOptions]),
)


VAMPIRES_PDI = PipelineOptions(
    name="",
    target="",
    filenames=DEFAULT_FILENAME,
    output_directory=DEFAULT_DIRS[PipelineOptions],
    frame_centers=CamCtrOption(cam1=[], cam2=[]),
    calibrate=CalibrateOptions(
        master_darks=CamFileInput(cam1="", cam2=""),
        master_flats=CamFileInput(cam1="", cam2=""),
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
    filenames=DEFAULT_FILENAME,
    output_directory=DEFAULT_DIRS[PipelineOptions],
    frame_centers=dict(cam1=[], cam2=[]),
    calibrate=CalibrateOptions(
        master_darks=CamFileInput(cam1="", cam2=""),
        master_flats=CamFileInput(cam1="", cam2=""),
        output_directory=DEFAULT_DIRS[CalibrateOptions],
    ),
    coregister=CoregisterOptions(method="peak", output_directory=DEFAULT_DIRS[CoregisterOptions]),
    collapse=CollapseOptions(method="median", output_directory=DEFAULT_DIRS[CollapseOptions]),
)

VAMPIRES_MAXIMAL = PipelineOptions(
    name="",
    target="",
    filenames=DEFAULT_FILENAME,
    output_directory=DEFAULT_DIRS[PipelineOptions],
    frame_centers=dict(cam1=[], cam2=[]),
    calibrate=CalibrateOptions(
        master_darks=CamFileInput(cam1="", cam2=""),
        master_flats=CamFileInput(cam1="", cam2=""),
        output_directory=DEFAULT_DIRS[CalibrateOptions],
    ),
    frame_select=FrameSelectOptions(
        cutoff=0.3, metric="normvar", output_directory=DEFAULT_DIRS[FrameSelectOptions]
    ),
    coregister=CoregisterOptions(method="peak", output_directory=DEFAULT_DIRS[CoregisterOptions]),
    collapse=CollapseOptions(method="median", output_directory=DEFAULT_DIRS[CollapseOptions]),
    polarimetry=PolarimetryOptions(output_directory=DEFAULT_DIRS[PolarimetryOptions]),
)
