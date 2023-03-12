from .config import *

__all__ = ["VAMPIRES_SINGLECAM", "VAMPIRES_PDI", "VAMPIRES_SDI"]

DEFAULT_DIRS = {
    ProductOptions: "products",
    CalibrateOptions: "calibrated",
    FrameSelectOptions: "selected",
    RegisterOptions: "registered",
    CollapseOptions: "collapsed",
    PolarimetryOptions: "pdi",
}


VAMPIRES_SINGLECAM = PipelineOptions(
    name="",
    frame_centers=CamCtrOption(cam1=[]),
    calibrate=CalibrateOptions(
        output_directory=DEFAULT_DIRS[CalibrateOptions],
    ),
    register=RegisterOptions(method="peak", output_directory=DEFAULT_DIRS[RegisterOptions]),
    collapse=CollapseOptions(method="median", output_directory=DEFAULT_DIRS[CollapseOptions]),
    products=ProductOptions(output_directory=DEFAULT_DIRS[ProductOptions]),
)

VAMPIRES_PDI = PipelineOptions(
    name="",
    frame_centers=CamCtrOption(cam1=[], cam2=[]),
    calibrate=CalibrateOptions(
        output_directory=DEFAULT_DIRS[CalibrateOptions],
    ),
    register=RegisterOptions(method="peak", output_directory=DEFAULT_DIRS[RegisterOptions]),
    collapse=CollapseOptions(method="median", output_directory=DEFAULT_DIRS[CollapseOptions]),
    polarimetry=PolarimetryOptions(
        output_directory=DEFAULT_DIRS[PolarimetryOptions], ip=IPOptions()
    ),
    products=ProductOptions(output_directory=DEFAULT_DIRS[ProductOptions]),
)


VAMPIRES_SDI = PipelineOptions(
    name="",
    frame_centers=dict(cam1=[], cam2=[]),
    calibrate=CalibrateOptions(
        output_directory=DEFAULT_DIRS[CalibrateOptions],
    ),
    register=RegisterOptions(method="peak", output_directory=DEFAULT_DIRS[RegisterOptions]),
    collapse=CollapseOptions(method="median", output_directory=DEFAULT_DIRS[CollapseOptions]),
    products=ProductOptions(output_directory=DEFAULT_DIRS[ProductOptions]),
)
