from .config import *

__all__ = ["VAMPIRES_SINGLECAM", "VAMPIRES_PDI", "VAMPIRES_HALPHA", "VAMPIRES_MAXIMAL"]

DEFAULT_DIRS = {
    ProductsOptions: "products",
    CalibrateOptions: "calibrated",
    FrameSelectOptions: "selected",
    RegisterOptions: "registered",
    CollapseOptions: "collapsed",
    PolarimetryOptions: "pdi",
}


VAMPIRES_SINGLECAM = PipelineOptions(
    name="",
    target="",
    frame_centers=CamCtrOption(cam1=[]),
    calibrate=CalibrateOptions(
        master_darks=CamFileInput(cam1=""),
        master_flats=CamFileInput(cam1=""),
        output_directory=DEFAULT_DIRS[CalibrateOptions],
    ),
    register=RegisterOptions(method="peak", output_directory=DEFAULT_DIRS[RegisterOptions]),
    collapse=CollapseOptions(method="median", output_directory=DEFAULT_DIRS[CollapseOptions]),
    products=ProductsOptions(output_directory=DEFAULT_DIRS[ProductsOptions]),
)


VAMPIRES_PDI = PipelineOptions(
    name="",
    target="",
    frame_centers=CamCtrOption(cam1=[], cam2=[]),
    calibrate=CalibrateOptions(
        master_darks=CamFileInput(cam1="", cam2=""),
        master_flats=CamFileInput(cam1="", cam2=""),
        output_directory=DEFAULT_DIRS[CalibrateOptions],
    ),
    register=RegisterOptions(method="peak", output_directory=DEFAULT_DIRS[RegisterOptions]),
    collapse=CollapseOptions(method="median", output_directory=DEFAULT_DIRS[CollapseOptions]),
    polarimetry=PolarimetryOptions(
        output_directory=DEFAULT_DIRS[PolarimetryOptions], ip=IPOptions()
    ),
    products=ProductsOptions(output_directory=DEFAULT_DIRS[ProductsOptions]),
)


VAMPIRES_HALPHA = PipelineOptions(
    name="",
    target="",
    frame_centers=dict(cam1=[], cam2=[]),
    calibrate=CalibrateOptions(
        master_darks=CamFileInput(cam1="", cam2=""),
        master_flats=CamFileInput(cam1="", cam2=""),
        output_directory=DEFAULT_DIRS[CalibrateOptions],
    ),
    register=RegisterOptions(method="peak", output_directory=DEFAULT_DIRS[RegisterOptions]),
    collapse=CollapseOptions(method="median", output_directory=DEFAULT_DIRS[CollapseOptions]),
    products=ProductsOptions(output_directory=DEFAULT_DIRS[ProductsOptions]),
)

VAMPIRES_MAXIMAL = PipelineOptions(
    name="",
    target="",
    frame_centers=dict(cam1=[], cam2=[]),
    calibrate=CalibrateOptions(
        master_darks=CamFileInput(cam1="", cam2=""),
        master_flats=CamFileInput(cam1="", cam2=""),
        output_directory=DEFAULT_DIRS[CalibrateOptions],
    ),
    frame_select=FrameSelectOptions(
        cutoff=0.3, metric="normvar", output_directory=DEFAULT_DIRS[FrameSelectOptions]
    ),
    register=RegisterOptions(method="peak", output_directory=DEFAULT_DIRS[RegisterOptions]),
    collapse=CollapseOptions(method="median", output_directory=DEFAULT_DIRS[CollapseOptions]),
    polarimetry=PolarimetryOptions(output_directory=DEFAULT_DIRS[PolarimetryOptions]),
    products=ProductsOptions(output_directory=DEFAULT_DIRS[ProductsOptions]),
)
