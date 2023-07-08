from .config import *

__all__ = ["VAMPIRES_SINGLECAM", "VAMPIRES_PDI", "VAMPIRES_SDI"]

DEFAULT_DIRS = {
    ProductOptions: "products",
    CalibrateOptions: "calibrated",
    FrameSelectOptions: "selected",
    RegisterOptions: "registered",
    CollapseOptions: "collapsed",
    PolarimetryOptions: "pdi",
    DiffOptions: "diff",
    AnalysisOptions: "analysis",
}


VAMPIRES_BLANK = PipelineOptions(
    name="",
    calibrate=CalibrateOptions(
        output_directory=DEFAULT_DIRS[CalibrateOptions],
    ),
    products=ProductOptions(output_directory=DEFAULT_DIRS[ProductOptions]),
)

VAMPIRES_SINGLECAM = PipelineOptions(
    name="",
    calibrate=CalibrateOptions(
        output_directory=DEFAULT_DIRS[CalibrateOptions],
    ),
    register=RegisterOptions(method="peak", output_directory=DEFAULT_DIRS[RegisterOptions]),
    collapse=CollapseOptions(method="median", output_directory=DEFAULT_DIRS[CollapseOptions]),
    analysis=AnalysisOptions(
        model="gaussian", aper_rad=10, output_directory=DEFAULT_DIRS[AnalysisOptions]
    ),
    products=ProductOptions(output_directory=DEFAULT_DIRS[ProductOptions]),
)

VAMPIRES_PDI = PipelineOptions(
    name="",
    calibrate=CalibrateOptions(
        output_directory=DEFAULT_DIRS[CalibrateOptions],
    ),
    register=RegisterOptions(method="peak", output_directory=DEFAULT_DIRS[RegisterOptions]),
    collapse=CollapseOptions(method="median", output_directory=DEFAULT_DIRS[CollapseOptions]),
    analysis=AnalysisOptions(
        model="gaussian", aper_rad=10, output_directory=DEFAULT_DIRS[AnalysisOptions]
    ),
    polarimetry=PolarimetryOptions(
        output_directory=DEFAULT_DIRS[PolarimetryOptions], ip=IPOptions()
    ),
    products=ProductOptions(output_directory=DEFAULT_DIRS[ProductOptions]),
)


VAMPIRES_SDI = PipelineOptions(
    name="",
    calibrate=CalibrateOptions(
        output_directory=DEFAULT_DIRS[CalibrateOptions],
    ),
    register=RegisterOptions(method="peak", output_directory=DEFAULT_DIRS[RegisterOptions]),
    collapse=CollapseOptions(method="median", output_directory=DEFAULT_DIRS[CollapseOptions]),
    analysis=AnalysisOptions(
        model="gaussian", aper_rad=10, output_directory=DEFAULT_DIRS[AnalysisOptions]
    ),
    diff=DiffOptions(output_directory=DEFAULT_DIRS[DiffOptions]),
    products=ProductOptions(output_directory=DEFAULT_DIRS[ProductOptions]),
)
