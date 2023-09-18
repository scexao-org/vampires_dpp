from .config import *

__all__ = ["VAMPIRES_SINGLECAM", "VAMPIRES_PDI", "VAMPIRES_SDI"]


VAMPIRES_BLANK = PipelineConfig(
    name="",
)

VAMPIRES_SINGLECAM = PipelineConfig(
    name="",
    register=RegisterConfig(),
    collapse=CollapseConfig(),
)

VAMPIRES_PDI = PipelineConfig(
    name="",
    register=RegisterConfig(),
    collapse=CollapseConfig(),
    polarimetry=PolarimetryConfig(),
)


VAMPIRES_SDI = PipelineConfig(
    name="",
    register=RegisterConfig(),
    collapse=CollapseConfig(),
    make_diff_images=True,
)
