from .config import CollapseConfig, PipelineConfig, PolarimetryConfig

__all__ = ["VAMPIRES_SINGLECAM", "VAMPIRES_PDI", "VAMPIRES_SDI"]


VAMPIRES_BLANK = PipelineConfig(name="")

VAMPIRES_SINGLECAM = PipelineConfig(name="", collapse=CollapseConfig())

VAMPIRES_PDI = PipelineConfig(name="", collapse=CollapseConfig(), polarimetry=PolarimetryConfig())

VAMPIRES_SDI = PipelineConfig(name="", collapse=CollapseConfig(), make_diff_images=True)
