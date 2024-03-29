import click
from packaging.version import Version

from vampires_dpp import __version__
from vampires_dpp.pipeline.config import PipelineConfig
from vampires_dpp.pipeline.pipeline import Pipeline


def upgrade_config(config_dict: dict) -> Pipeline:
    config_version = Version(config_dict["version"])
    ## start with version 0.7, first breaking changes
    if config_version < Version("0.7"):
        config_dict = upgrade_to_0p7(config_dict)
    if config_version < Version("0.8"):
        config_dict = upgrade_to_0p8(config_dict)
    if config_version < Version("0.9"):
        config_dict = upgrade_to_0p8(config_dict)
    config_dict["version"] = __version__
    pipeline = PipelineConfig(**config_dict)
    return pipeline


upgrade_config.__doc__ = f"""
Tries to upgrade an old configuration to a new version, appropriately reflecting name changes and 
prompting the user for input when needed.

.. admonition::
    :class: tip

    In some cases (e.g. pre v0.6) it is necessary to recreate a configuration using `dpp new` 
    because this function is not able to convert those version ranges.

Parameters
----------
config : Pipeline
    Input Pipeline configuration

Returns
-------
Pipeline
    Pipeline configuration upgraded to the current package version ({__version__}).
"""


def upgrade_to_0p7(config_dict):
    if "polarimetry" in config_dict:
        # added method to polarimetry between "difference" and "mueller"
        # defaults to difference since that was all that was supported
        config_dict["polarimetry"]["method"] = click.prompt(
            "Which polarization calibration would you like to use?",
            type=click.Choice(["difference", "leastsq"], case_sensitive=False),
            default="difference",
        )
        # derotate_pa renamed to adi_sync and logic inverted
        adi_sync = not config_dict["polarimetry"].pop("derotate_pa", False)
        config_dict["polarimetry"]["adi_sync"] = adi_sync

    return config_dict


def upgrade_to_0p8(config_dict):
    # switch from "darks" to "backgrounds"
    if "calibrate" in config_dict:
        config_dict["calibrate"]["master_backgrounds"] = config_dict["calibrate"].pop(
            "master_darks"
        )
    if click.confirm("Would you like to make difference images?", default=False):
        config_dict["diff"] = dict(output_directory="diff")
    if click.confirm("Would you like to do PSF analysis?", default=True):
        config_dict["analysis"] = dict(output_directory="analysis", photometry=dict(aper_rad=10))

    return config_dict


def upgrade_to_0p9(config_dict):
    return config_dict
