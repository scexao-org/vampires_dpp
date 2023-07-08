from packaging.version import Version

from vampires_dpp import __version__
from vampires_dpp.pipeline.config import *
from vampires_dpp.pipeline.pipeline import Pipeline


def upgrade_config(config_dict: dict) -> Pipeline:
    f"""
    Tries to upgrade an old configuration to a new version, appropriately reflecting name changes and prompting the user for input when needed.

    .. admonition::
        :class: tip

        In some cases (e.g. pre v0.6) it is necessary to recreate a configuration using `dpp create` because this function is not able to convert those version ranges.

    Parameters
    ----------
    config : Pipeline
        Input Pipeline configuration

    Returns
    -------
    Pipeline
        Pipeline configuration upgraded to the current package version ({__version__}).
    """
    Version(config_dict["version"])
    config_dict["version"] = __version__
    pipeline = Pipeline(**config_dict)
    return pipeline
