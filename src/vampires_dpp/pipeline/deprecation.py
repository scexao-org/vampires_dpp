from packaging.version import Version

from vampires_dpp import __version__
from vampires_dpp.pipeline.config import *
from vampires_dpp.pipeline.pipeline import Pipeline


def upgrade_config(config: Pipeline) -> Pipeline:
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
    config_version = Version(config.version)
    output_config = Pipeline.from_str(config.to_toml())
    ## start with version 0.6
    if config_version < Version("0.6"):
        ## no changes yet, this breaking change was because of removing a script
        pass

    # update version
    output_config.version = __version__
    return output_config
