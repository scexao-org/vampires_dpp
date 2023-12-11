import collections

import click

import vampires_dpp as dpp

from .astro import astro
from .centroids import centroid
from .new import new_config
from .organization import sort_raw, table, upgrade
from .prep import norm, prep
from .run import pdi, run


# https://stackoverflow.com/a/58323807
class OrderedGroup(click.Group):
    def __init__(self, name=None, commands=None, **attrs):
        super().__init__(name, commands, **attrs)
        #: the registered subcommands by their exported names.
        self.commands = commands or collections.OrderedDict()

    def list_commands(self, ctx):
        return self.commands


@click.group(name="dpp", cls=OrderedGroup, no_args_is_help=True)
@click.version_option(dpp.__version__, "--version", "-v", prog_name="vampires_dpp")
def main():
    pass


main.add_command(sort_raw)
main.add_command(norm)
main.add_command(prep)
main.add_command(new_config)
main.add_command(centroid)
main.add_command(astro)
main.add_command(run)
main.add_command(pdi)
main.add_command(table)
main.add_command(upgrade)

if __name__ == "__main__":
    main()
