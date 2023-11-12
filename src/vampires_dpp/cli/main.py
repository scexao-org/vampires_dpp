import click

import vampires_dpp as dpp

from .centroids import centroid
from .new import new_config
from .organization import sort, table, upgrade
from .prep import norm, prep
from .run import pdi, run


@click.group(name="main", no_args_is_help=True)
@click.version_option(dpp.__version__, "--version", "-v", prog_name="vampires_dpp")
def main():
    pass


main.add_command(sort)
main.add_command(norm)
main.add_command(prep)
main.add_command(new_config)
main.add_command(centroid)
main.add_command(run)
main.add_command(pdi)
main.add_command(table)
main.add_command(upgrade)

if __name__ == "__main__":
    main()
