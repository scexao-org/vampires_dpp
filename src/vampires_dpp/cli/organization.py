from multiprocessing import cpu_count
from pathlib import Path

import click
import tomli

import vampires_dpp as dpp
from vampires_dpp.organization import header_table, sort_files
from vampires_dpp.pipeline.deprecation import upgrade_config

__all__ = ("sort_raw", "table", "upgrade")

########## sort ##########


@click.command(
    name="sort",
    short_help="Sort raw data",
    help="Sorts raw data based on the data type. This will either use the `DATA-TYP` header value or the `U_OGFNAM` header, depending on when your data was taken.",
)
@click.argument(
    "filenames", nargs=-1, type=click.Path(dir_okay=False, readable=True, path_type=Path)
)
@click.option(
    "--outdir",
    "-o",
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    default=Path.cwd(),
    help="Output directory.",
)
@click.option(
    "--num-proc",
    "-j",
    default=1,
    type=click.IntRange(1, cpu_count()),
    help="Number of processes to use.",
    show_default=True,
)
@click.option("--ext", "-e", default=0, help="HDU extension")
@click.option(
    "--copy/--no-copy",
    "-c/-nc",
    default=True,
    prompt="Would you like to copy files?",
    help="copy files instead of moving them",
)
@click.option(
    "-d",
    "--decompress",
    is_flag=True,
    prompt=True,
    help="Would you like to decompress .fits.gz files while copying?",
)
@click.option("--quiet", "-q", is_flag=True, help="Silence progress bars and extraneous logging.")
def sort_raw(filenames, outdir, num_proc=1, ext=0, copy=False, quiet=False, decompress=False):
    sort_files(
        filenames,
        copy=copy,
        ext=ext,
        output_directory=outdir,
        num_proc=num_proc,
        quiet=quiet,
        decompress=decompress,
    )


########## table ##########


@click.command(
    name="table",
    short_help="Create CSV from headers",
    help="Go through each file and combine the header information into a single CSV.",
)
@click.argument(
    "filenames", nargs=-1, type=click.Path(dir_okay=False, readable=True, path_type=Path)
)
@click.option("-e", "--ext", help="FITS header extension to use for table", default=0)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=Path.cwd() / "header_table.csv",
    help="Output path without file extension.",
)
@click.option(
    "--num-proc",
    "-j",
    default=1,
    type=click.IntRange(1, cpu_count()),
    help="Number of processes to use.",
)
@click.option("-q", "--quiet", is_flag=True, help="Silence progress bars and extraneous logging.")
def table(filenames, ext, output, num_proc, quiet):
    # handle name clashes
    outpath = Path(output).resolve()
    outname = outpath.with_name(f"{outpath.name}.csv")

    if outname.exists():
        click.confirm(
            f"{outname.name} already exists in the output directory. Overwrite?", abort=True
        )
    df = header_table(filenames, num_proc=num_proc, quiet=quiet, ext=ext, fix=False)
    df.to_csv(outname)
    return outname


########## upgrade ##########


@click.command(
    name="upgrade",
    short_help="Upgrade configuration file",
    help=f"Tries to automatically upgrade a configuration file to the current version ({dpp.__version__}), prompting where necessary.",
)
@click.argument("config", type=click.Path(dir_okay=False, readable=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Output path.",
)
@click.option(
    "--edit", "-e", is_flag=True, help="Launch configuration file in editor after creation."
)
def upgrade(config, output, edit):
    if output is None:
        click.confirm("Are you sure you want to modify your configuration in-place?", abort=True)
        output = config
    with Path(config).open("rb") as fh:
        input_toml = tomli.load(fh)
    output_config = upgrade_config(input_toml)
    output_config.to_file(output)
    click.echo(f"File saved to {output.name}")
    if not edit:
        edit |= click.confirm("Would you like to edit this config file now?")

    if edit:
        click.edit(filename=output)
