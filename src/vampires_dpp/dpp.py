import glob
import multiprocessing as mp
import os
import re
import readline
import sys
from multiprocessing import cpu_count
from pathlib import Path

import astropy.units as u
import click
import tomli
import tqdm.auto as tqdm
from astropy.io import fits
from loguru import logger

import vampires_dpp as dpp
from vampires_dpp.calibration import (
    normalize_file,
    process_background_files,
    process_flat_files,
)
from vampires_dpp.constants import DEFAULT_NPROC
from vampires_dpp.organization import header_table, sort_files
from vampires_dpp.pipeline.config import (
    CamFileInput,
    CollapseConfig,
    ObjectConfig,
    PipelineConfig,
    PolarimetryConfig,
    SpecphotConfig,
)
from vampires_dpp.pipeline.deprecation import upgrade_config
from vampires_dpp.pipeline.pipeline import Pipeline
from vampires_dpp.pipeline.templates import (
    VAMPIRES_BLANK,
    VAMPIRES_PDI,
    VAMPIRES_SDI,
    VAMPIRES_SINGLECAM,
)
from vampires_dpp.specphot.filters import FILTERS
from vampires_dpp.specphot.query import get_simbad_table, get_ucac_flux, get_ucac_table
from vampires_dpp.util import check_version, load_fits_key
from vampires_dpp.wcs import get_gaia_astrometry

logger.remove(0)
logger.add(sys.stderr, level="INFO")


########## main ##########


@click.group(name="main", no_args_is_help=True)
@click.version_option(dpp.__version__, "--version", "-v", prog_name="vampires_dpp")
def main():
    pass


########## prep ##########


@main.group(
    name="prep",
    short_help="Prepare calibration files",
    help="Create calibration files from background files (darks or sky frames) and flats.",
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
    default=DEFAULT_NPROC,
    type=click.IntRange(1, cpu_count()),
    help="Number of processes to use.",
    show_default=True,
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Silence progress bars and extraneous logging.",
)
@click.pass_context
def prep(ctx, outdir, quiet, num_proc):
    # prepare context
    ctx.ensure_object(dict)
    ctx.obj["outdir"] = outdir
    ctx.obj["quiet"] = quiet
    ctx.obj["num_proc"] = num_proc


@prep.command(
    name="back",
    short_help="background files (darks/skies)",
    help="Create background files from darks/skies. Each input file will be collapsed. Groups of files with the same exposure time, EM gain, and frame size will be median-combined together to create a super-background file.",
)
@click.argument(
    "filenames",
    nargs=-1,
    type=click.Path(dir_okay=False, readable=True, path_type=Path),
)
@click.option(
    "--collapse",
    "-c",
    type=click.Choice(["median", "mean", "varmean", "biweight"], case_sensitive=False),
    default="median",
    help="Frame collapse method",
    show_default=True,
)
@click.option("--force", "-f", is_flag=True, help="Force processing of files")
@click.pass_context
def back(ctx, filenames, collapse, force):
    process_background_files(
        filenames,
        collapse=collapse,
        force=force,
        output_directory=ctx.obj["outdir"] / "background",
        quiet=ctx.obj["quiet"],
        num_proc=ctx.obj["num_proc"],
    )


@prep.command(
    name="flat",
    short_help="flat-field files",
    help="Create flat-field files. Each input file will be collapsed with background-subtraction if files are provided. Groups of files with the same exposure time, EM gain, frame size, and filter will be median-combined together to create a super-flat file.",
)
@click.argument(
    "filenames",
    nargs=-1,
    type=click.Path(dir_okay=False, readable=True, path_type=Path),
)
@click.option(
    "--back",
    "-b",
    type=click.Path(exists=True, path_type=Path),
    help="Background file to subtract from each flat-field. If a directory, will match the background files in that directory to the exposure times, EM gains, and frame sizes. Note: will search the output directory for background files if none are given.",
)
@click.option(
    "--collapse",
    "-c",
    type=click.Choice(["median", "mean", "varmean", "biweight"], case_sensitive=False),
    default="median",
    help="Frame collapse method",
    show_default=True,
)
@click.option("--force", "-f", is_flag=True, help="Force processing of files")
@click.pass_context
def flat(ctx, filenames, back, collapse, force):
    # if directory, filter non-FITS files and sort for background files
    if back is None:
        back = ctx.obj["outdir"]
    calib_files = list(back.glob("**/[!._]*.fits")) + list(back.glob("**/[!._]*.fits.fz"))
    process_flat_files(
        filenames,
        collapse=collapse,
        force=force,
        output_directory=ctx.obj["outdir"] / "flat",
        quiet=ctx.obj["quiet"],
        num_proc=ctx.obj["num_proc"],
        background_files=calib_files,
    )


########## centroid ##########


@main.command(name="centroid", help="Fit the centroids")
@click.argument(
    "config",
    type=click.Path(dir_okay=False, readable=True, path_type=Path),
)
@click.argument(
    "filenames",
    nargs=-1,
    type=click.Path(dir_okay=False, readable=True, path_type=Path),
)
@click.option("-o", "--outdir", default=Path.cwd(), type=Path, help="Output file directory")
@click.option(
    "--num-proc",
    "-j",
    default=DEFAULT_NPROC,
    type=click.IntRange(1, cpu_count()),
    help="Number of processes to use.",
    show_default=True,
)
def centroid(config: Path, filenames, num_proc, outdir):
    # make sure versions match within SemVar
    pipeline = Pipeline(PipelineConfig.from_file(config), workdir=outdir)
    if not check_version(pipeline.config.dpp_version, dpp.__version__):
        raise ValueError(
            f"Input pipeline version ({pipeline.config.dpp_version}) is not compatible with installed version of `vampires_dpp` ({dpp.__version__}). Try running `dpp upgrade {config}`."
        )
    table = header_table(filenames, num_proc=num_proc)
    pipeline.save_centroids(table)


########## new ##########


def pathCompleter(text, state):
    """This is the tab completer for systems paths.
    Only tested on *nix systems
    """
    # replace ~ with the user's home dir. See https://docs.python.org/2/library/os.path.html
    if "~" in text:
        text = os.path.expanduser(text)

    # autocomplete directories with having a trailing slash
    if os.path.isdir(text):
        text += "/"

    return [x for x in glob.glob(text + "*")][state]


def createListCompleter(items):
    """This is a closure that creates a method that autocompletes from
    the given list.

    Since the autocomplete function can't be given a list to complete from
    a closure is used to create the listCompleter function with a list to complete
    from.
    """
    list_strings = map(str, items)

    def listCompleter(text, state):
        if not text:
            return list(list_strings)[state]
        else:
            matches = filter(lambda s: s.startswith(text), list_strings)
            return list(matches)[state]

    return listCompleter


@main.command(name="new", help="Generate configuration files")
@click.argument(
    "config",
    type=click.Path(dir_okay=False, readable=True, path_type=Path),
)
@click.option(
    "--edit", "-e", is_flag=True, help="Launch configuration file in editor after creation."
)
@click.pass_context
def new_config(ctx, config, edit):
    readline.set_completer_delims(" \t\n;")
    readline.parse_and_bind("tab: complete")

    ## check if output file exists
    if config.is_file():
        overwrite = click.confirm(
            f"{config.name} already exists in output directory, would you like to overwrite it?",
            default=False,
        )
        if not overwrite:
            ctx.exit()

    ## get template
    template_choices = ["none", "singlecam", "pdi", "sdi"]

    readline.set_completer(createListCompleter(template_choices))
    template = click.prompt(
        "Choose a starting template",
        type=click.Choice(template_choices, case_sensitive=False),
        default="none",
    )
    match template:
        case "singlecam":
            tpl = VAMPIRES_SINGLECAM
        case "pdi":
            tpl = VAMPIRES_PDI
        case "sdi":
            tpl = VAMPIRES_SDI
        case _:
            tpl = VAMPIRES_BLANK
    readline.set_completer()

    ## get name
    name_guess = config.stem
    name = click.prompt(f"Path-friendly name for this reduction", default=name_guess)
    tpl.name = name_guess if name == "" else name.replace(" ", "_").replace("/", "")

    ## get target
    obj = click.prompt("SIMBAD-friendly object name (optional)", default="")
    coord = None
    if obj != "":
        rad = 1
        cat = "dr3"
        while True:
            coord = get_gaia_astrometry(obj, catalog=cat, radius=rad)
            if coord is not None:
                break

            click.echo(f'  Could not find {obj} in GAIA {cat.upper()} with {rad}" radius.')
            _input = click.prompt(
                "Query different catalog (dr1/dr2/dr3), enter search radius in arcsec, or enter new object name (optional)"
            )
            match _input:
                case "":
                    # give up
                    break
                case "dr1" | "dr2" | "dr3":
                    # try different catalog
                    cat = _input
                case _:
                    try:
                        # if a number was entered, increase search radius
                        rad = float(_input)
                    except ValueError:
                        # otherwise try a new object
                        obj = _input

        if coord is not None:
            tpl.object = ObjectConfig(
                object=obj,
                ra=coord.ra.to_string("hour", sep=":", pad=True),
                dec=coord.dec.to_string("deg", sep=":", pad=True),
                parallax=coord.distance.to(u.mas, equivalencies=u.parallax()).value,
                pm_ra=coord.pm_ra_cosdec.to(u.mas / u.year).value,
                pm_dec=coord.pm_dec.to(u.mas / u.year).value,
                frame=coord.frame.name,
                obstime=str(coord.obstime),
            )
    if coord is None:
        click.echo(" - No coordinate information set; will only use header values.")

    readline.set_completer(pathCompleter)
    calib_dir = click.prompt("Enter path to calibration files", default="")
    readline.set_completer()
    if calib_dir != "":
        tpl.calibrate.calib_directory = Path(calib_dir)
        tpl.calibrate.back_subtract = click.confirm(
            " - Subtract backgrounds?", default=tpl.calibrate.back_subtract
        )
        tpl.calibrate.flat_correct = click.confirm(
            " - Flat correct?", default=tpl.calibrate.flat_correct
        )
    else:
        tpl.calibrate.calib_directory = None

    tpl.calibrate.save_intermediate = click.confirm(
        "Would you like to save calibrated files?", default=tpl.calibrate.save_intermediate
    )

    ## Coronagraph
    tpl.coronagraphic = click.confirm("Did you use a coronagraph?", default=False)

    ## Analysis
    tpl.analysis.window_size = click.prompt(
        "Enter analysis window size (px)", type=int, default=tpl.analysis.window_size
    )
    # tpl.analysis.subtract_radprof = click.confirm(
    #     "Would you like to subtract a radial profile for analysis?", default=tpl.coronagraphic
    # )
    # tpl.analysis.strehl = click.confirm(
    #     "Would you like to estimate the Strehl ratio?", default=False
    # )

    aper_rad = click.prompt('Enter aperture radius (px/"auto")', default=tpl.analysis.aper_rad)
    try:
        aper_rad = float(aper_rad)
    except ValueError:
        pass
    if not isinstance(aper_rad, str) and aper_rad > tpl.analysis.window_size / 2:
        aper_rad = tpl.analysis.window_size / 2
        click.echo(f" ! Reducing aperture radius to match window size ({aper_rad:.0f} px)")
    tpl.analysis.aper_rad = aper_rad

    ann_rad = None
    if tpl.analysis.aper_rad != "auto" and click.confirm(
        "Would you like to subtract background annulus?", default=False
    ):
        resp = click.prompt(
            " - Enter comma-separated inner and outer radius (px)",
            default=f"{max(aper_rad, tpl.analysis.window_size / 2 - 5)}, {tpl.analysis.window_size / 2}",
        )
        ann_rad = list(map(float, resp.replace(" ", "").split(",")))
        if ann_rad[1] > tpl.analysis.window_size / 2:
            ann_rad[1] = tpl.analysis.window_size / 2
            click.echo(
                f" ! Reducing annulus outer radius to match window size ({ann_rad[1]:.0f} px)"
            )
        if ann_rad[0] >= ann_rad[1]:
            ann_rad[0] = max(aper_rad, ann_rad[0] - 5)
            click.echo(f" ! Reducing annulus inner radius to ({ann_rad[0]:.0f} px)")
        tpl.analysis.ann_rad = ann_rad

    ## Specphot Cal
    if click.confirm("Would you like to do flux calibration?", default=tpl.specphot is not None):
        readline.set_completer(pathCompleter)
        source = click.prompt(
            ' - Enter source type ("pickles"/"zeropoint"/path to spectrum)', default="pickles"
        )
        readline.set_completer()
        if source == "pickles":
            if tpl.object is not None:
                simbad_table = get_simbad_table(tpl.object.object)
                ucac_table = get_ucac_table(tpl.object.object)
                mag, mag_band = get_ucac_flux(ucac_table)

                sptype = re.match(r"\w\d[IV]{1,3}", simbad_table["SP_TYPE"][0]).group()
                click.echo(f" * Found UCAC4 info: {sptype} {mag_band}={mag:.02f}")
            else:
                mag = ""
                mag_band = "V"
                sptype = "G0V"
            sptype = click.prompt(" - Enter spectral type", default=sptype)
            if sptype not in PICKLES_MAP.keys():
                click.echo(
                    " ! No match in pickles stellar library - you will have to edit manually"
                )
            mag = click.prompt(" - Enter source magnitude", default=mag, type=float)
            mag_band = click.prompt(
                " - Enter source magnitude passband",
                default=mag_band,
                type=click.Choice(list(FILTERS.keys()), case_sensitive=False),
            )
        else:
            sptype = mag = mag_band = None

        metric = click.prompt(
            " - Select which metric to use for flux",
            default="photometry",
            type=click.Choice(["photometry", "sum"]),
        )
        tpl.specphot = SpecphotConfig(
            source=source, sptype=sptype, mag=mag, mag_band=mag_band, flux_metric=metric
        )
    else:
        tpl.specphot = None

    ## Collapsing
    if click.confirm("Would you like to collapse your data?", default=tpl.collapse is not None):
        collapse_choices = ["median", "mean", "varmean", "biweight"]
        readline.set_completer(createListCompleter(collapse_choices))
        collapse_method = click.prompt(
            " - Choose a collapse method",
            type=click.Choice(collapse_choices, case_sensitive=False),
            default="median",
        )
        readline.set_completer()
        tpl.collapse = CollapseConfig(method=collapse_method)

        ## Frame selection
        if click.confirm(
            "Would you like to do frame selection?", default=tpl.collapse.frame_select is not None
        ):
            tpl.collapse.select_cutoff = click.prompt(
                " - Enter a cutoff quantile (0 to 1, larger means more discarding)", type=float
            )

            metric_choices = ["normvar", "peak", "strehl"]
            readline.set_completer(createListCompleter(metric_choices))
            tpl.collapse.frame_select = click.prompt(
                " - Choose a frame selection metric",
                type=click.Choice(metric_choices, case_sensitive=False),
                default="normvar",
            )
            readline.set_completer()
        else:
            tpl.collapse.frame_select = None

        ## Registration
        do_register = click.confirm(
            f"Would you like to do frame registration?", default=tpl.collapse.centroid is not None
        )
        if do_register:
            centroid_choices = ["com", "peak", "gauss", "quad"]
            readline.set_completer(createListCompleter(centroid_choices))
            tpl.collapse.centroid = click.prompt(
                " - Choose a registration method",
                type=click.Choice(centroid_choices, case_sensitive=False),
                default="com",
            )
            readline.set_completer()
            # if register_method == "dft":
            #     dft_factor = click.prompt(" -   Enter DFT upsample factor", default=1, type=int)
        else:
            tpl.collapse.centroid = None

        tpl.collapse.recenter = click.confirm(
            " - Would you like to recenter the collapsed data after a model PSF fit?", default=True
        )

    tpl.make_diff_images = click.confirm(
        "Would you like to make difference images?", default=tpl.make_diff_images
    )

    ## Polarization
    if click.confirm("Would you like to do polarimetry?", default=tpl.polarimetry is not None):
        calib_choices = ["doublediff", "triplediff", "leastsq"]
        readline.set_completer(createListCompleter(calib_choices))
        pol_method = click.prompt(
            " - Choose a polarimetric calibration method",
            type=click.Choice(calib_choices, case_sensitive=False),
            default="triplediff",
        )
        readline.set_completer()
        tpl.polarimetry = PolarimetryConfig(method=pol_method)
        if "diff" in tpl.polarimetry.method:
            tpl.polarimetry.mm_correct = click.confirm(
                " - Would you like to to Mueller-matrix correction?",
                default=tpl.polarimetry.mm_correct,
            )
        tpl.polarimetry.use_ideal_mm = click.confirm(
            " - Would you like to use an idealized Muller matrix?",
            default=tpl.polarimetry.use_ideal_mm,
        )
        tpl.polarimetry.ip_correct = click.confirm(
            " - Would you like to do IP touchup?", default=tpl.polarimetry.ip_correct is not None
        )
        if tpl.polarimetry.ip_correct:
            default = "annulus" if tpl.coronagraphic else "aperture"
            tpl.polarimetry.ip_method = click.prompt(
                " - Select IP correction method",
                type=click.Choice(["aperture", "annulus"], case_sensitive=False),
                default=default,
            )
            if tpl.polarimetry.ip_method == "aperture":
                tpl.polarimetry.ip_radius = click.prompt(
                    " - Enter IP aperture radius (px)", type=float, default=8
                )
            elif tpl.polarimetry.ip_method == "annulus":
                resp = click.prompt(
                    " - Enter comma-separated inner and outer radius (px)",
                    default="10, 16",
                )
                ann_rad = list(map(float, resp.replace(" ", "").split(",")))
                tpl.polarimetry.ip_radius = ann_rad[0]
                tpl.polarimetry.ip_radius2 = ann_rad[1]
    else:
        tpl.polarimetry = None

    tpl.save(config)
    click.echo(f"File saved to {config.name}")
    if not edit:
        edit |= click.confirm("Would you like to edit this config file now?")

    if edit:
        click.edit(filename=config)


########## normalize ##########


@main.command(name="norm", help="Normalize VAMPIRES data files")
@click.argument(
    "filenames",
    nargs=-1,
    type=click.Path(dir_okay=False, readable=True, path_type=Path),
)
@click.option("-o", "--outdir", type=Path, default=Path.cwd() / "prep", help="Output directory")
@click.option(
    "-d/-nd",
    "--deint/--no-deint",
    default=False,
    help="Deinterleave files into FLC states (WARNING: only apply this to old VAMPIRES data downloaded directly from `sonne`)",
)
@click.option(
    "-f/-nf",
    "--filter-empty/--no-filter-empty",
    default=True,
    help="Filter empty frames from data (post deinterleaving, if applicable)",
)
@click.option(
    "--num-proc",
    "-j",
    default=DEFAULT_NPROC,
    type=click.IntRange(1, cpu_count()),
    help="Number of processes to use.",
    show_default=True,
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Silence progress bars and extraneous logging.",
)
def norm(filenames, deint: bool, filter_empty: bool, num_proc: int, quiet: bool, outdir: Path):
    jobs = []
    kwargs = dict(deinterleave=deint, filter_empty=filter_empty, output_directory=outdir)
    with mp.Pool(num_proc) as pool:
        for filename in filenames:
            jobs.append(pool.apply_async(normalize_file, args=(filename,), kwds=kwargs))

        iter = jobs if quiet else tqdm.tqdm(jobs, desc="Normalizing files")
        results = [job.get() for job in iter]

    return results


########## run ##########


@main.command(name="run", help="Run the data processing pipeline")
@click.argument(
    "config",
    type=click.Path(dir_okay=False, readable=True, path_type=Path),
)
@click.argument(
    "filenames",
    nargs=-1,
    type=click.Path(dir_okay=False, readable=True, path_type=Path),
)
@click.option("-o", "--outdir", default=Path.cwd(), type=Path, help="Output file directory")
@click.option(
    "--num-proc",
    "-j",
    default=DEFAULT_NPROC,
    type=click.IntRange(1, cpu_count()),
    help="Number of processes to use.",
    show_default=True,
)
def run(config: Path, filenames, num_proc, outdir):
    # make sure versions match within SemVar
    logfile = outdir / "debug.log"
    logfile.unlink(missing_ok=True)
    logger.add(logfile, level="DEBUG", enqueue=True, colorize=False)
    pipeline = Pipeline(PipelineConfig.from_file(config), workdir=outdir)
    if not check_version(pipeline.config.dpp_version, dpp.__version__):
        raise ValueError(
            f"Input pipeline version ({pipeline.config.dpp_version}) is not compatible with installed version of `vampires_dpp` ({dpp.__version__}). Try running `dpp upgrade {config}`."
        )
    pipeline.run(filenames, num_proc=num_proc)
    pipeline.run_polarimetry(num_proc=num_proc)


########## pdi ##########


@main.command(name="pdi", help="Run the polarimetric differential imaging pipeline")
@click.argument(
    "config",
    type=click.Path(dir_okay=False, readable=True, path_type=Path),
)
@click.argument(
    "filenames",
    nargs=-1,
    type=click.Path(dir_okay=False, readable=True, path_type=Path),
)
@click.option("-o", "--outdir", default=Path.cwd(), type=Path, help="Output file directory")
@click.option(
    "--num-proc",
    "-j",
    default=DEFAULT_NPROC,
    type=click.IntRange(1, cpu_count()),
    help="Number of processes to use.",
    show_default=True,
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Silence progress bars and extraneous logging.",
)
def pdi(config, filenames, num_proc, quiet, outdir):
    # make sure versions match within SemVar
    logger.add(outdir / "debug.log", level="DEBUG", enqueue=True, colorize=False)
    pipeline = Pipeline(PipelineConfig.from_file(config), workdir=outdir)
    if not check_version(pipeline.config.dpp_version, dpp.__version__):
        raise ValueError(
            f"Input pipeline version ({pipeline.config.dpp_version}) is not compatible with installed version of `vampires_dpp` ({dpp.__version__}). Try running `dpp upgrade {config}`."
        )
    pipeline.run_polarimetry(num_proc=num_proc)


########## table ##########


@main.command(
    name="table",
    short_help="Create CSV from headers",
    help="Go through each file and combine the header information into a single CSV.",
)
@click.argument(
    "filenames",
    nargs=-1,
    type=click.Path(dir_okay=False, readable=True, path_type=Path),
)
@click.option(
    "-t",
    "--type",
    "_type",
    default="csv",
    type=click.Choice(["sql", "csv"], case_sensitive=False),
    help="Save as a CSV file or create a headers table in a sqlite database",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=(Path.cwd() / "header_table").name,
    help="Output path without file extension.",
)
@click.option(
    "--num-proc",
    "-j",
    default=DEFAULT_NPROC,
    type=click.IntRange(1, cpu_count()),
    help="Number of processes to use.",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Silence progress bars and extraneous logging.",
)
def table(filenames, _type, output, num_proc, quiet):
    # handle name clashes
    outpath = Path(output).resolve()
    if _type == "csv":
        outname = outpath.with_name(f"{outpath.name}.csv")
    elif _type == "sql":
        outname = outpath.with_name(f"{outpath.name}.db")

    if outname.exists():
        click.confirm(
            f"{outname.name} already exists in the output directory. Overwrite?", abort=True
        )
    df = header_table(filenames, num_proc=num_proc, quiet=quiet)
    if _type == "csv":
        df.to_csv(outname)
    elif _type == "sql":
        df.to_sql("headers", f"sqlite:///{outname.absolute()}")
    return outname


########## upgrade ##########


@main.command(
    name="upgrade",
    short_help="Upgrade configuration file",
    help=f"Tries to automatically upgrade a configuration file to the current version ({dpp.__version__}), prompting where necessary.",
)
@click.argument(
    "config",
    type=click.Path(dir_okay=False, readable=True, path_type=Path),
)
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
    with open(config, "rb") as fh:
        input_toml = tomli.load(fh)
    output_config = upgrade_config(input_toml)
    output_config.to_file(output)
    click.echo(f"File saved to {output.name}")
    if not edit:
        edit |= click.confirm("Would you like to edit this config file now?")

    if edit:
        click.edit(filename=output)


if __name__ == "__main__":
    main()
