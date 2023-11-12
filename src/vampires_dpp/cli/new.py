import contextlib
import re
import readline
from pathlib import Path

import astropy.units as u
import click

from vampires_dpp.pipeline.config import (
    CollapseConfig,
    ObjectConfig,
    PolarimetryConfig,
    SpecphotConfig,
)
from vampires_dpp.pipeline.templates import (
    VAMPIRES_BLANK,
    VAMPIRES_PDI,
    VAMPIRES_SDI,
    VAMPIRES_SINGLECAM,
)
from vampires_dpp.specphot.filters import FILTERS
from vampires_dpp.specphot.query import get_simbad_table, get_ucac_flux, get_ucac_table
from vampires_dpp.wcs import get_gaia_astrometry

__all__ = "new_config"

########## new ##########


def pathCompleter(text: str, state: int) -> str:
    """This is the tab completer for systems paths.
    Only tested on *nix systems
    """
    # replace ~ with the user's home dir. See https://docs.python.org/2/library/os.path.html
    path = Path(text).expanduser()

    # autocomplete directories with having a trailing slash
    if path.is_dir():
        text += "/"

    matches = path.glob(text + "*")
    result = list(matches)[state]
    return str(result)


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


def get_starting_template():
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
    return tpl


def get_target_settings(template):
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
            template.object = ObjectConfig(
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

    return template


def get_base_settings(template):
    ## Coronagraph
    template.coronagraphic = click.confirm("Did you use a coronagraph?", default=False)
    template.save_adi_cubes = click.confirm(
        "Would you like to save ADI cubes?", default=template.save_adi_cubes
    )


@click.command(name="new", help="Generate configuration files")
@click.argument("config", type=click.Path(dir_okay=False, readable=True, path_type=Path))
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

    ## get name
    name_guess = config.stem
    name = click.prompt("Path-friendly name for this reduction", default=name_guess)

    tpl = get_starting_template()

    tpl.name = name_guess if name == "" else name.replace(" ", "_").replace("/", "")

    tpl = get_target_settings(tpl)

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
    with contextlib.suppress(ValueError):
        aper_rad = float(aper_rad)
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
            "Would you like to do frame registration?", default=tpl.collapse.centroid is not None
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
                    " - Enter comma-separated inner and outer radius (px)", default="10, 16"
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
