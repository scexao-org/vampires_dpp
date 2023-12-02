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
from vampires_dpp.specphot.query import (
    get_simbad_flux,
    get_simbad_table,
    get_ucac_flux,
    get_ucac_table,
)
from vampires_dpp.wcs import get_gaia_astrometry

__all__ = "new_config"

########## new ##########


def pathCompleter(text: str, state: int) -> str:
    """This is the tab completer for systems paths.
    Only tested on *nix systems
    """
    # replace ~ with the user's home dir. See https://docs.python.org/2/library/os.path.html
    path = Path(text).expanduser()
    if path.is_dir():
        matches = path.glob("[!.]*")
    else:
        glob = f"{path.name}*"
        matches = path.parent.glob(glob)
    return str(list(matches)[state])


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
    return template


def get_calib_settings(template):
    readline.set_completer(pathCompleter)
    calib_dir = click.prompt("Enter path to calibration files", default="")
    readline.set_completer()
    if calib_dir != "":
        template.calibrate.calib_directory = Path(calib_dir)
        template.calibrate.back_subtract = click.confirm(
            " - Subtract backgrounds?", default=template.calibrate.back_subtract
        )
        template.calibrate.flat_correct = click.confirm(
            " - Flat correct?", default=template.calibrate.flat_correct
        )
    else:
        template.calibrate.calib_directory = None

    template.calibrate.save_intermediate = click.confirm(
        "Would you like to save calibrated files?", default=template.calibrate.save_intermediate
    )
    return template


def get_analysis_settings(template):
    ## Analysis
    template.analysis.window_size = click.prompt(
        "Enter analysis window size (px)", type=int, default=template.analysis.window_size
    )
    # template.analysis.subtract_radprof = click.confirm(
    #     "Would you like to subtract a radial profile for analysis?", default=template.coronagraphic
    # )
    # template.analysis.strehl = click.confirm(
    #     "Would you like to estimate the Strehl ratio?", default=False
    # )

    aper_rad = click.prompt('Enter aperture radius (px/"auto")', default=template.analysis.aper_rad)
    with contextlib.suppress(ValueError):
        aper_rad = float(aper_rad)
    if not isinstance(aper_rad, str) and aper_rad > template.analysis.window_size / 2:
        aper_rad = template.analysis.window_size / 2
        click.echo(f" ! Reducing aperture radius to match window size ({aper_rad:.0f} px)")
    template.analysis.aper_rad = aper_rad

    ann_rad = None
    if template.analysis.aper_rad != "auto" and click.confirm(
        "Would you like to subtract background annulus?", default=False
    ):
        in_rad = max(aper_rad, template.analysis.window_size / 2 - 5)
        out_rad = template.analysis.window_size / 2
        resp = click.prompt(
            " - Enter comma-separated inner and outer radius (px)", default=f"{in_rad}, {out_rad}"
        )
        ann_rad = list(map(float, resp.replace(" ", "").split(",")))
        if ann_rad[1] > template.analysis.window_size / 2:
            ann_rad[1] = template.analysis.window_size / 2
            click.echo(
                f" ! Reducing annulus outer radius to match window size ({ann_rad[1]:.0f} px)"
            )
        if ann_rad[0] >= ann_rad[1]:
            ann_rad[0] = max(aper_rad, ann_rad[0] - 5)
            click.echo(f" ! Reducing annulus inner radius to ({ann_rad[0]:.0f} px)")
        template.analysis.ann_rad = ann_rad

    template.analysis.dft_factor = click.prompt(
        " - Enter DFT upsample factor", default=template.analysis.dft_factor, type=int
    )
    return template


def get_specphot_settings(template):
    ## Specphot Cal
    if click.confirm(
        "Would you like to do flux calibration?", default=template.specphot is not None
    ):
        readline.set_completer(pathCompleter)
        source = click.prompt(
            ' - Enter source type ("pickles"/path to spectrum)', default="pickles"
        )
        readline.set_completer()
        if source == "pickles":
            if template.object is not None:
                simbad_table = get_simbad_table(template.object.object)
                sptype = re.match(r"\w\d[IV]{1,3}", simbad_table["SP_TYPE"][0]).group()
                ucac_table = get_ucac_table(template.object.object)
                ucac_res = get_ucac_flux(ucac_table)
                if ucac_res is not None:
                    mag, mag_band = ucac_res
                    click.echo(f" * Found UCAC4 info: {sptype} {mag_band}={mag:.02f}")
                else:
                    sim_res = get_simbad_flux(simbad_table)
                    if sim_res is not None:
                        mag, mag_band = sim_res
                        click.echo(f" * Found SIMBAD info: {sptype} {mag_band}={mag:.02f}")
                    else:
                        click.echo(" ! Could not determine object flux automatically")
                        mag = ""
                        mag_band = "V"

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
        template.specphot = SpecphotConfig(
            source=source, sptype=sptype, mag=mag, mag_band=mag_band, flux_metric=metric
        )
    else:
        template.specphot = None

    return template


def get_collapse_settings(template):
    ## Collapsing
    if click.confirm(
        "Would you like to collapse your data?", default=template.collapse is not None
    ):
        collapse_choices = ["median", "mean", "varmean", "biweight"]
        readline.set_completer(createListCompleter(collapse_choices))
        collapse_method = click.prompt(
            " - Choose a collapse method",
            type=click.Choice(collapse_choices, case_sensitive=False),
            default="median",
        )
        readline.set_completer()
        template.collapse = CollapseConfig(method=collapse_method)

        ## Frame selection
        if click.confirm(
            "Would you like to do frame selection?",
            default=template.collapse.frame_select is not None,
        ):
            template.collapse.select_cutoff = click.prompt(
                " - Enter a cutoff quantile (0 to 1, larger means more discarding)", type=float
            )

            metric_choices = ["normvar", "peak", "strehl"]
            readline.set_completer(createListCompleter(metric_choices))
            template.collapse.frame_select = click.prompt(
                " - Choose a frame selection metric",
                type=click.Choice(metric_choices, case_sensitive=False),
                default="normvar",
            )
            readline.set_completer()
        else:
            template.collapse.frame_select = None

        ## Registration
        if click.confirm(
            "Would you like to do frame registration?",
            default=template.collapse.centroid is not None,
        ):
            centroid_choices = ["dft", "com", "peak", "gauss", "quad"]
            readline.set_completer(createListCompleter(centroid_choices))
            template.collapse.centroid = click.prompt(
                " - Choose a registration method",
                type=click.Choice(centroid_choices, case_sensitive=False),
                default="dft",
            )
            readline.set_completer()
        else:
            template.collapse.centroid = None

        if click.confirm("Would you like to recenter the collapsed data?", default=True):
            template.collapse.recenter = click.prompt(
                " - Enter recenter registration method",
                type=click.Choice(["com", "peak", "gauss", "dft"], case_sensitive=False),
                default="com",
            )
        else:
            template.collapse.recenter = None
    else:
        template.collapse = None
    return template


def get_pdi_settings(template):
    ## Polarization
    if click.confirm("Would you like to do polarimetry?", default=template.polarimetry is not None):
        calib_choices = ["doublediff", "triplediff", "leastsq"]
        readline.set_completer(createListCompleter(calib_choices))
        pol_method = click.prompt(
            " - Choose a polarimetric calibration method",
            type=click.Choice(calib_choices, case_sensitive=False),
            default="triplediff",
        )
        readline.set_completer()
        template.polarimetry = PolarimetryConfig(method=pol_method)
        if "diff" in template.polarimetry.method:
            template.polarimetry.mm_correct = click.confirm(
                " - Would you like to to Mueller-matrix correction?",
                default=template.polarimetry.mm_correct,
            )
        template.polarimetry.use_ideal_mm = click.confirm(
            " - Would you like to use an idealized Muller matrix?",
            default=template.polarimetry.use_ideal_mm,
        )
        template.polarimetry.ip_correct = click.confirm(
            " - Would you like to do IP touchup?",
            default=template.polarimetry.ip_correct is not None,
        )
        if template.polarimetry.ip_correct:
            default = "annulus" if template.coronagraphic else "aperture"
            template.polarimetry.ip_method = click.prompt(
                " - Select IP correction method",
                type=click.Choice(["aperture", "annulus"], case_sensitive=False),
                default=default,
            )
            if template.polarimetry.ip_method == "aperture":
                template.polarimetry.ip_radius = click.prompt(
                    " - Enter IP aperture radius (px)", type=float, default=8
                )
            elif template.polarimetry.ip_method == "annulus":
                resp = click.prompt(
                    " - Enter comma-separated inner and outer radius (px)", default="10, 16"
                )
                ann_rad = list(map(float, resp.replace(" ", "").split(",")))
                template.polarimetry.ip_radius = ann_rad[0]
                template.polarimetry.ip_radius2 = ann_rad[1]
    else:
        template.polarimetry = None

    return template


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
    tpl = get_base_settings(tpl)
    tpl.name = name_guess if name == "" else name.replace(" ", "_").replace("/", "")
    tpl = get_target_settings(tpl)
    tpl = get_calib_settings(tpl)
    tpl = get_analysis_settings(tpl)
    tpl = get_specphot_settings(tpl)
    tpl = get_collapse_settings(tpl)
    tpl = get_pdi_settings(tpl)

    if click.confirm("Would you like to make difference images?", default=False):
        tpl.diff_images = click.prompt(
            " - Select difference method", type=click.Choice(("singlediff", "doublediff"))
        )
    else:
        tpl.diff_images = None

    tpl.save(config)
    click.echo(f"File saved to {config.name}")
    if not edit:
        edit |= click.confirm("Would you like to edit this config file now?")

    if edit:
        click.edit(filename=config)
