import re
import readline
from pathlib import Path

import astropy.units as u
import click

from vampires_dpp.pipeline.config import (
    NRMConfig,
    PipelineConfig,
    PolarimetryConfig,
    SpecphotConfig,
    TargetConfig,
)
from vampires_dpp.specphot.filters import FILTERS
from vampires_dpp.specphot.pickles import check_spectral_type_in_pickles
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


def get_target_settings(template: PipelineConfig) -> PipelineConfig:
    ## get target
    name = click.prompt("SIMBAD-friendly object name (optional)", default="")
    coord = None
    if name != "":
        rad = 1
        cat = "dr3"
        while True:
            coord = get_gaia_astrometry(name, catalog=cat, radius=rad)
            if coord is not None:
                break

            click.echo(f'  Could not find {name} in GAIA {cat.upper()} with {rad}" radius.')
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
                        name = _input

        if coord is not None:
            template.target = TargetConfig(
                name=name,
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


def get_base_settings(template: PipelineConfig) -> PipelineConfig:
    ## Coronagraph
    template.coronagraphic = click.confirm("Did you use a coronagraph?", default=False)
    if not template.coronagraphic:
        template.planetary = click.confirm("Was this a planet/solar system object?", default=False)
    template.save_adi_cubes = click.confirm(
        "Would you like to save ADI cubes?", default=template.save_adi_cubes
    )
    return template


def get_calib_settings(template: PipelineConfig) -> PipelineConfig:
    click.secho("Frame Calibration", bold=True)
    readline.set_completer(pathCompleter)
    calib_dir = click.prompt("Enter path to calibration files (or press enter to skip)", default="")
    readline.set_completer()
    if calib_dir != "":
        template.calibrate.calib_directory = Path(calib_dir)
        template.calibrate.back_subtract = click.confirm(
            " - Subtract backgrounds (if available)?", default=template.calibrate.back_subtract
        )
        template.calibrate.flat_correct = click.confirm(
            " - Flat correct (if available)?", default=template.calibrate.flat_correct
        )
    else:
        template.calibrate.calib_directory = None
    template.calibrate.save_intermediate = click.confirm(
        "Would you like to save intermediate calibrated files?",
        default=template.calibrate.save_intermediate,
    )
    return template


def get_analysis_settings(template: PipelineConfig) -> PipelineConfig:
    click.secho("Frame Analysis", bold=True)
    ## Analysis
    template.analysis.window_size = click.prompt(
        "Enter analysis window size (px)", type=int, default=template.analysis.window_size
    )
    template.analysis.fit_psf_model = click.confirm(
        "Would you like to fit a PSF model?", default=template.analysis.fit_psf_model
    )
    # if template.analysis.fit_psf_model:
    #     template.analysis.psf_model = click.prompt(
    #         " - Choose PSF model",
    #         type=click.Choice(["moffat", "gaussian"], case_sensitive=False),
    #         default=template.analysis.psf_model,
    #     )
    template.analysis.photometry = click.confirm(
        "Would you like to do aperture photometry?", default=template.analysis.photometry
    )
    if template.analysis.photometry:
        aper_rad = click.prompt(
            "Enter aperture radius (px)", default=template.analysis.phot_aper_rad, type=float
        )
        if aper_rad > template.analysis.window_size / 2:
            aper_rad = template.analysis.window_size / 2
            click.echo(f" ! Reducing aperture radius to match window size ({aper_rad:.0f} px)")
        template.analysis.phot_aper_rad = aper_rad

        ann_rad = None
        if click.confirm("Would you like to subtract background annulus?", default=False):
            in_rad = max(aper_rad, template.analysis.window_size / 2 - 5)
            out_rad = template.analysis.window_size / 2
            resp = click.prompt(
                " - Enter comma-separated inner and outer radius (px)",
                default=f"{in_rad}, {out_rad}",
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
            template.analysis.phot_ann_rad = ann_rad

    template.analysis.strehl = click.confirm(
        "Would you like to measure the Strehl ratio?", default=template.analysis.strehl
    )

    return template


def get_combine_settings(template: PipelineConfig) -> PipelineConfig:
    click.secho("Combining Data", bold=True)
    template.combine.method = click.prompt(
        "How would you like to combine data?",
        type=click.Choice(["cube", "pdi"], case_sensitive=False),
        default=template.combine.method,
    )
    template.combine.save_intermediate = click.confirm(
        "Would you like to save the intermediate combined data?",
        default=template.combine.save_intermediate,
    )
    return template


def get_frame_select_settings(template: PipelineConfig) -> PipelineConfig:
    ## Frame selection
    click.secho("Frame Selection", bold=True)
    template.frame_select.frame_select = click.confirm(
        "Would you like to do frame selection?", default=template.frame_select.frame_select
    )
    if template.frame_select.frame_select:
        metric_choices = ["normvar", "peak", "strehl"]
        readline.set_completer(createListCompleter(metric_choices))
        template.frame_select.metric = click.prompt(
            " - Choose a frame selection metric",
            type=click.Choice(metric_choices, case_sensitive=False),
            default=template.frame_select.metric,
        )
        template.frame_select.cutoff = click.prompt(
            " - Enter a cutoff quantile (0 to 1, larger means more discarding)", type=float
        )
        readline.set_completer()
        template.frame_select.save_intermediate = click.confirm(
            "Would you like to save the intermediate selected data?",
            default=template.frame_select.save_intermediate,
        )

    return template


def get_alignment_settings(template: PipelineConfig) -> PipelineConfig:
    ## Registration
    click.secho("Frame Alignment", bold=True)
    template.align.align = click.confirm(
        "Would you like to align each frame?", default=template.align.align
    )
    if template.align.align:
        method_choices = ["dft", "com", "peak", "model"]
        readline.set_completer(createListCompleter(method_choices))
        template.align.method = click.prompt(
            " - Choose centroiding metric",
            type=click.Choice(method_choices, case_sensitive=False),
            default=template.align.method,
        )
        readline.set_completer()
    template.align.pad = click.confirm(
        "Would you like to pad each frame?", default=template.align.pad
    )
    template.align.crop_width = click.prompt(
        "Enter post-align crop size", default=template.align.crop_width, type=int
    )
    template.align.reproject = click.confirm(
        "Would you like to reproject VCAM2 data to fix scale and rotation differences?",
        default=template.align.reproject,
    )
    template.align.save_intermediate = click.confirm(
        "Would you like to save the intermediate registered data?",
        default=template.align.save_intermediate,
    )

    return template


def get_coadd_settings(template: PipelineConfig) -> PipelineConfig:
    ## Collapsing
    click.secho("Coadding", bold=True)
    template.coadd.coadd = click.confirm(
        "Would you like to coadd your data?", default=template.coadd.coadd
    )
    if template.coadd.coadd:
        collapse_choices = ["median", "mean", "varmean", "biweight"]
        readline.set_completer(createListCompleter(collapse_choices))
        template.coadd.method = click.prompt(
            " - Choose a coadd method",
            type=click.Choice(collapse_choices, case_sensitive=False),
            default=template.coadd.method,
        )
        readline.set_completer()

        template.coadd.recenter = click.confirm(
            " - Would you like to recenter the coadded data?", default=template.coadd.recenter
        )
        if template.coadd.recenter:
            template.coadd.recenter_method = click.prompt(
                " - Enter post-align registration method",
                type=click.Choice(["com", "peak", "gauss", "dft"], case_sensitive=False),
                default=template.coadd.recenter_method,
            )
    return template


def get_diff_image_config(template: PipelineConfig) -> PipelineConfig:
    ## Diff images
    click.secho("Difference Imaging", bold=True)
    template.diff_images.make_diff = click.confirm(
        "Would you like to make difference/sum images?", default=template.diff_images.make_diff
    )
    if template.diff_images.make_diff:
        template.diff_images.save_double = click.confirm(
            "Would you like to save the double diff/sum images (only works if FLC was used)?",
            default=template.diff_images.save_double,
        )
    return template


def get_specphot_settings(template: PipelineConfig) -> PipelineConfig:
    click.secho("Spectrophotometric Calibration", bold=True)
    ## Specphot Cal
    unit_choices = ["e-/s", "contrast", "Jy", "Jy/arcsec^2"]
    readline.set_completer(createListCompleter(unit_choices))
    unit = click.prompt(
        "Choose output units",
        type=click.Choice(unit_choices, case_sensitive=False),
        default=template.specphot.unit,
    )
    readline.set_completer()
    # default values
    source = sptype = mag = mag_band = None
    metric = "photometry"
    if "Jy" in unit:
        readline.set_completer(pathCompleter)
        source = click.prompt(
            ' - Enter spectrum source ("pickles", "zeropoints", or path to spectrum)',
            default="pickles",
        )
        readline.set_completer()
        if source == "pickles":
            if template.target is not None:
                click.echo("...Attempting to look up stellar flux from UCAC4/SIMBAD")
                simbad_table = get_simbad_table(template.target.name)
                sptype = re.match(r"\w\d[IV]{0,3}", simbad_table["SP_TYPE"][0]).group()
                if len(sptype) == 2:
                    sptype += "V"

                ucac_table = get_ucac_table(template.target.name)
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
            while not check_spectral_type_in_pickles(sptype):
                sptype = click.prompt(" - Enter spectral type", default=sptype)

            mag = click.prompt(" - Enter source magnitude", default=mag, type=float)
            mag_band = click.prompt(
                " - Enter source magnitude passband",
                default=mag_band,
                type=click.Choice(list(FILTERS.keys()), case_sensitive=False),
            )
    if unit != "e-/s" and source != "zeropoints":
        metric = click.prompt(
            " - Select which metric to use for flux",
            default="photometry",
            type=click.Choice(["photometry", "sum"]),
        )
    template.specphot = SpecphotConfig(
        unit=unit, source=source, sptype=sptype, mag=mag, mag_band=mag_band, flux_metric=metric
    )

    return template


def _get_pdi_settings_nrm(template: PipelineConfig) -> PipelineConfig:
    calib_choices = ["doublediff", "triplediff"]
    readline.set_completer(createListCompleter(calib_choices))
    pol_method = click.prompt(
        " - Choose a polarimetric calibration method",
        type=click.Choice(calib_choices, case_sensitive=False),
        default="triplediff",
    )
    readline.set_completer()

    template.polarimetry = PolarimetryConfig(method=pol_method)
    template.polarimetry.mm_correct = True
    template.polarimetry.use_ideal_mm = click.confirm(
        " - Would you like to use an idealized Muller matrix?",
        default=template.polarimetry.use_ideal_mm,
    )
    template.polarimetry.ip_correct = False
    return template


def get_pdi_settings(template: PipelineConfig) -> PipelineConfig:
    click.secho("Polarimetry", bold=True)
    ## Polarization
    if click.confirm("Would you like to do polarimetry?", default=template.polarimetry is not None):
        # NRM needs special case, give its own function
        if template.nrm is not None:
            return _get_pdi_settings_nrm(template)
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

        template.polarimetry.cyl_stokes = click.prompt(
            " - Choose cylindrical Stokes system",
            type=click.Choice(["azimuthal", "radial"], case_sensitive=False),
            default=template.polarimetry.cyl_stokes,
        )
    else:
        template.polarimetry = None

    return template


def _set_nrm_defaults(template: PipelineConfig) -> PipelineConfig:
    """Set defaults for other configs when NRM is running"""
    # analysis--avoid PSF fitting/modeling, these aren't PSFs
    template.analysis.fit_psf_model = False
    template.analysis.photometry = False
    template.analysis.strehl = False

    # frame select -- use normvar or peak, not strehl
    template.frame_select.metric = "normvar"

    # alignement -- peak index works well, don't pad, and crop to 512
    template.align.align = True
    template.align.method = "peak"
    template.align.pad = False
    template.align.crop_width = 512
    template.align.reproject = False

    # coadd -- don't
    template.coadd.coadd = False
    template.coadd.recenter = False
    template.coadd.recenter_method = "peak"

    # specphot
    template.specphot.unit = "e-/s"

    return template


def get_nrm_settings(template: PipelineConfig) -> PipelineConfig:
    ## Collapsing
    click.secho("NRM", bold=True)
    do_nrm = click.confirm("Would you like to do NRM analysis?", default=template.nrm is not None)
    if do_nrm:
        template.nrm = NRMConfig()
        template = _set_nrm_defaults(template)
    else:
        template.nrm = None
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

    tpl = PipelineConfig()
    tpl = get_base_settings(tpl)
    tpl.name = name_guess if name == "" else name.replace(" ", "_").replace("/", "")
    tpl = get_target_settings(tpl)
    tpl = get_combine_settings(tpl)
    tpl = get_calib_settings(tpl)
    # run NRM early to set a bunch of defaults
    tpl = get_nrm_settings(tpl)
    tpl = get_analysis_settings(tpl)
    tpl = get_frame_select_settings(tpl)
    tpl = get_alignment_settings(tpl)
    tpl = get_coadd_settings(tpl)
    tpl = get_specphot_settings(tpl)
    tpl = get_diff_image_config(tpl)
    tpl = get_pdi_settings(tpl)

    tpl.save(config)
    click.echo(f"File saved to {config.name}")

    if not edit:
        edit |= click.confirm("Would you like to edit this config file now?")

    if edit:
        click.edit(filename=config)
