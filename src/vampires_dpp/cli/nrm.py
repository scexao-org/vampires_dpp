import multiprocessing
from collections.abc import Sequence
from pathlib import Path

import click
import numpy as np
import tomli
import tomli_w
from astropy.nddata import Cutout2D
from numpy.typing import NDArray

from vampires_dpp._logging import configure_logging
from vampires_dpp.cli.centroids import create_raw_input_psfs
from vampires_dpp.nrm.alignment import check_mask_align
from vampires_dpp.nrm.params import get_amical_parameters
from vampires_dpp.nrm.windowing import window_cube
from vampires_dpp.organization import header_table
from vampires_dpp.paths import Paths
from vampires_dpp.pipeline.config import PipelineConfig
from vampires_dpp.specphot.filters import determine_filterset_from_header

logger = configure_logging()


@click.group(name="nrm", short_help="NRM specific tools", help="NRM mask alignment, analysis")
@click.option(
    "--num-proc",
    "-j",
    default=1,
    type=click.IntRange(1, multiprocessing.cpu_count()),
    help="Number of processes to use.",
    show_default=True,
)
@click.pass_context
def nrm(ctx, num_proc):
    # prepare context
    ctx.ensure_object(dict)
    ctx.obj["num_proc"] = num_proc


@nrm.command(
    name="align",
    short_help="Check uv scale/theta alignment",
    help="Check uv scale/theta alignment from mean combined data in `aux` directory. Plots are saved to `nrm/figures`. Adjust `nrm.uv` and `nrm.theta` in the TOML file to test different values.",
)
@click.argument("config", type=click.Path(dir_okay=False, readable=True, path_type=Path))
@click.argument(
    "filenames", nargs=-1, type=click.Path(dir_okay=False, readable=True, path_type=Path)
)
@click.option("-o", "--outdir", default=Path.cwd(), type=Path, help="Output file directory")
@click.pass_context
def align(ctx, config: Path, filenames, outdir):
    # make sure versions match within SemVar
    pipeline_config = PipelineConfig.from_file(config)
    # figure out outpath
    paths = Paths(outdir)
    paths.aux.mkdir(parents=True, exist_ok=True)

    centroids = get_centroids(paths.aux, pipeline_config.name)
    # npsfs = 4 if pipeline_config.coronagraphic else 1
    # choose 5 random files
    table = header_table(filenames, num_proc=ctx.obj["num_proc"]).sort_values(["MJD", "U_CAMERA"])
    obsmodes = table["OBS-MOD"].unique()
    if len(obsmodes) > 1:
        msg = f"Found {len(obsmodes)} unique OBS-MOD, make sure you're only processing one type of VAMPIRES data. Will proceed with first mode: {obsmodes.iloc[0]}"
        click.echo(msg)
    # default for standard obs, overwritten by MBI
    # fields = det # choose 5 random files
    fields = determine_filterset_from_header(table.iloc[0])
    cams = table["U_CAMERA"].unique()
    uv_thetas = get_uv_theta_manual(cams=cams, fields=fields)
    basename = paths.aux / f"{pipeline_config.name}_uv_theta"
    save_uv_theta(uv_thetas, fields, basename)
    name = paths.aux / f"{pipeline_config.name}_mean_image"
    # choose 4 to 20 files, depending on file size (avoid loading more than 500 frames, ~2GB of MBI)
    number_files = int(max(4, min(10, 500 // table["NAXIS3"].median())))
    input_hduls_dict = create_raw_input_psfs(table, basename=name, max_files=number_files)

    figdir = paths.nrm / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    for key, input_hdul in input_hduls_dict.items():
        input_frame = np.nan_to_num(input_hdul[0].data)
        header = input_hdul[0].header
        # get cutout/s
        for field_idx, centroid in enumerate(centroids[key].values()):
            size = 256
            cutout = Cutout2D(input_frame, centroid[0, ::-1], size).data
            if header["U_CAMERA"] == 1:
                cutout = np.flipud(cutout)
            cube, header = window_cube(cutout[None, :, :], size=80, header=header)
            uv, theta = uv_thetas[key][field_idx]

            save_basename = figdir / f"{key}_{fields[field_idx]}_uv={uv}_theta={theta}_"
            check_mask_align(
                cube,
                params=get_amical_parameters(header),
                save_path=str(save_basename),
                uv=uv,
                theta=theta,
            )


def get_centroids(basepath, name):
    centroids = {}
    for key in ("cam1", "cam2"):
        path = basepath / f"{name}_centroids_{key}.toml"
        if not path.exists():
            msg = f"Could not locate centroid file for {key}, expected it to be at {path}."
            msg += " Make sure you have run `dpp centroid` first."
            raise ValueError(msg)
        with path.open("rb") as fh:
            _centroids = tomli.load(fh)
        centroids[key] = {}
        for field, ctrs in _centroids.items():
            centroids[key][field] = np.flip(np.atleast_2d(ctrs), axis=-1)

        logger.debug(f"{key} frame center is {centroids[key]} (y, x)")
    return centroids


def get_uv_theta_manual(cams: Sequence[int], fields: Sequence[str]) -> dict[str, NDArray]:
    uv_theta = {f"cam{cam:.0f}": np.empty((len(fields), 2), dtype="f4") for cam in cams}
    for cam_key, arr in uv_theta.items():
        click.echo(f"Enter comma-separated uv, theta pair for {click.style(cam_key, bold=True)}:")
        for i, field in enumerate(fields):
            response = click.prompt(f" - Field: {field} (uv, theta_deg)")
            arr[i] = [float(r) for r in response.split(",")]
    return uv_theta


def save_uv_theta(
    uv_thetas: dict[str, NDArray], fields: Sequence[str], basename: Path
) -> dict[str, Path]:
    outpaths = {}
    for cam_key, uv_theta_arr in uv_thetas.items():
        outpaths[cam_key] = basename.with_name(f"{basename.name}_{cam_key}.toml")
        payload = {}
        for field_idx in range(uv_theta_arr.shape[0]):
            field = fields[field_idx]
            payload[field] = {
                "uv": float(uv_theta_arr[field_idx, 0]),
                "theta": float(uv_theta_arr[field_idx, 1]),
            }
        with outpaths[cam_key].open("wb") as fh:
            tomli_w.dump(payload, fh)
        logger.info(f"Saved {cam_key} mask parameters to {outpaths[cam_key]}")
    return outpaths
