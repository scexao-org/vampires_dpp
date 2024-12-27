from pathlib import Path

import click
import numpy as np
from astropy.io import fits
from tqdm.auto import tqdm

from vampires_dpp.image_registration import get_mbi_cutout


def setup_ds9(cmap="viridis", scale="sqrt"):
    try:
        from pyds9 import DS9
    except ImportError as e:
        msg = "Could not import pyds9, make sure it is installed and working"
        raise RuntimeError(msg) from e
    ds9 = DS9()
    # clear frames
    ds9.set("frame delete all")
    # cmap and scale
    ds9.set(f"cmap {cmap}")
    if scale == "zscale":
        ds9.set("scale linear")
        ds9.set("scale zscale")
    elif scale == "sqrt":
        ds9.set("scale sqrt")
        ds9.set("scale minmax")
    elif scale == "log":
        ds9.set("scale log")
        ds9.set("scale minmax")
    # set up two frames in tile mode, go to first
    ds9.set("tile")
    ds9.set("lock frame image")
    # cube options
    ds9.set("cube interval 0.001")
    return ds9


def quick_view(filename, cmap="viridis", scale="zscale"):
    ds9 = setup_ds9(cmap, scale)
    cube, hdr = fits.getdata(filename, header=True)
    if "MBI" in hdr["OBS-MOD"]:
        cube = np.array(
            [get_mbi_cutout(frame, field="F720", camera=hdr["U_CAMERA"]).data for frame in cube]
        )
    frame = np.mean(cube, axis=0)
    ds9.set("frame new")
    nz, ny, nx = cube.shape
    ds9.set(f"array [xdim={nx},ydim={ny},zdim={nz},bitpix={hdr['BITPIX']}]", cube)
    ds9.set("frame new")
    ds9.set_np2arr(frame)
    ds9.set("frame first")
    ds9.set("cube play")
    return frame


@click.command(name="select", help="View files in DS9 and sort out bad files")
@click.argument("filenames", nargs=-1, type=Path)
def quick_select(filenames):
    # handle name clashes
    outdir = Path.cwd()
    select_path = outdir / "filelist_select.txt"
    reject_path = outdir / "filelist_reject.txt"
    if select_path.is_file():
        msg = f"{select_path.name} already exists in the output directory. Overwrite?"
        if click.confirm(msg, default=False):
            select_path.unlink()
            reject_path.unlink()

    select_path.parent.mkdir(parents=True, exist_ok=True)

    for filename in tqdm(filenames, desc="Progress"):
        path = Path(filename)
        quick_view(path, scale="sqrt")
        if click.confirm("Would you like to keep this file?", default=True):
            outpath = select_path
        else:
            outpath = reject_path
        with outpath.open("a") as fh:
            fh.write(f"{path}\n")


if __name__ == "__main__":
    quick_select()
