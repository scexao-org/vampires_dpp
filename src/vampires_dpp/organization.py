import multiprocessing as mp
import shutil
from collections import OrderedDict
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm.auto import tqdm

from .headers import fix_header
from .util import load_fits_header


def dict_from_header_file(filename: PathLike, **kwargs) -> OrderedDict:
    """Parse a FITS header from a file and extract the keys and values as an ordered dictionary. Multi-line keys like ``COMMENTS`` and ``HISTORY`` will be combined with commas. The resolved path will be inserted with the ``path`` key.

    Parameters
    ----------
    filename : str
        FITS file to parse
    **kwargs
        All keyword arguments will be passed to ``load_fits_header``

    Returns
    -------
    OrderedDict
    """
    path = Path(filename)
    summary = OrderedDict()
    # add path to row before the FITS header keys
    summary["path"] = str(path.resolve().absolute())
    header = load_fits_header(filename)
    summary.update(dict_from_header(header, **kwargs))
    return summary


def dict_from_header(header: fits.Header, excluded=("COMMENT", "HISTORY"), fix=True) -> OrderedDict:
    """Parse a FITS header and extract the keys and values as an ordered dictionary. Multi-line keys like ``COMMENTS`` and ``HISTORY`` will be combined with commas. The resolved path will be inserted with the ``path`` key.

    Parameters
    ----------
    header : Header
        FITS header to parse

    Returns
    -------
    OrderedDict
    """
    header = fix_header(header) if fix else header
    summary = OrderedDict()
    for k, v in header.items():
        if k == "" or k in excluded:
            continue
        summary[k] = v
    return summary


def header_table(
    filenames: list[PathLike], num_proc: int | None = None, quiet: bool = False, **kwargs
) -> pd.DataFrame:
    """Generate a pandas dataframe from the FITS headers parsed from the given files.

    Parameters
    ----------
    filenames : list[pathlike]
    num_proc : int, optional
        Number of processes to use in multiprocessing, by default mp.cpu_count()
    quiet : bool, optional
        Silence the progress bar, by default False

    Returns
    -------
    pandas.DataFrame
    """
    if num_proc is None:
        num_proc = min(8, mp.cpu_count())
    with mp.Pool(num_proc) as pool:
        jobs = [pool.apply_async(dict_from_header_file, args=(f,), kwds=kwargs) for f in filenames]
        iter = jobs if quiet else tqdm(jobs, desc="Parsing FITS headers")
        rows = [job.get() for job in iter]

    return pd.DataFrame(rows)


# set up commands for parser to dispatch to
def sort_files(
    filenames: list[PathLike],
    copy: bool = False,
    output_directory: PathLike | None = None,
    num_proc: int = min(8, mp.cpu_count()),
    quiet: bool = False,
    decompress: bool = False,
    **kwargs,
) -> list[Path]:
    outdir = Path(output_directory) if output_directory is not None else Path(filenames[0]).parent
    jobs = []
    with mp.Pool(num_proc) as pool:
        for filename in filenames:
            kwds = dict(outdir=outdir, copy=copy, decompress=decompress, **kwargs)
            jobs.append(pool.apply_async(sort_file, args=(filename,), kwds=kwds))

        iter = jobs if quiet else tqdm(jobs, desc="Sorting files")
        results = [job.get() for job in iter]

    return results


def sort_file(
    filename: PathLike, outdir: PathLike, copy: bool = False, decompress: bool = False, **kwargs
) -> Path:
    path = Path(filename)
    header = load_fits_header(path, **kwargs)

    # data pre 2023/02/02 does not store DATA-TYP
    # meaningfully, so use ad-hoc sorting method
    if header["DATA-TYP"] == "ACQUISITION":
        foldname = foldername_old(outdir, path, header)
    elif "U_EMGAIN" in header:
        foldname = foldername_halfold(outdir, header)
    else:
        foldname = foldername_new(outdir, header)

    newname = foldname / path.name
    foldname.mkdir(parents=True, exist_ok=True)
    if decompress:
        newname = foldname / path.name.replace(".fits.fz", ".fits")
        if not newname.exists():
            with fits.open(path) as hdul:
                if len(hdul) < 2:
                    msg = f"{path}  did not have expected HDU at index 1"
                    raise RuntimeError(msg)
                fits.writeto(newname, hdul[1].data, header=hdul[1].header)
    elif copy:
        shutil.copy(path, newname)
    else:
        path.replace(newname)
    return newname


def foldername_new(outdir: Path, header: fits.Header):
    filt1 = header["FILTER01"]
    filt2 = header["FILTER02"]
    filt_str = f"{filt1}_{filt2}"
    exptime = header["EXPTIME"] * 1e6  # us
    sz = f"{header['NAXIS1']:03d}x{header['NAXIS2']:03d}"
    match header["DATA-TYP"]:
        case "OBJECT":
            # subsort based on filter, EM gain, and exposure time
            subdir = f"{filt_str}_{exptime:09.0f}us_{sz}"
            foldname = outdir / header["OBJECT"].replace(" ", "_") / subdir
        case "DARK":
            subdir = f"{exptime:09.0f}us_{sz}"
            foldname = outdir / "darks" / subdir
        # put sky flats separately because they are usually
        # background frames, not flats
        case "SKYFLAT":
            subdir = f"{exptime:09.0f}us_{sz}"
            foldname = outdir / "skies" / subdir
        case "FLAT" | "DOMEFLAT":
            subdir = f"{filt_str}_{exptime:09.0f}us_{sz}"
            foldname = outdir / "flats" / subdir
        case "COMPARISON":
            subdir = f"{filt_str}_{exptime:09.0f}us_{sz}"
            foldname = outdir / "pinholes" / subdir
        case _:
            foldname = outdir

    return foldname


def foldername_halfold(outdir: Path, header: fits.Header):
    filt = header["U_FILTER"]
    gain = header["U_EMGAIN"]
    exptime = header["U_AQTINT"] / 1e3  # ms
    sz = f"{header['NAXIS1']:03d}x{header['NAXIS2']:03d}"
    match header["DATA-TYP"]:
        case "OBJECT":
            # subsort based on filter, EM gain, and exposure time
            subdir = f"{filt}_em{gain:.0f}_{exptime:05.0f}ms_{sz}"
            foldname = outdir / header["OBJECT"].replace(" ", "_") / subdir
        case "DARK":
            subdir = f"em{gain:.0f}_{exptime:05.0f}ms_{sz}"
            foldname = outdir / "darks" / subdir
        # put sky flats separately because they are usually
        # background frames, not flats
        case "SKYFLAT":
            subdir = f"em{gain:.0f}_{exptime:05.0f}ms_{sz}"
            foldname = outdir / "skies" / subdir
        case "FLAT" | "DOMEFLAT":
            subdir = f"{filt}_em{gain:.0f}_{exptime:05.0f}ms_{sz}"
            foldname = outdir / "flats" / subdir
        case "COMPARISON":
            subdir = f"{filt}_em{gain:.0f}_{exptime:05.0f}ms_{sz}"
            foldname = outdir / "pinholes" / subdir
        case _:
            foldname = outdir

    return foldname


def foldername_old(outdir: Path, path: Path, header: fits.Header):
    name = header.get("U_OGFNAM", path.name)
    filt = header["U_FILTER"]
    gain = header["U_EMGAIN"]
    exptime = header["U_AQTINT"] / 1e3  # ms
    sz = f"{header['NAXIS1']:03d}x{header['NAXIS2']:03d}"
    if "dark" in name:
        subdir = f"em{gain:.0f}_{exptime:05.0f}ms_{sz}"
        foldname = outdir / "darks" / subdir
    elif "skies" in name or "sky" in name:
        subdir = f"em{gain:.0f}_{exptime:05.0f}ms_{sz}"
        foldname = outdir / "skies" / subdir
    elif "flat" in name:
        subdir = f"{filt}_em{gain:.0f}_{exptime:05.0f}ms_{sz}"
        foldname = outdir / "flats" / subdir
    elif "pinhole" in name:
        subdir = f"{filt}_em{gain:.0f}_{exptime:05.0f}ms_{sz}"
        foldname = outdir / "pinholes" / subdir
    else:
        # subsort based on filter, EM gain, and exposure time
        subdir = f"{filt}_em{gain:.0f}_{exptime:05.0f}ms_{sz}"
        foldname = outdir / header["OBJECT"].replace(" ", "_") / subdir

    return foldname


def check_file(filename) -> bool:
    """Checks if file can be loaded and if there are empty slices

    Parameters
    ----------
    filename : PathLike

    Returns
    -------
    bool
        Returns True if file can be loaded and no empty frames
    """
    path = Path(filename)
    ext = 1 if ".fits.fz" in path.name else 0
    with fits.open(path) as hdus:
        hdr = hdus[ext].header
        data = hdus[ext].data
        if "U_FLCSTT" not in hdr:
            data = data[2:]
        return np.any(data, axis=(-2, -1)).all()


def check_files(
    filenames: list[PathLike], num_proc: int = min(8, mp.cpu_count()), quiet: bool = False
) -> list[bool]:
    jobs = []
    with mp.Pool(num_proc) as pool:
        for filename in filenames:
            jobs.append(pool.apply_async(check_file, args=(filename,)))

        iter = jobs if quiet else tqdm(jobs, desc="Checking files")
        results = [job.get() for job in iter]

    return results
