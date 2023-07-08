import multiprocessing as mp
import shutil
from collections import OrderedDict
from os import PathLike
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm.auto import tqdm


def dict_from_header_file(filename: PathLike, **kwargs) -> OrderedDict:
    """
    Parse a FITS header from a file and extract the keys and values as an ordered dictionary. Multi-line keys like ``COMMENTS`` and ``HISTORY`` will be combined with commas. The resolved path will be inserted with the ``path`` key.

    Parameters
    ----------
    filename : str
        FITS file to parse
    **kwargs
        All keyword arguments will be passed to ``fits.getheader``

    Returns
    -------
    OrderedDict
    """
    path = Path(filename)
    summary = OrderedDict()
    # add path to row before the FITS header keys
    summary["path"] = path.resolve()
    ext = 1 if ".fits.fz" in path.name else 0
    header = fits.getheader(filename, ext=ext, **kwargs)
    summary.update(dict_from_header(header))
    return summary


def dict_from_header(header: fits.Header) -> OrderedDict:
    """
    Parse a FITS header and extract the keys and values as an ordered dictionary. Multi-line keys like ``COMMENTS`` and ``HISTORY`` will be combined with commas. The resolved path will be inserted with the ``path`` key.

    Parameters
    ----------
    header : Header
        FITS header to parse

    Returns
    -------
    OrderedDict
    """
    summary = OrderedDict()
    multi_entry_keys = {"COMMENT": [], "HISTORY": []}
    for k, v in header.items():
        if k == "":
            continue
        if k in multi_entry_keys:
            multi_entry_keys[k].append(v.lstrip())
        summary[k] = v

    for k, l in multi_entry_keys.items():
        if len(l) > 0:
            summary[k] = ", ".join(l)

    return summary


def header_table(
    filenames: list[PathLike],
    num_proc: int = min(8, mp.cpu_count()),
    quiet: bool = False,
) -> pd.DataFrame:
    """
    Generate a pandas dataframe from the FITS headers parsed from the given files.

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
    with mp.Pool(num_proc) as pool:
        jobs = [pool.apply_async(dict_from_header_file, args=(f,)) for f in filenames]
        iter = jobs if quiet else tqdm(jobs, desc="Parsing FITS headers")
        rows = [job.get() for job in iter]

    df = pd.DataFrame(rows)
    df.sort_values("MJD", inplace=True)
    return df


# set up commands for parser to dispatch to
def sort_files(
    filenames: list[PathLike],
    copy: bool = False,
    output_directory: Optional[PathLike] = None,
    num_proc: int = min(8, mp.cpu_count()),
    quiet: bool = False,
    **kwargs,
) -> list[Path]:
    if output_directory is not None:
        outdir = Path(output_directory)
    else:
        outdir = Path(filenames[0]).parent
    jobs = []
    with mp.Pool(num_proc) as pool:
        for filename in filenames:
            kwds = dict(outdir=outdir, copy=copy, **kwargs)
            jobs.append(pool.apply_async(sort_file, args=(filename,), kwds=kwds))

        iter = jobs if quiet else tqdm(jobs, desc="Sorting files")
        results = [job.get() for job in iter]

    return results


def sort_file(filename: PathLike, outdir: PathLike, copy: bool = False, **kwargs) -> Path:
    path = Path(filename)
    header = fits.getheader(path, **kwargs)

    foldname = foldername_new(outdir, header)

    newname = foldname / path.name
    foldname.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy(path, newname)
    else:
        path.replace(newname)
    return newname


def foldername_new(outdir: PathLike, header: fits.Header):
    filt = f"{header['FILTER01']}_{header['FILTER02']}"
    exptime = header["EXPTIME"] * 1e6  # us
    sz = f"{header['NAXIS1']:04d}x{header['NAXIS2']:04d}"
    match header["DATA-TYP"]:
        case "OBJECT":
            # subsort based on filter and exposure time
            subdir = f"{filt}_{exptime:09.0f}us_{sz}"
            foldname = outdir / header["OBJECT"].replace(" ", "_") / subdir
        case "DARK":
            subdir = f"{exptime:05.0f}us_{sz}"
            foldname = outdir / "darks" / subdir
        # put sky flats separately because they are usually
        # background frames, not flats
        case "SKYFLAT":
            subdir = f"{exptime:09.0f}us_{sz}"
            foldname = outdir / "skies" / subdir
        case "FLAT" | "DOMEFLAT":
            subdir = f"{filt}_{exptime:09.0f}us_{sz}"
            foldname = outdir / "flats" / subdir
        case "COMPARISON":
            subdir = f"{filt}_{exptime:09.0f}us_{sz}"
            foldname = outdir / "pinholes" / subdir
        case _:
            foldname = outdir

    return foldname


def check_file(filename) -> bool:
    """
    Checks if file can be loaded and if there are empty slices

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
    try:
        with fits.open(filename) as hdus:
            hdus[ext].header
            data = hdus[ext].data
            return np.any(data, axis=(-2, -1)).all()
    except:
        return False


def check_files(
    filenames: list[PathLike],
    num_proc: int = min(8, mp.cpu_count()),
    quiet: bool = False,
) -> list[bool]:
    jobs = []
    with mp.Pool(num_proc) as pool:
        for filename in filenames:
            jobs.append(pool.apply_async(check_file, args=(filename,)))

        iter = jobs if quiet else tqdm(jobs, desc="Checking files")
        results = [job.get() for job in iter]

    return results
