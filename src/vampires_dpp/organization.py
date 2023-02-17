import multiprocessing as mp
import shutil
from collections import Ordereddict
from os import PathLike
from pathlib import Path
from typing import Optional

import pandas as pd
from astropy.io import fits
from tqdm.auto import tqdm


def dict_from_header_file(filename: PathLike, **kwargs) -> Ordereddict:
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
    Ordereddict
    """
    path = Path(filename)
    summary = Ordereddict()
    # add path to row before the FITS header keys
    summary["path"] = path.resolve()
    ext = 1 if ".fits.fz" in path.name else 0
    header = fits.getheader(filename, ext=ext, **kwargs)
    summary.update(dict_from_header(header))
    return summary


def dict_from_header(header: fits.Header) -> Ordereddict:
    """
    Parse a FITS header and extract the keys and values as an ordered dictionary. Multi-line keys like ``COMMENTS`` and ``HISTORY`` will be combined with commas. The resolved path will be inserted with the ``path`` key.

    Parameters
    ----------
    header : Header
        FITS header to parse

    Returns
    -------
    Ordereddict
    """
    summary = Ordereddict()
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

    # data pre 2023/02/02 does not store DATA-TYP
    # meaningfully, so use ad-hoc sorting method
    if header["DATA-TYP"] == "ACQUISITION":
        foldname = foldername_old(outdir, path, header)
    else:
        foldname = foldername_new(outdir, header)

    newname = foldname / path.name
    foldname.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy(path, newname)
    else:
        path.replace(newname)
    return newname


def foldername_new(outdir: PathLike, header: fits.Header, subdir: PathLike = "raw"):
    match header["DATA-TYP"]:
        case "OBJECT":
            # subsort based on filter, EM gain, and exposure time
            filt = header["U_FILTER"]
            gain = header["U_EMGAIN"]
            exptime = header["U_AQTINT"] / 1e3  # ms
            subdir = f"{filt}_em{gain:.0f}_{exptime:05.0f}ms"
            foldname = outdir / header["OBJECT"].replace(" ", "_") / subdir
        case "DARK":
            foldname = outdir / "darks" / subdir
        # put sky flats separately because they are usually
        # background frames, not flats
        case "SKYFLAT":
            foldname = outdir / "skies" / subdir
        case "FLAT" | "DOMEFLAT":
            foldname = outdir / "flats" / subdir
        case "COMPARISON":
            foldname = outdir / "pinholes" / subdir
        case _:
            foldname = outdir

    return foldname


def foldername_old(outdir: PathLike, path: Path, header: fits.Header, subdir: PathLike = "raw"):
    name = header.get("U_OGFNAM", path.name)
    if "dark" in name:
        foldname = outdir / "darks" / subdir
    elif "skies" in name or "sky" in name:
        foldname = outdir / "skies" / subdir
    elif "flat" in name:
        foldname = outdir / "flats" / subdir
    elif "pinhole" in name:
        foldname = outdir / "pinholes" / subdir
    else:
        # subsort based on filter, EM gain, and exposure time
        filt = header["U_FILTER"]
        gain = header["U_EMGAIN"]
        exptime = header["U_AQTINT"] / 1e3  # ms
        subdir = f"{filt}_em{gain:.0f}_{exptime:05.0f}ms"
        foldname = outdir / header["OBJECT"].replace(" ", "_") / subdir

    return foldname
