import multiprocessing as mp
import shutil
from collections import OrderedDict
from os import PathLike
from pathlib import Path
from typing import List, Optional

import pandas as pd
from astropy.io import fits
from tqdm.auto import tqdm


def dict_from_header_file(filename: PathLike, **kwargs):
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
    summary = OrderedDict()
    # add path to row before the FITS header keys
    summary["path"] = Path(filename).resolve()
    header = fits.getheader(filename, **kwargs)
    summary.update(dict_from_header(header))
    return summary


def dict_from_header(header: fits.Header):
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
    filenames: List[PathLike],
    ext: int | str = 0,
    num_proc: int = min(8, mp.cpu_count()),
    quiet: bool = False,
) -> pd.DataFrame:
    """
    Generate a pandas dataframe from the FITS headers parsed from the given files.

    Parameters
    ----------
    filenames : List[pathlike]
    ext : int or str, optional
        FITS extension to parse from, by default 0
    num_proc : int, optional
        Number of processes to use in multiprocessing, by default mp.cpu_count()
    quiet : bool, optional
        Silence the progress bar, by default False

    Returns
    -------
    pandas.DataFrame
    """
    with mp.Pool(num_proc) as pool:
        kwds = dict(ext=ext)
        jobs = [pool.apply_async(dict_from_header_file, args=(f,), kwds=kwds) for f in filenames]
        iter = jobs if quiet else tqdm(jobs, desc="Parsing FITS headers")
        rows = [job.get() for job in iter]

    df = pd.DataFrame(rows)
    df.sort_values("MJD", inplace=True)
    return df


# set up commands for parser to dispatch to
def sort(
    filenames: List[PathLike],
    copy: bool = False,
    ext: int | str = 0,
    output_directory: Optional[PathLike] = None,
    num_proc=min(8, mp.cpu_count()),
):
    if output_directory is not None:
        outdir = Path(output_directory)
    else:
        outdir = Path(filenames[0]).parent
    jobs = []
    with mp.Pool(num_proc) as pool:
        for filename in filenames:
            kwds = dict(outdir=outdir, copy=copy)
            jobs.append(pool.apply_async(sort_file, args=(filename,), kwds=kwds))

        results = [job.get() for job in tqdm(jobs, desc="Sorting files")]

    return results


def sort_file(filename, outdir, copy=False, **kwargs):
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


def foldername_new(outdir, header, subdir="raw"):
    match header["DATA-TYP"]:
        case "OBJECT":
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


def foldername_old(outdir, path, header, subdir="raw"):
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
        foldname = outdir / header["OBJECT"].replace(" ", "_") / subdir

    return foldname
