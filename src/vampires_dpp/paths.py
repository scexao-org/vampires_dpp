import re
from pathlib import Path

from vampires_dpp.specphot.filters import determine_filterset_from_header


class Paths:
    def __init__(self, workdir=None):
        self.workdir = Path.cwd() if workdir is None else Path(workdir)

    @property
    def aux(self) -> Path:
        return self.workdir / "aux"

    @property
    def calibrated(self) -> Path:
        return self.workdir / "calibrated"

    @property
    def combined(self) -> Path:
        return self.workdir / "combined"

    @property
    def metrics(self) -> Path:
        return self.workdir / "metrics"

    @property
    def selected(self) -> Path:
        return self.workdir / "selected"

    @property
    def aligned(self) -> Path:
        return self.workdir / "aligned"

    @property
    def coadded(self) -> Path:
        return self.workdir / "coadded"

    @property
    def nrm(self) -> Path:
        return self.workdir / "nrm"

    @property
    def diff(self) -> Path:
        return self.workdir / "diff"

    @property
    def adi(self) -> Path:
        return self.workdir / "adi"

    @property
    def pdi(self) -> Path:
        return self.workdir / "pdi"

    @property
    def mm(self) -> Path:
        return self.pdi / "mm"

    @property
    def stokes(self) -> Path:
        return self.pdi / "stokes"


def make_dirs(paths, config):
    paths.aux.mkdir(parents=True, exist_ok=True)
    paths.metrics.mkdir(parents=True, exist_ok=True)
    # intermediate calib data
    if config.calibrate.save_intermediate:
        paths.calibrated.mkdir(parents=True, exist_ok=True)
    # intermediate combined data
    if config.combine.save_intermediate:
        paths.combined.mkdir(parents=True, exist_ok=True)
    # intermediate selected data
    if config.frame_select.save_intermediate:
        paths.selected.mkdir(parents=True, exist_ok=True)
    # intermediate registered, or we are not coadding
    # (i.e., registered data is final product)
    if config.align.save_intermediate:
        paths.aligned.mkdir(parents=True, exist_ok=True)
    # final collapsed data
    if config.coadd.coadd:
        paths.coadded.mkdir(parents=True, exist_ok=True)
    if config.diff_images.make_diff:
        paths.diff.mkdir(parents=True, exist_ok=True)
    if config.save_adi_cubes:
        paths.adi.mkdir(parents=True, exist_ok=True)
    if config.polarimetry is not None:
        paths.pdi.mkdir(parents=True, exist_ok=True)
        if config.polarimetry.mm_correct:
            paths.mm.mkdir(parents=True, exist_ok=True)
        paths.stokes.mkdir(parents=True, exist_ok=True)
    if config.nrm is not None:
        paths.nrm.mkdir(parents=True, exist_ok=True)


def get_paths(
    filename, /, suffix=None, outname=None, output_directory=None, filetype=".fits", **kwargs
):
    path = Path(filename)
    _suffix = "" if suffix is None else f"_{suffix}"
    if output_directory is None:
        output_directory = path.parent
    else:
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
    if outname is None:
        outname = re.sub(r"\.fits(\..*)?", f"{_suffix}{filetype}", path.name)
    outpath = output_directory / outname
    return path, outpath


def any_file_newer(filenames, outpath):
    out_mt = Path(outpath).stat().st_mtime
    # check if input is only a single file
    if isinstance(filenames, Path | str):
        return Path(filenames).stat().st_mtime > out_mt
    else:
        return any(Path(f).stat().st_mtime > out_mt for f in filenames)


def get_reduced_path(paths: Paths, config, group_key: str) -> Path:
    if config.coadd.coadd:
        base = paths.coadded
        suffix = "coll"
    elif config.align.save_intermediate:
        base = paths.aligned
        suffix = "reg"
    elif config.frame_select.save_intermediate:
        base = paths.selected
        suffix = "fs"
    elif config.combine.save_intermediate:
        base = paths.combined
        suffix = "comb"
    return base / f"{config.name}_{group_key}_{suffix}.fits"


def get_nrm_paths(output_path, header):
    fields = determine_filterset_from_header(header)
    paths = []
    for field in fields:
        real_output_path = output_path.with_name(output_path.name.replace("vis", f"{field}_vis"))
        paths.append(real_output_path)
    return paths
