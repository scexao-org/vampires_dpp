import re
from pathlib import Path


class Paths:
    def __init__(self, workdir=Path.cwd()):
        self.workdir = Path(workdir)

    @property
    def preproc_dir(self) -> Path:
        return self.workdir / "preproc"

    @property
    def products_dir(self) -> Path:
        return self.workdir / "products"

    @property
    def calibrated_dir(self) -> Path:
        return self.workdir / "calibrated"

    @property
    def metrics_dir(self) -> Path:
        return self.workdir / "metrics"

    @property
    def collapsed_dir(self) -> Path:
        return self.workdir / "collapsed"

    @property
    def diff_dir(self) -> Path:
        return self.workdir / "diff"

    @property
    def adi_dir(self) -> Path:
        return self.products_dir / "adi"

    @property
    def pdi_dir(self) -> Path:
        return self.workdir / "pdi"

    @property
    def mm_dir(self) -> Path:
        return self.pdi_dir / "mm"

    @property
    def stokes_dir(self) -> Path:
        return self.pdi_dir / "stokes"

def make_dirs(paths, config):
    paths.preproc_dir.mkdir(parents=True, exist_ok=True)
    paths.products_dir.mkdir(parents=True, exist_ok=True)
    paths.metrics_dir.mkdir(parents=True, exist_ok=True)
    if config.calibrate.save_intermediate:
        paths.calib_dir.mkdir(parents=True, exist_ok=True)
    if config.collapse:
        paths.collapsed_dir.mkdir(parents=True, exist_ok=True)
    if config.make_diff_images:
        paths.diff_dir.mkdir(parents=True, exist_ok=True)
    if config.save_adi_cubes:
        paths.adi_dir.mkdir(parents=True, exist_ok=True)
    if config.polarimetry:
        paths.pdi_dir.mkdir(parents=True, exist_ok=True)
        paths.mm_dir.mkdir(parents=True, exist_ok=True)
        paths.stokes_dir.mkdir(parents=True, exist_ok=True)

def get_paths(filename, /, suffix=None, outname=None, output_directory=None, filetype=".fits", **kwargs):
    path = Path(filename)
    _suffix = "" if suffix is None else f"_{suffix}"
    if output_directory is None:
        output_directory = path.parent
    else:
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
    if outname is None:
        outname = re.sub("\.fits(\..*)?", f"{_suffix}{filetype}", path.name)
    outpath = output_directory / outname
    return path, outpath


def any_file_newer(filenames, outpath):
    out_mt = Path(outpath).stat().st_mtime
    # check if input is only a single file
    if isinstance(filenames, Path) or isinstance(filenames, str):
        return Path(filenames).stat().st_mtime > out_mt
    else:
        gen = (Path(f).stat().st_mtime > out_mt for f in filenames)
        return any(gen)
