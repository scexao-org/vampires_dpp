from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vampires_dpp.pipeline.config import PipelineConfig
from vampires_dpp.pipeline.pipeline import Pipeline

# opinionated style defaults :)
plt.rcParams.update({"figure.dpi": 300})


def _plot_flux(table, metrics, savepath: Path, name: str = ""):
    means_errs = np.array(
        [get_mean_and_err(metric["photf"], metric["phote"]) for metric in metrics]
    )

    xs = table["UT"]
    fig, ax = plt.subplots()
    if means_errs.shape[-1] == 1:
        labels = (table["FIELD"].unique()[0],)
    elif means_errs.shape[-1] == 3:
        labels = ("F610", "F670", "F720")
    else:
        labels = ("F610", "F670", "F720", "F760")

    for wlidx in range(means_errs.shape[-1]):
        ax.errorbar(
            xs,
            means_errs[:, 0, wlidx],
            yerr=means_errs[:, 1, wlidx],
            lw=0,
            marker=".",
            label=labels[wlidx],
        )

    ax.legend()

    # ax.xaxis.set_major_locator(HourLocator(interval=1))
    # ax.xaxis.set_major_formatter(AutoDateFormatter("%H:%M"))

    ax.set(xlabel="time (UT)", ylabel="flux (e-/s)")

    filename = savepath / f"{name}_flux.pdf"
    fig.savefig(filename)


def _plot_centroids(metrics, savepath: Path):
    ...


def _plot_strehl(table, metrics, savepath: Path, name: str = ""):
    means_errs = np.array([get_mean_and_err(metric["strehl"]) for metric in metrics])

    xs = table["UT"]
    fig, ax = plt.subplots()
    if means_errs.shape[-1] == 1:
        labels = (table["FIELD"].unique()[0],)
    elif means_errs.shape[-1] == 3:
        labels = ("F610", "F670", "F720")
    else:
        labels = ("F610", "F670", "F720", "F760")

    for wlidx in range(means_errs.shape[-1]):
        ax.scatter(xs, means_errs[:, 0, wlidx] * 100, marker=".", label=labels[wlidx])

    ax.legend()

    # ax.xaxis.set_major_locator(HourLocator(interval=1))
    # ax.xaxis.set_major_formatter(AutoDateFormatter("%H:%M"))

    ax.set(xlabel="time (UT)", ylabel="Strehl ratio (%)")

    filename = savepath / f"{name}_strehl.pdf"
    fig.savefig(filename)


def plot_metrics(metrics, savepath: Path):
    _plot_flux(metrics)


def load_metrics(filenames):
    metrics = []
    for file in filenames:
        metrics.append(np.load(file))
    return metrics


def get_mean_and_err(values, errors=None):
    mean_value = np.mean(values, axis=(-2, -1))
    N = np.prod(values.shape[-2:])
    error = np.std(values, axis=(-2, -1)) / np.sqrt(N)

    if errors is not None:
        rms = np.sqrt(np.sum(np.power(errors, 2), axis=(-2, -1))) / N
        error = np.hypot(error, rms)
    return mean_value, error


if __name__ == "__main__":
    _dir = Path("/Volumes/mlucasSSD1/workdir/20241219/HD34700/")
    file = _dir / "metrics" / "20241219_HD34700_vampires_080_cam2_FLCNA_metrics.npz"

    workdir = Path.cwd() / "tmp"
    workdir.mkdir(exist_ok=True)
    config = PipelineConfig.from_file(_dir / "20241219_HD34700_vampires.toml")
    pipeline = Pipeline(config, workdir, verbose=True)

    table = pd.read_csv(_dir / "aux" / f"{config.name}_table.csv")

    groups = table.groupby("U_CAMERA")
    cam1_table = groups.get_group(1)
    cam2_table = groups.get_group(2)

    cam1_filenames = (_dir / "metrics").glob("20241219*cam1*.npz")
    cam2_filenames = (_dir / "metrics").glob("20241219*cam2*.npz")
    cam1_metrics = load_metrics(cam1_filenames)
    cam2_metrics = load_metrics(cam2_filenames)

    _plot_flux(cam1_table, cam1_metrics, workdir, name=f"{config.name}_cam1")
    _plot_flux(cam2_table, cam2_metrics, workdir, name=f"{config.name}_cam2")

    _plot_strehl(cam1_table, cam1_metrics, workdir, name=f"{config.name}_cam1")
    _plot_strehl(cam2_table, cam2_metrics, workdir, name=f"{config.name}_cam2")
