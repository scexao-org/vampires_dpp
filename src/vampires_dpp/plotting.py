from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import time
from matplotlib import dates

from vampires_dpp.pipeline.config import PipelineConfig
from vampires_dpp.pipeline.pipeline import Pipeline

# opinionated style defaults :)
plt.rcParams.update({"figure.dpi": 300, "axes.grid": True})


def get_color(filter_name: str):
    color_list = ["#E7884C", "#D44E3f", "#9A3251", "#5E2140"]
    match filter_name:
        case "F610":
            color = color_list[0]
        case "F670":
            color = color_list[1]
        case "F720":
            color = color_list[2]
        case "F760":
            color = color_list[3]
        case "625-50":
            color = color_list[0]
        case "675-50":
            color = color_list[1]
        case "725-50" | "750-50":
            color = color_list[2]
        case "775-50":
            color = color_list[3]
        case "Halpha":
            color_list[1]
        case "Ha-Cont":
            color_list[2]
        case "SII":
            color_list[2]
        case "SII-Cont":
            color_list[3]
        case _:
            color_list[2]
    return color


def _plot_flux(table, metrics, savepath: Path, name: str = ""):
    means_errs = np.array(
        [get_mean_and_err(metric["photf"], metric["phote"]) for metric in metrics]
    )
    times = time.Time(table["MJD"], format="mjd").to_datetime()

    width = 6
    aspect_ratio = 1.2 / 1.616
    height = width * aspect_ratio
    fig, ax = plt.subplots()
    fig.set_size_inches(width, height)

    if means_errs.shape[-1] == 1:
        labels = (table["FILTER01"].unique()[0],)
    elif means_errs.shape[-1] == 3:
        labels = ("F610", "F670", "F720")
    else:
        labels = ("F610", "F670", "F720", "F760")

    for wlidx in range(means_errs.shape[-1]):
        color = get_color(labels[wlidx])
        ax.errorbar(
            times,
            means_errs[:, 0, wlidx],
            yerr=means_errs[:, 1, wlidx],
            linewidth=0,
            marker=".",
            color=color,
            label=labels[wlidx],
            zorder=4,
        )
    ax.legend()

    ax.set(
        xlabel="time (UT)",
        ylabel="instrumental flux (e-/s)",
        title=f"{name.replace('_', ' ')} flux",
    )

    fmt = dates.AutoDateFormatter(ax.xaxis.get_major_locator(), defaultfmt="%H:%M:%S")
    ax.xaxis.set_major_formatter(fmt)
    ax.tick_params(axis="x", labelrotation=45)

    # secondary y-axis in magnitude
    def transform(flux):
        return -2.5 * np.log10(flux)

    def inv_transform(mag):
        return 10 ** (-0.4 * mag)

    ax2 = ax.secondary_yaxis(location="right", functions=(transform, inv_transform))
    ax2.set(ylabel="instrumental mag")

    # save file
    filename = savepath / f"{name}_flux.pdf"
    fig.tight_layout()
    fig.savefig(filename)


def _plot_centroids(table, metrics, key, savepath: Path, name: str = ""):
    x_means_errs = np.array([get_mean_and_err(metric[f"{key}x"]) for metric in metrics])
    y_means_errs = np.array([get_mean_and_err(metric[f"{key}y"]) for metric in metrics])

    width = 6
    aspect_ratio = 1.2 / 1.616
    height = width * aspect_ratio
    fig, ax = plt.subplots()
    fig.set_size_inches(width, height)
    if x_means_errs.shape[-1] == 1:
        labels = (table["FILTER01"].unique()[0],)
    elif x_means_errs.shape[-1] == 3:
        labels = ("F610", "F670", "F720")
    else:
        labels = ("F610", "F670", "F720", "F760")

    for wlidx in range(x_means_errs.shape[-1]):
        color = get_color(labels[wlidx])
        ax.scatter(
            x_means_errs[:, 0, wlidx] - x_means_errs[:, 0, wlidx].mean(),
            y_means_errs[:, 0, wlidx] - y_means_errs[:, 0, wlidx].mean(),
            marker=".",
            label=labels[wlidx],
            color=color,
            zorder=4,
        )
    ax.axhline(0, color="0.2", zorder=3)
    ax.axvline(0, color="0.2", zorder=3)

    ax.legend()

    ax.set(
        xlabel=r"$\Delta$x (pix)",
        ylabel=r"$\Delta$y (pix)",
        title=f"{name.replace('_', ' ')} {key} centroid",
    )

    # save file
    filename = savepath / f"{name}_centroid_scatter.pdf"
    fig.tight_layout()
    fig.savefig(filename)


def _plot_centroids_over_time(table, metrics, key, savepath: Path, name: str = ""):
    x_means_errs = np.array([get_mean_and_err(metric[f"{key}x"]) for metric in metrics])
    y_means_errs = np.array([get_mean_and_err(metric[f"{key}y"]) for metric in metrics])
    delta_x = x_means_errs - x_means_errs.mean(axis=0, keepdims=True)
    delta_y = y_means_errs - y_means_errs.mean(axis=0, keepdims=True)
    dist_xy = np.hypot(delta_y, delta_x)
    angle_xy = np.rad2deg(np.arctan2(delta_y, delta_x))
    angle_xy[angle_xy < 0] += 360

    times = time.Time(table["MJD"], format="mjd").to_datetime()

    width = 6
    nrows = 4
    aspect_ratio = nrows / 1.616
    height = width * aspect_ratio
    fig, axes = plt.subplots(nrows=nrows, sharex=True)
    fig.set_size_inches(width, height)

    if x_means_errs.shape[-1] == 1:
        labels = (table["FILTER01"].unique()[0],)
    elif x_means_errs.shape[-1] == 3:
        labels = ("F610", "F670", "F720")
    else:
        labels = ("F610", "F670", "F720", "F760")

    for wlidx in range(x_means_errs.shape[-1]):
        color = get_color(labels[wlidx])
        axes[0].scatter(
            times, delta_x[:, 0, wlidx], marker=".", label=labels[wlidx], color=color, zorder=4
        )
        axes[1].scatter(
            times, delta_y[:, 0, wlidx], marker=".", label=labels[wlidx], color=color, zorder=4
        )
        axes[2].scatter(
            times, dist_xy[:, 0, wlidx], marker=".", label=labels[wlidx], color=color, zorder=4
        )
        axes[3].scatter(
            times, angle_xy[:, 0, wlidx], marker=".", label=labels[wlidx], color=color, zorder=4
        )
    for ax in axes:
        fmt = dates.AutoDateFormatter(ax.xaxis.get_major_locator(), defaultfmt="%H:%M:%S")
        ax.xaxis.set_major_formatter(fmt)
        ax.tick_params(axis="x", labelrotation=45)

    axes[0].legend()

    axes[0].axhline(0, color="0.2", zorder=3)
    axes[1].axhline(0, color="0.2", zorder=3)
    axes[3].axhline(0, color="0.2", zorder=3)

    axes[0].set(ylabel=r"$\Delta$x (pix)")
    axes[1].set(ylabel=r"$\Delta$y (pix)")
    axes[2].set(ylabel=r"distance (pix)")
    axes[3].set(ylabel=r"angle (deg)", ylim=(0, 360))

    axes[-1].set(xlabel="time (UT)")

    fig.suptitle(f"{name.replace('_', ' ')} {key} centroid")

    # save file
    filename = savepath / f"{name}_centroid.pdf"
    fig.tight_layout()
    fig.savefig(filename)


def _plot_strehl(table, metrics, savepath: Path, name: str = ""):
    means_errs = np.array([get_mean_and_err(metric["strehl"]) for metric in metrics])
    times = time.Time(table["MJD"], format="mjd").to_datetime()
    width = 6
    aspect_ratio = 1.2 / 1.616
    height = width * aspect_ratio
    fig, ax = plt.subplots()
    fig.set_size_inches(width, height)
    if means_errs.shape[-1] == 1:
        labels = (table["FILTER01"].unique()[0],)
    elif means_errs.shape[-1] == 3:
        labels = ("F610", "F670", "F720")
    else:
        labels = ("F610", "F670", "F720", "F760")

    for wlidx in range(means_errs.shape[-1]):
        color = get_color(labels[wlidx])
        ax.scatter(
            times,
            means_errs[:, 0, wlidx] * 100,
            marker=".",
            label=labels[wlidx],
            color=color,
            zorder=4,
        )

    ax.legend()

    ax.set(
        xlabel="time (UT)",
        ylabel="Strehl ratio (%)",
        ylim=(0, 100),
        title=f"{name.replace('_', ' ')} strehl",
    )

    fmt = dates.AutoDateFormatter(ax.xaxis.get_major_locator(), defaultfmt="%H:%M:%S")
    ax.xaxis.set_major_formatter(fmt)
    ax.tick_params(axis="x", labelrotation=45)

    # save file
    filename = savepath / f"{name}_strehl.pdf"
    fig.tight_layout()
    fig.savefig(filename)


def load_metrics(filenames):
    metrics = []
    for file in filenames:
        metrics.append(np.load(file))
    return metrics


def get_mean_and_err(values, errors=None):
    # dimensions are (wl_idx, psf_idx, frame_idx)
    mean_value = np.nanmean(values, axis=(-2, -1))
    N = np.prod(values.shape[-2:])
    error = np.nanstd(values, axis=(-2, -1)) / np.sqrt(N)

    if errors is not None:
        rms = np.sqrt(np.nansum(np.power(errors, 2), axis=(-2, -1))) / N
        error = np.hypot(error, rms)
    return mean_value, error


if __name__ == "__main__":
    # _dir = Path("/Volumes/mlucasSSD1/workdir/20241219/HD34700/")
    # file = _dir / "metrics" / "20241219_HD34700_vampires_080_cam2_FLCNA_metrics.npz"

    # workdir = Path.cwd() / "tmp"
    # workdir.mkdir(exist_ok=True)
    # config = PipelineConfig.from_file(_dir / "20241219_HD34700_vampires.toml")

    _dir = Path("/Volumes/mlucasSSD1/workdir/20230101/vampires/ABAUR/")

    workdir = Path.cwd() / "tmp"
    workdir.mkdir(exist_ok=True)
    config = PipelineConfig.from_file(_dir / "20230101_ABAur_vampires.toml")

    pipeline = Pipeline(config, workdir, verbose=True)

    table = pd.read_csv(_dir / "aux" / f"{config.name}_table.csv")

    groups = table.groupby("U_CAMERA")
    cam1_table = groups.get_group(1)
    cam2_table = groups.get_group(2)

    # cam1_filenames = (_dir / "metrics").glob("20241219*cam1*.npz")
    # cam2_filenames = (_dir / "metrics").glob("20241219*cam2*.npz")
    cam1_filenames = (_dir / "metrics").glob("20230101*cam1*.npz")
    cam2_filenames = (_dir / "metrics").glob("20230101*cam2*.npz")

    cam1_metrics = load_metrics(cam1_filenames)
    cam2_metrics = load_metrics(cam2_filenames)

    _plot_flux(cam1_table, cam1_metrics, workdir, name=f"{config.name}_cam1")
    _plot_flux(cam2_table, cam2_metrics, workdir, name=f"{config.name}_cam2")

    _plot_centroids(cam1_table, cam1_metrics, "dft", workdir, name=f"{config.name}_cam1")
    _plot_centroids(cam2_table, cam2_metrics, "dft", workdir, name=f"{config.name}_cam2")

    _plot_centroids_over_time(cam1_table, cam1_metrics, "dft", workdir, name=f"{config.name}_cam1")
    _plot_centroids_over_time(cam2_table, cam2_metrics, "dft", workdir, name=f"{config.name}_cam2")

    _plot_strehl(cam1_table, cam1_metrics, workdir, name=f"{config.name}_cam1")
    _plot_strehl(cam2_table, cam2_metrics, workdir, name=f"{config.name}_cam2")
