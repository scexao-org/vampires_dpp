from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def make_nrm_plots(nrm_results: dict, outpath: Path, name: str):
    fig = _make_azimuth_v2_plots(nrm_results, name)
    filename = outpath / f"{name.replace('_', ' ')}_v2_azimuth.pdf"
    fig.savefig(filename)
    fig = _make_azimuth_cp_plots(nrm_results, name)
    filename = outpath / f"{name.replace('_', ' ')}_cp_azimuth.pdf"
    fig.savefig(filename)


def _make_azimuth_v2_plots(nrm_results, name: str):
    plt.rcParams.update({"axes.grid": True})
    fig, axes = plt.subplots(
        nrows=nrm_results["visibilities"].shape[1], ncols=2, sharex=True, sharey=True, squeeze=False
    )

    width = 6
    aspect_ratio = 1 / 1.616
    height = width * aspect_ratio

    fig.set_size_inches(width, height)

    bl_length = np.hypot(nrm_results["u"], nrm_results["v"])
    bl_azimuth_rad = np.arctan(nrm_results["v"] / nrm_results["u"])
    bl_norm = (bl_length - bl_length.min()) / np.ptp(bl_length)
    colors = plt.get_cmap("jet")(bl_norm)
    norm = mpl.colors.Normalize(vmin=bl_length.min(), vmax=bl_length.max())
    sm = mpl.cm.ScalarMappable(norm=norm, cmap="jet")
    for wl_idx in range(nrm_results["visibilities"].shape[1]):
        for idx in range(len(bl_azimuth_rad)):
            axes[wl_idx, 0].errorbar(
                bl_azimuth_rad[idx],
                nrm_results["visibilities"][2, wl_idx, idx],
                yerr=nrm_results["visibilities_err"][2, wl_idx, idx],
                c=colors[idx],
                marker="o",
            )
            axes[wl_idx, 1].errorbar(
                bl_azimuth_rad[idx],
                nrm_results["visibilities"][3, wl_idx, idx],
                yerr=nrm_results["visibilities_err"][3, wl_idx, idx],
                c=colors[idx],
                marker="o",
            )
    fig.colorbar(sm, label="baseline length (m)")  # , ax=axes, use_gridspec=True)
    axes[0, 0].set(title="Stokes Q")
    axes[0, 1].set(title="Stokes U")
    for ax in axes[-1]:
        ax.set(xlabel="azimuth (rad)")
    for i, ax in enumerate(axes[:, 0]):
        ax.set(ylabel=rf"Field {i} - $|\nu|^2$")

    fig.suptitle(f"{name.replace('_', ' ')}")
    # save file
    fig.tight_layout()
    return fig


def _make_azimuth_cp_plots(nrm_results, name: str):
    plt.rcParams.update({"axes.grid": True})
    fig, axes = plt.subplots(
        nrows=nrm_results["visibilities"].shape[1], ncols=2, sharex=True, sharey=True, squeeze=False
    )

    width = 6
    aspect_ratio = 1 / 1.616
    height = width * aspect_ratio

    fig.set_size_inches(width, height)

    bl_length = np.hypot(nrm_results["u"], nrm_results["v"])
    bl_azimuth_rad = np.arctan(nrm_results["v"] / nrm_results["u"])
    bl_norm = (bl_length - bl_length.min()) / np.ptp(bl_length)
    colors = plt.get_cmap("jet")(bl_norm)
    norm = mpl.colors.Normalize(vmin=bl_length.min(), vmax=bl_length.max())
    sm = mpl.cm.ScalarMappable(norm=norm, cmap="jet")
    for wl_idx in range(nrm_results["closure_phases"].shape[1]):
        for idx in range(len(bl_azimuth_rad)):
            axes[wl_idx, 0].errorbar(
                bl_azimuth_rad[idx],
                nrm_results["closure_phases"][2, wl_idx, idx],
                yerr=nrm_results["closure_phases_err"][2, wl_idx, idx],
                c=colors[idx],
                marker="o",
            )
            axes[wl_idx, 1].errorbar(
                bl_azimuth_rad[idx],
                nrm_results["closure_phases"][3, wl_idx, idx],
                yerr=nrm_results["closure_phases_err"][3, wl_idx, idx],
                c=colors[idx],
                marker="o",
            )
    fig.colorbar(sm, label="baseline length (m)")  # , ax=axes, use_gridspec=True)
    axes[0, 0].set(title="Stokes Q")
    axes[0, 1].set(title="Stokes U")
    for ax in axes[-1]:
        ax.set(xlabel="azimuth (rad)")
    for i, ax in enumerate(axes[:, 0]):
        ax.set(ylabel=rf"Field {i} - CP")

    fig.suptitle(f"{name.replace('_', ' ')}")
    # save file
    fig.tight_layout()
    return fig
