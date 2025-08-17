from pathlib import Path

import amical
import numpy as np
from astropy.io import fits

from vampires_dpp.nrm.params import get_amical_parameters
from vampires_dpp.nrm.windowing import window_cube
from vampires_dpp.specphot.filters import determine_filterset_from_header


def extract_observables(config, input_filename, output_path: Path, force: bool = False):
    """Runs AMICAL and extracts observables to HDF5 file. Will skip if file already exists and force is False"""
    if not force and output_path.exists():
        return output_path
    with fits.open(input_filename) as hdul:
        cube = hdul[0].data
        header = hdul[0].header
    params = get_amical_parameters(header)
    fields = determine_filterset_from_header(header)
    paths = []
    for wl_idx in range(cube.shape[1]):
        real_output_path = output_path.with_name(
            output_path.name.replace("vis", f"{fields[wl_idx]}_vis")
        )
        paths.append(real_output_path)
        if not force and real_output_path.exists():
            continue

        data, header = window_cube(np.nan_to_num(cube[:, wl_idx]), size=80, header=header)

        observables = amical.extract_bs(
            data,
            str(input_filename),
            targetname=config.target.name,
            display=False,
            compute_cp_cov=False,
            theta_detector=config.nrm.theta,
            scaling_uv=config.nrm.uv,
            savepath=False,
            **params,
        )
        amical.save_bs_hdf5(observables, str(real_output_path))
    return paths
