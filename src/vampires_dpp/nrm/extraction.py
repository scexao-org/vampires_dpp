from pathlib import Path

import amical
from astropy.io import fits

from vampires_dpp.nrm.params import get_amical_parameters


def extract_observables(config, input_filename, uv, theta, output_path: Path, force: bool = False):
    """Runs AMICAL and extracts observables to HDF5 file. Will skip if file already exists and force is False"""
    if not force and output_path.exists():
        return output_path
    with fits.open(input_filename) as hdul:
        data = hdul[0].data
        header = hdul[0].header
    params = get_amical_parameters(header)

    observables = amical.extract_bs(
        data,
        input_filename,
        targetname=config.target.name,
        display=False,
        compute_cp_cov=False,
        theta_detector=theta,
        scaling_uv=uv,
        savepath=False,
        **params,
    )

    amical.save_bs_hdf5(observables, output_path)
    return output_path
