import tempfile
from pathlib import Path

import amical
import numpy as np

from vampires_dpp.nrm.params import get_amical_parameters
from vampires_dpp.nrm.windowing import window_cube
from vampires_dpp.specphot.filters import determine_filterset_from_header


def extract_observables(
    config, input_hdul, output_path: Path, uv_thetas: dict, force: bool = False
):
    """Runs AMICAL and extracts observables to HDF5 file. Will skip if file already exists and force is False"""
    if not force and output_path.exists():
        return output_path

    cube = input_hdul[0].data
    header = input_hdul[0].header
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
        field = fields[wl_idx]
        with tempfile.NamedTemporaryFile(delete=True) as tmpfile:
            tmp_path = Path(tmpfile.name)
            input_hdul.writeto(tmp_path)
            observables = amical.extract_bs(
                data,
                str(tmp_path),
                targetname=config.name,
                display=False,
                compute_cp_cov=False,
                theta_detector=uv_thetas[field]["theta"],
                scaling_uv=uv_thetas[field]["uv"],
                savepath=False,
                **params,
            )
            amical.save_bs_hdf5(observables, str(real_output_path))
    return paths

    print(f"Temporary file: {tmp_path}")

    # Use tmp_path as a Path object
    tmp_path.write_text("Hello from a temp file!")


# The file will persist after context exit if delete=False
