from typing import Any

from astropy.io import fits


def get_amical_parameters(header: fits.Header) -> dict[str, Any]:
    params = {}
    match header["U_MASK"]:
        case "SAM-18" | "SAM-18-Nudged":
            params["maskname"] = "g18"
            params["fw_splodge"] = 0.7
            params["hole_diam"] = 0.162
            params["cutoff"] = 8e-2
            params["n_wl"] = 3

    # filter is always 750-50 because uv scaling is done manually
    params["filtname"] = "750-50"
    # data is already flipped in calibration, but still leave True
    params["fliplr"] = True
    params["instrum"] = "VAMPIRES"
    return params
