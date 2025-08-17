import warnings

import h5py
import numpy as np
import pandas as pd
import tqdm.auto as tqdm
from munch import munchify as dict2class
from numpy.typing import NDArray

from vampires_dpp.pdi.processing import TRIPLEDIFF_SETS, triple_diff_dict


def generate_bootstrap_samples(data: NDArray) -> NDArray:
    rng = np.random.default_rng()
    # generate
    num_frames = data.shape[-2]
    idxs = rng.choice(np.arange(num_frames), size=num_frames)
    samples = np.mean(data[:, idxs], axis=1)
    return samples


def triple_quotient_dict(
    input_dict: dict[tuple[float, str, int], NDArray]
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    ## make difference images
    # single diff (cams)
    pQ0 = input_dict[(0.0, "A", 1)] / input_dict[(0.0, "A", 2)]
    pIQ0 = input_dict[(0.0, "A", 1)] * input_dict[(0.0, "A", 2)]
    pQ1 = input_dict[(0.0, "B", 1)] / input_dict[(0.0, "B", 2)]
    pIQ1 = input_dict[(0.0, "B", 1)] * input_dict[(0.0, "B", 2)]

    mQ0 = input_dict[(45.0, "A", 1)] / input_dict[(45.0, "A", 2)]
    mIQ0 = input_dict[(45.0, "A", 1)] * input_dict[(45.0, "A", 2)]
    mQ1 = input_dict[(45.0, "B", 1)] / input_dict[(45.0, "B", 2)]
    mIQ1 = input_dict[(45.0, "B", 1)] * input_dict[(45.0, "B", 2)]

    pU0 = input_dict[(22.5, "A", 1)] / input_dict[(22.5, "A", 2)]
    pIU0 = input_dict[(22.5, "A", 1)] * input_dict[(22.5, "A", 2)]
    pU1 = input_dict[(22.5, "B", 1)] / input_dict[(22.5, "B", 2)]
    pIU1 = input_dict[(22.5, "B", 1)] * input_dict[(22.5, "B", 2)]

    mU0 = input_dict[(67.5, "A", 1)] / input_dict[(67.5, "A", 2)]
    mIU0 = input_dict[(67.5, "A", 1)] * input_dict[(67.5, "A", 2)]
    mU1 = input_dict[(67.5, "B", 1)] / input_dict[(67.5, "B", 2)]
    mIU1 = input_dict[(67.5, "B", 1)] * input_dict[(67.5, "B", 2)]

    # double difference (FLC1 / FLC2)
    pQ = np.sqrt(pQ0 / pQ1)
    pIQ = np.sqrt(pIQ0 * pIQ1)

    mQ = np.sqrt(mQ0 / mQ1)
    mIQ = np.sqrt(mIQ0 * mIQ1)

    pU = np.sqrt(pU0 / pU1)
    pIU = np.sqrt(pIU0 * pIU1)

    mU = np.sqrt(mU0 / mU1)
    mIU = np.sqrt(mIU0 * mIU1)

    # triple difference (HWP1 / HWP2)
    Q = np.sqrt(pQ / mQ)
    IQ = np.sqrt(pIQ * mIQ)
    U = np.sqrt(pU / mU)
    IU = np.sqrt(pIU * mIU)

    return IQ, IU, Q, U


def process_nrm_polarimetry(table: pd.DataFrame, nbootstrap: int = 1000):
    nfields = len(table["nrm_paths"][0])

    # data_sets will be a dict where each key is a pol state
    # and each value will be an (nfields, nframes, nvalues) array
    visibilities_dict = {}
    closure_phases_dict = {}
    for key in TRIPLEDIFF_SETS:
        # key: HWP, FLC, CAM
        mask = (
            (table["RET-ANG1"] == key[0])
            & (table["U_FLC"] == key[1])
            & (table["U_CAMERA"] == key[2])
        )
        subset = table.loc[mask]
        if len(subset) == 0:
            msg = f"Could not find a subset with HWP, FLC, CAM state {key}"
            warnings.warn(msg, stacklevel=2)
            continue

        visibilities = []
        closure_phases = []
        for field_idx in range(nfields):
            paths = [path[field_idx] for path in subset["nrm_paths"]]
            us, vs, vis, cp = load_observables(paths)
            visibilities.append(vis)
            closure_phases.append(cp)
        visibilities_dict[key] = np.array(visibilities)
        closure_phases_dict[key] = np.array(closure_phases)

    # use for loops to be able to nest the means, avoiding n^2 storage complexity
    _vis_results = []
    _cp_results = []
    for _ in tqdm.trange(nbootstrap, desc="Boostrapping PDI"):
        _vis_dict = {k: generate_bootstrap_samples(v) for k, v in visibilities_dict.items()}
        _cp_dict = {k: generate_bootstrap_samples(v) for k, v in closure_phases_dict.items()}
        _vis_res = triple_quotient_dict(_vis_dict)
        _cp_res = triple_diff_dict(_cp_dict)
        _vis_results.append(_vis_res)
        _cp_results.append(_cp_res)
    # so each array in *_dict has (nstokes, nbootstrap, nfields, nvalues)
    visibilities_mean = np.mean(_vis_results, axis=0)
    visibilities_std = np.std(_vis_results, axis=0)
    closure_phases_mean = np.mean(_cp_results, axis=0)
    closure_phases_std = np.std(_cp_results, axis=0)
    return {
        "u": us,
        "v": vs,
        "baselines": np.hypot(us, vs),
        "azimuth": np.arctan(vs / us),
        "visibilities_samples": np.array(_vis_results),
        "visibilities": visibilities_mean,
        "visibilities_err": visibilities_std,
        "closure_phases_samples": np.array(_cp_results),
        "closure_phases": closure_phases_mean,
        "closure_phases_err": closure_phases_std,
    }


def load_observables(path_list):
    amical_objs = [load_bs_hdf5(f) for f in path_list]
    us = np.median([ami.u for ami in amical_objs], axis=0)
    vs = np.median([ami.v for ami in amical_objs], axis=0)
    visibilities = np.concatenate([ami.matrix["v2_arr"] for ami in amical_objs])
    closure_phases = np.concatenate([ami.matrix["cp_arr"] for ami in amical_objs])

    return us, vs, visibilities, closure_phases


# yeeted out of the amical codebase because it can't be changed I guess?
def load_bs_hdf5(filename):
    """Load hdf5 file and format as class like object (same
    format as `amical.extract_bs()`
    """

    hf2 = h5py.File(filename, "r")
    dict_bs = {"matrix": {}, "infos": {"hdr": {}}, "mask": {}}
    obs = hf2["obs"]

    for o in obs:
        dict_bs[o] = obs[o][()]

    matrix = hf2["matrix"]
    for key in matrix:
        dict_bs["matrix"][key] = matrix[key][()]

    if len(dict_bs["matrix"]["cp_cov"]) == 1:
        dict_bs["matrix"]["cp_cov"] = None

    mask = hf2["mask"]
    for key in mask:
        if key not in ["u1", "u2", "v1", "v2"]:
            dict_bs["mask"][key] = mask[key][()]

    t3_coord = {
        "u1": mask["u1"][()],
        "u2": mask["u2"][()],
        "v1": mask["v1"][()],
        "v2": mask["v2"][()],
    }

    dict_bs["mask"]["t3_coord"] = t3_coord

    infos = hf2["infos"]
    for key in infos:
        dict_bs["infos"][key] = infos[key][()]

    hdr = hf2["hdr"]
    for key in hdr:
        dict_bs["infos"]["hdr"][key] = hdr[key][()]

    bs_save = dict2class(dict_bs)
    return bs_save
