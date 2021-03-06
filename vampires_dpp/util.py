import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import circmean


def flc_inds(flc_states: ArrayLike, n=4):
    """
    Find consistent runs of FLC states.

    A consistent FLC run will have either 2 or 4 files per HWP state, and will have exactly 4 HWP states per cycle. Sometimes when VAMPIRES is syncing with CHARIS a HWP state will get skipped, creating partial HWP cycles. This function will return the indices which create consistent HWP cycles from the given list of FLC states, which should already be sorted by time.

    Parameters
    ----------
    flc_states : ArrayLike
        The FLC states to sort through
    n : int, optional
        The number of files per HWP state, either 2 or 4. By default 4

    Returns
    -------
    inds :
        The indices for which `flc_states` forms consistent HWP cycles
    """
    states = np.asarray(flc_states)
    N_cycle = n * 4
    state_list = np.repeat([1, 2, 3, 4], n)
    inds = []
    idx = 0
    while idx <= len(flc_states) - N_cycle:
        if np.all(states[idx : idx + N_cycle] == state_list):
            inds.extend(range(idx, idx + N_cycle))
            idx += N_cycle
        else:
            idx += 1

    return inds


def average_angle(angles: ArrayLike):
    """
    Return the circular mean of the given angles in degrees.

    Parameters
    ----------
    angles : ArrayLike
        Angles in degrees, between [180, -180]

    Returns
    -------
    average_angle
        The average angle in degrees via the circular mean
    """
    rads = np.deg2rad(angles)
    radmean = circmean(rads, high=np.pi, low=-np.pi)
    return np.rad2deg(radmean)
