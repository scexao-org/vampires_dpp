import logging
from typing import Annotated, Literal, TypeAlias

import numpy as np
from annotated_types import Interval
from astropy.io import fits
from numpy.typing import NDArray

__all__ = ("frame_select_hdul", "FrameSelectMetric")

logger = logging.getLogger(__file__)

FrameSelectMetric: TypeAlias = Literal["max", "l2norm", "normvar"]


def frame_select_hdul(
    hdul: fits.HDUList,
    metrics,
    *,
    quantile: Annotated[float, Interval(ge=0, le=1)] = 0,
    metric: FrameSelectMetric = "normvar",
) -> tuple[fits.HDUList, dict[str, NDArray]]:
    if quantile == 0:
        logger.debug("Skipping frame selection because quantile was 0")
        return hdul

    values = metrics[metric]

    # determine cutoff value and create mask
    # only remove if ALL wavelengths and all PSFs fail
    # reminder, values have shape (nframes, nlambda, npsfs)
    cutoff = np.nanquantile(values, quantile, axis=0, keepdims=True)
    metrics_mask = np.any(values >= cutoff, axis=(1, 2), keepdims=True)
    # filter our metrics (so that in the future we can get the filtered centroids)
    for key in metrics:
        metrics[key] = metrics[key][metrics_mask]

    # now, filter data and error arrays using mask
    data_mask = np.squeeze(metrics_mask)
    hdul[0].data = hdul[0].data[data_mask]
    bunit = hdul[0].header.get("BUNIT", "")
    hdul["ERR"].data = hdul["ERR"].data[data_mask]

    # update header info
    info = fits.Header()
    info["hierarch DPP FRAME_SELECT METRIC"] = metric, "Frame selection metric"
    info["hierarch DPP FRAME_SELECT QUANTILE"] = quantile, "Frame selection exclusion quantile"
    info["hierarch DPP FRAME_SELECT CUTOFF"] = (
        cutoff,
        f"[{bunit}] cutoff value for frame selection metric",
    )

    for hdu_idx in range(len(hdul)):
        hdul[hdu_idx].header.update(info)

    return hdul, metrics
