import logging
from typing import Annotated, Final, Literal, TypeAlias

import numpy as np
from annotated_types import Interval
from astropy.io import fits
from numpy.typing import NDArray

__all__ = ("frame_select_hdul", "FrameSelectMetric")

logger = logging.getLogger(__file__)

FrameSelectMetric: TypeAlias = Literal["max", "l2norm", "normvar"]


FRAME_SELECT_MAP: Final = {"peak": "max", "normvar": "nvar", "l2norm": "var"}


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

    # create masked metrics
    values = metrics[FRAME_SELECT_MAP[metric]]

    # determine cutoff value and create mask
    # only remove if ALL wavelengths and all PSFs fail
    # reminder, values have shape (nlambda, npsfs, nframes)
    cutoff = np.nanquantile(values, quantile, axis=2, keepdims=True)
    metrics_mask = np.any(values >= cutoff, axis=(0, 1))
    # filter our metrics (so that in the future we can get the filtered centroids)
    output_metrics = {}
    for key in metrics:
        output_metrics[key] = metrics[key][..., metrics_mask]

    # now, filter data and error arrays using mask
    hdul[0].data = hdul[0].data[metrics_mask]
    bunit = hdul[0].header.get("BUNIT", "")
    hdul["ERR"].data = hdul["ERR"].data[metrics_mask]

    # update header info
    info = fits.Header()
    info["hierarch DPP FRAME_SELECT METRIC"] = metric, "Frame selection metric"
    info["hierarch DPP FRAME_SELECT QUANTILE"] = quantile, "Frame selection quantile"
    info["hierarch DPP FRAME_SELECT CUTOFF"] = (
        np.median(cutoff),
        f"[{bunit}] median frame selection cutoff value",
    )

    for hdu in hdul:
        hdu.header.update(info)

    return hdul, output_metrics
