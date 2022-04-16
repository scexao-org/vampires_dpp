
def frame_center(image):
    """
    Find the center of the frame or cube in pixel coordinates

    Parameters
    ----------
    image : ndarray
        N-D array with the final two dimensions as the (y, x) axes.

    Returns
    -------
    (cy, cx)
        A tuple of the image center in pixel coordinates
    """
    ny = image.shape[-2]
    nx = image.shape[-1]
    return (ny - 1) / 2, (nx - 1) / 2
    