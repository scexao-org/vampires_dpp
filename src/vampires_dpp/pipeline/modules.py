import numpy as np

from vampires_dpp.image_registration import offset_centroids
from vampires_dpp.indexing import cutout_inds


def get_psf_centroids_mpl(mean_image, npsfs=1, nfields=1, suptitle=None, outpath=None):
    import matplotlib.colors as col
    import matplotlib.pyplot as plt

    plt.ion()
    fig, ax = plt.subplots()
    # plot mean image
    fig.suptitle(suptitle)
    ax.imshow(mean_image, origin="lower", cmap="magma", norm=col.LogNorm())
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    fig.tight_layout
    fig.show()

    # get list of (x, y) tuples from user clicks
    output = np.empty((nfields, npsfs, 2))
    for i in range(nfields):
        ax.set_title(f"Please select centroids for field {i + 1}")
        points = fig.ginput(npsfs, show_clicks=True)
        for j, point in enumerate(points):
            inds = cutout_inds(mean_image, center=(point[1], point[0]), window=15)
            output[i, j] = offset_centroids(mean_image, inds)["quad"]
            ax.text(output[i, j, 1] + 2, output[i, j, 0] + 2, str(i), c="green")
        ax.scatter(output[i, :, 1], output[i, :, 0], marker="+", c="green")
        fig.show()
    plt.show(block=False)
    plt.ioff()

    # flip output so file is saved as (x, y)
    return np.flip(output, axis=-1)
