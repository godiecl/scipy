#  Copyright (c) 2025. Departamento de Ingenieria de Sistemas y Computacion.
import logging
import numpy as np

from matplotlib import pyplot as plt

from scipy.linalg import svd

from logger import configure_logging
from skimage import io, color

from typeguard import typechecked


@typechecked
def main():
    # configure logging
    configure_logging(log_level=logging.DEBUG)
    log = logging.getLogger(__name__)
    log.debug("Starting ..")

    # load the pleiades image
    pleiades = io.imread(
        "../../data/pleiades.png"
        # "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Pleiades_large.jpg/960px-Pleiades_large.jpg"
    )
    log.debug(f"Pleiades image loaded: {pleiades.shape}")

    # convert the image to grayscales
    pleiades_grey = color.rgb2gray(pleiades)
    log.debug(f"Pleiades image converted to grayscale: {pleiades_grey.shape}")

    # svd decomposition
    u, s, vt = svd(pleiades_grey, full_matrices=False)
    log.debug(f"SVD decomposition done: u={u.shape}, s={s.shape}, vt={vt.shape}")

    # compress the image using the first k singular values
    k = 12
    log.debug(f"Compressing with k={k} singular values.")
    s = np.diag(s[:k])  # keep only the first k singular values
    pleiades_compressed = np.dot(u[:, :k], np.dot(s, vt[:k, :]))

    original_size = pleiades_grey.size
    compressed_size = u[:, :k].size + s.size + vt[:k, :].size
    compression_ratio = 100 * (1 - compressed_size / original_size)

    log.debug(f"Original size: {original_size}.")
    log.debug(f"Compressed size: {compressed_size}.")
    log.debug(f"Reduction: {compression_ratio:.2f}%.")


    # plot the original, greyscale and compressed images
    plt.figure(figsize=(15, 5))

    # show the original image
    plt.subplot(1, 3, 1)
    plt.title("The Pleides original")
    plt.imshow(pleiades)
    plt.axis("off")

    # show the greyscale image
    plt.subplot(1, 3, 2)
    plt.title("The Pleides in grey")
    plt.imshow(pleiades_grey, cmap="gray")
    plt.axis("off")

    # show the greyscale image
    plt.subplot(1, 3, 3)
    plt.title("The Pleides compressed grey")
    plt.imshow(pleiades_compressed, cmap="gray")
    plt.axis("off")

    plt.show()

    log.debug("Done.")


if __name__ == "__main__":
    main()
