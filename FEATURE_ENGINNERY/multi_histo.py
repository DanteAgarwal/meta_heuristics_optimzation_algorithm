# multi_histogram_features.py

import numpy as np
import cv2

def extract_from_image(image, mask=None, bins=16, grid=(2, 2)):
    """
    Divide image into grids and compute histogram features from each region.

    Parameters
    ----------
    image : ndarray
        Input image (grayscale or BGR).
    mask : ndarray or None
        Optional binary mask for ROI.
    bins : int
        Number of histogram bins.
    grid : tuple
        Number of (rows, cols) to split the image into.

    Returns
    -------
    features : ndarray
        Flattened histogram values from each grid region.
    labels : list
        Feature labels.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = image.shape
    gh, gw = h // grid[0], w // grid[1]

    features = []
    labels = []

    for i in range(grid[0]):
        for j in range(grid[1]):
            patch = image[i*gh:(i+1)*gh, j*gw:(j+1)*gw]
            if mask is not None:
                patch_mask = mask[i*gh:(i+1)*gh, j*gw:(j+1)*gw]
            else:
                patch_mask = None

            hist = cv2.calcHist([patch], [0], patch_mask, [bins], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)

            labels.extend([f"MultiHist_r{i}_c{j}_bin{k}" for k in range(bins)])

    return np.array(features), labels
