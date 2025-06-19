# histogram_features.py

import numpy as np
import cv2

def extract_from_image(image, mask=None, bins=32):
    """
    Compute normalized histogram features from a grayscale or color image.

    Parameters
    ----------
    image : ndarray
        Input image (grayscale or BGR).
    mask : ndarray or None
        Optional mask for ROI.
    bins : int
        Number of bins to compute histogram.

    Returns
    -------
    features : ndarray
        Flattened histogram values.
    labels : list
        Names for each histogram bin feature.
    """
    if len(image.shape) == 3:  # Convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if mask is None:
        mask = np.ones(image.shape, dtype=np.uint8)

    hist = cv2.calcHist([image], [0], mask, [bins], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    labels = [f"Hist_bin_{i}" for i in range(bins)]

    return hist, labels
