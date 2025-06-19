# hog_features.py

import numpy as np
import cv2
from skimage import feature

def extract_from_image(image, orientations=9, block_norm='L2', resize_to=(512, 512)):
    """
    Parameters
    ----------
    image : ndarray
        Input image, grayscale or RGB.
    orientations : int
        Number of orientation bins.
    block_norm : str
        Block normalization method.
    resize_to : tuple
        Resize image before HOG extraction.

    Returns
    -------
    features : ndarray
        Flattened HOG descriptor.
    labels : list
        List of feature names.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if resize_to:
        image = cv2.resize(image, resize_to)

    fd, _ = feature.hog(image, orientations=orientations,
                        block_norm=block_norm, visualize=True)

    labels = [f'HOG_{i}' for i in range(fd.shape[0])]
    return fd, labels
