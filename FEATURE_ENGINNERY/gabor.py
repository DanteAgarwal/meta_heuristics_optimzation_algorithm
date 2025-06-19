# gabor_features.py

import numpy as np
import cv2
from skimage.filters import gabor_kernel
from scipy import signal

def _image_xor(f):
    f = f.astype(np.uint8)
    return np.bitwise_xor(f, np.ones_like(f, dtype=np.uint8))

def extract_from_image(image, mask=None, deg=4, freq=[0.05, 0.4]):
    """
    Parameters
    ----------
    image : ndarray
        Grayscale or RGB image.
    mask : ndarray or None
        Binary mask of the same size as image. If None, full image is considered ROI.
    deg : int
        Number of directions (e.g., 4 means 0, 45, 90, 135 degrees).
    freq : list
        Frequencies to use for Gabor kernels.

    Returns
    -------
    features : ndarray
        Flattened array of mean and std values from each Gabor response.
    labels : list
        Corresponding feature labels.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if mask is None:
        mask = np.ones(image.shape, dtype=np.uint8)

    kernels = []
    labels = []
    for theta_i in range(deg):
        theta = theta_i / deg * np.pi
        for f in freq:
            kernel = np.real(gabor_kernel(f, theta=theta))
            kernels.append(kernel)
            labels.append(f'Gabor_th_{round(theta * 180 / np.pi)}_freq_{f}_mean')
            labels.append(f'Gabor_th_{round(theta * 180 / np.pi)}_freq_{f}_std')

    mask_c = _image_xor(mask)
    mask_convs = [
        np.abs(np.sign(signal.convolve2d(mask_c, np.ones(k.shape), mode='same')) - 1)
        for k in kernels
    ]

    features = np.zeros((len(kernels), 2), dtype=np.float64)
    for k, kernel in enumerate(kernels):
        filtered = signal.convolve2d(image, kernel, mode='same')
        filtered *= mask_convs[k]
        roi = filtered.ravel()[mask_convs[k].ravel() == 1]
        features[k, 0] = roi.mean() if roi.size else 0
        features[k, 1] = roi.std() if roi.size else 0

    return features.flatten(), labels
