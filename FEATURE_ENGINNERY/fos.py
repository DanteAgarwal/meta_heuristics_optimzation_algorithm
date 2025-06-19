# fos_features.py

import numpy as np
import cv2

def extract_from_image(image, mask=None):
    """
    Extract first-order statistical features from a grayscale image.

    Parameters
    ----------
    image : ndarray
        Grayscale or RGB image.
    mask : ndarray or None
        Optional binary mask (same size as image). If None, full image is used.

    Returns
    -------
    features : ndarray
        Extracted FOS feature values.
    labels : list
        Corresponding feature names.
    """

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if mask is None:
        mask = np.ones(image.shape, dtype=np.uint8)
    else:
        mask = mask.astype(np.uint8)

    image = image.astype(np.uint8)
    level_min, level_max = 0, 255
    bins = (level_max - level_min) + 1
    roi = image[mask.astype(bool)].ravel()
    H = np.histogram(roi, bins=bins, range=(level_min, level_max), density=True)[0]
    i = np.arange(0, bins)

    features = np.zeros(16, dtype=np.float64)
    features[0] = np.dot(i, H)                                       # Mean
    features[1] = np.dot((i - features[0])**2, H)                    # Variance
    features[2] = np.percentile(roi, 50)                             # Median
    features[3] = np.argmax(H)                                       # Mode
    features[4] = np.dot((i - features[0])**3, H) / (np.sqrt(features[1])**3 + 1e-8)  # Skewness
    features[5] = np.dot((i - features[0])**4, H) / (np.sqrt(features[1])**4 + 1e-8)  # Kurtosis
    features[6] = np.sum(H**2)                                       # Energy
    features[7] = -np.sum(H * np.log(H + 1e-16))                     # Entropy
    features[8] = roi.min()                                          # Min
    features[9] = roi.max()                                          # Max
    features[10] = (np.sqrt(features[1]) / features[0]) if features[0] != 0 else 0  # Coefficient of Variation
    features[11] = np.percentile(roi, 10)
    features[12] = np.percentile(roi, 25)
    features[13] = np.percentile(roi, 75)
    features[14] = np.percentile(roi, 90)
    features[15] = features[14] - features[11]                       # Histogram width

    labels = [
        "FOS_Mean", "FOS_Variance", "FOS_Median", "FOS_Mode", "FOS_Skewness",
        "FOS_Kurtosis", "FOS_Energy", "FOS_Entropy", "FOS_MinimalGrayLevel",
        "FOS_MaximalGrayLevel", "FOS_CoefficientOfVariation", "FOS_10Percentile",
        "FOS_25Percentile", "FOS_75Percentile", "FOS_90Percentile", "FOS_HistogramWidth"
    ]

    return features, labels
