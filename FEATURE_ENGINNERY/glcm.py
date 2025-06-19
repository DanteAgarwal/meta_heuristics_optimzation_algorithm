# glcm_features.py

import numpy as np
import cv2
import pandas as pd
from skimage.feature import greycomatrix, greycoprops

ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]
DISTANCES = [1, 3, 5]
PROPERTIES = ['energy', 'correlation', 'dissimilarity', 'homogeneity', 'contrast']

def extract_glcm_features(img_gray):
    features = {}
    for d in DISTANCES:
        for a in ANGLES:
            glcm = greycomatrix(img_gray, [d], [a], symmetric=True, normed=True)
            for prop in PROPERTIES:
                key = f"{prop}_d{d}_a{round(a, 2)}"
                features[key] = greycoprops(glcm, prop)[0, 0]
    return features

def extract_from_image(img):
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    return extract_glcm_features(img_gray)
