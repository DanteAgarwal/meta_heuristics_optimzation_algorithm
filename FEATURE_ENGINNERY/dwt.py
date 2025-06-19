# dwt_features.py

import pywt
import numpy as np
import cv2
import math

def _next_power_of_two(n):
    return 2 ** math.ceil(math.log(n, 2))

def _pad_image_power_2(f):
    N1, N2 = f.shape
    N1_deficit = _next_power_of_two(N1) - N1
    N2_deficit = _next_power_of_two(N2) - N2
    return np.pad(
        f,
        ((math.floor(N1_deficit / 2), N1_deficit - math.floor(N1_deficit / 2)),
         (math.floor(N2_deficit / 2), N2_deficit - math.floor(N2_deficit / 2))),
        mode='constant'
    )

def extract_from_image(image, mask=None, wavelet='bior3.3', levels=3):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = np.ones(image.shape)

    image = _pad_image_power_2(image)
    mask = _pad_image_power_2(mask)

    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=levels)
    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)

    coeffs_mask = pywt.wavedec2(mask, wavelet=wavelet, level=levels)
    coeff_arr_mask, coeff_slices_mask = pywt.coeffs_to_array(coeffs_mask)
    coeff_arr_mask[coeff_arr_mask != 0] = 1

    features = []
    labels = []
    for level in range(1, levels + 1):
        for name in ['da', 'dd', 'ad']:
            D_f = coeff_arr[coeff_slices[level][name]]
            D_mask = coeff_arr_mask[coeff_slices[level][name]]
            D = D_f.flatten()[D_mask.flatten().astype(bool)]
            features.append(abs(D).mean())
            features.append(abs(D).std())
            labels.append(f'DWT_{wavelet}_level_{level}_{name}_mean')
            labels.append(f'DWT_{wavelet}_level_{level}_{name}_std')

    return np.array(features), labels
