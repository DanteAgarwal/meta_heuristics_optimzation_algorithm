# glrlm_features.py

import numpy as np
import cv2

def run_length_encoding_1d(line, grayLevel, runLength):
    rle_matrix = np.zeros((grayLevel, runLength), dtype=np.uint32)
    prev_pixel = line[0]
    run_len = 1

    for pixel in line[1:]:
        if pixel == prev_pixel:
            run_len += 1
        else:
            rle_matrix[prev_pixel, min(run_len - 1, runLength - 1)] += 1
            run_len = 1
            prev_pixel = pixel
    rle_matrix[prev_pixel, min(run_len - 1, runLength - 1)] += 1
    return rle_matrix

def get_lines(image, angle):
    h, w = image.shape
    if angle == 0:
        return [image[y, :] for y in range(h)]
    elif angle == 90:
        return [image[:, x] for x in range(w)]
    elif angle == 45:
        return [np.diag(np.fliplr(image), k) for k in range(-h + 1, w)]
    elif angle == 135:
        return [np.diag(image, k) for k in range(-h + 1, w)]
    else:
        raise ValueError("Angle must be 0, 45, 90, or 135")

def compute_glrlm_direction(image, grayLevel, runLength, angle):
    glrlm = np.zeros((grayLevel, runLength), dtype=np.uint32)
    lines = get_lines(image, angle)
    for line in lines:
        if line.size > 0:
            glrlm += run_length_encoding_1d(line, grayLevel, runLength)
    return glrlm

def compute_glrlm(image, grayLevel=256):
    runLength = max(image.shape)
    directions = [0, 45, 90, 135]
    matrices = [compute_glrlm_direction(image, grayLevel, runLength, d) for d in directions]
    return np.stack(matrices, axis=-1)

def _calculate_ij(rlmatrix):
    I, J = np.ogrid[0:rlmatrix.shape[0], 0:rlmatrix.shape[1]]
    return I, J + 1

def _calculate_s(rlmatrix):
    return np.sum(rlmatrix, axis=(0, 1), keepdims=True)[0, 0]

def _safe_divide(x, y):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.true_divide(x, y)
        result[np.isinf(result) | np.isnan(result)] = 0
    return result

def extract_from_image(image, grayLevel=256):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rlmatrix = compute_glrlm(image, grayLevel)
    I, J = _calculate_ij(rlmatrix)
    S = _calculate_s(rlmatrix)
    G = np.sum(rlmatrix, axis=1, keepdims=True)
    R = np.sum(rlmatrix, axis=0, keepdims=True)

    features = np.zeros(11, dtype=np.float64)

    features[0] = np.mean(np.sum(_safe_divide(rlmatrix, (J * J)), axis=(0, 1)) / S)
    features[1] = np.mean(np.sum(rlmatrix * (J * J), axis=(0, 1)) / S)
    features[2] = np.mean(np.sum(G * G, axis=(0, 1)) / S)
    features[3] = np.mean(np.sum(R * R, axis=(0, 1)) / S)

    gray_level, run_length, _ = rlmatrix.shape
    num_voxels = gray_level * run_length
    features[4] = np.mean(S / num_voxels)

    features[5] = np.mean(np.sum(_safe_divide(rlmatrix, (I * I)), axis=(0, 1)) / S)
    features[6] = np.mean(np.sum(rlmatrix * (I * I), axis=(0, 1)) / S)
    features[7] = np.mean(np.sum(_safe_divide(rlmatrix, (I * I * J * J)), axis=(0, 1)) / S)

    temp = rlmatrix * (I * I)
    features[8] = np.mean(np.sum(_safe_divide(temp, (J * J)), axis=(0, 1)) / S)

    temp = rlmatrix * (J * J)
    features[9] = np.mean(np.sum(_safe_divide(temp, (I * I)), axis=(0, 1)) / S)

    features[10] = np.mean(np.sum(rlmatrix * (I * I * J * J), axis=(0, 1)) / S)

    return features
