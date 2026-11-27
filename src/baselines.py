import cv2
import numpy as np


def trivial_baseline(gray):
    return (gray > 127).astype(np.uint8)


def simple_baseline(gray):
    _, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
