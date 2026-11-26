import numpy as np


def iou(pred, gt):
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / union)


def mse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean((a - b) ** 2))


def psnr(a, b, max_val=255.0):
    m = mse(a, b)
    if m == 0:
        return float("inf")
    return float(10 * np.log10(max_val ** 2 / m))
