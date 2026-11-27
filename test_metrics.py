import numpy as np
from src.metrics import iou, mse, psnr


def test_iou():
    a = np.array([[1, 1], [0, 0]])
    b = np.array([[1, 0], [1, 0]])
    assert abs(iou(a, b) - 1 / 3) < 1e-9
    assert iou(a, a) == 1.0
    assert iou(a, 1 - a) == 0.0
    assert iou(np.zeros((2, 2)), np.zeros((2, 2))) == 1.0


def test_mse_psnr():
    x = np.zeros((2, 2), dtype=np.uint8)
    y = np.full((2, 2), 2, dtype=np.uint8)
    assert mse(x, y) == 4.0
    assert abs(psnr(x, y) - 10 * np.log10(255 ** 2 / 4)) < 1e-9
    assert psnr(x, x) == float("inf")
