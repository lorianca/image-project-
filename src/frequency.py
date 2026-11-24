import cv2
import numpy as np


def centred_spectrum(gray: np.ndarray) -> np.ndarray:
  """Compute the centred log-magnitude Fourier spectrum (0-255 uint8)."""
  f_shift = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
  magnitude = 20.0 * np.log(np.abs(f_shift) + 1.0)
  return cv2.normalize(
      magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
  )


def gaussian_lpf(shape: tuple[int, int], d0: float) -> np.ndarray:
  """Build a 2D centred Gaussian low-pass filter with cutoff d0."""
  rows, cols = shape
  u = np.arange(rows) - rows / 2.0
  v = np.arange(cols) - cols / 2.0
  u_grid, v_grid = np.meshgrid(u, v, indexing="ij")
  return np.exp(-((u_grid**2 + v_grid**2) / (2.0 * d0**2)))
