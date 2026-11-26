import numpy as np


def median_numpy_3x3(gray: np.ndarray) -> np.ndarray:
  """Apply a 3x3 median filter in NumPy using reflect padding."""
  padded = np.pad(gray, pad_width=1, mode="reflect")
  patches = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
  return np.median(patches, axis=(-2, -1)).round().astype(np.uint8)


def compute_psnr_mse(
    reference: np.ndarray, target: np.ndarray
) -> tuple[float, float]:
  """Compute Mean Squared Error (MSE) and Peak Signal-to-Noise Ratio (PSNR)."""
  ref = reference.astype(np.float64)
  tgt = target.astype(np.float64)

  mse = float(np.mean((ref - tgt) ** 2))
  if mse == 0.0:
    return 0.0, float("inf")

  psnr = float(10.0 * np.log10((255.0**2) / mse))
  return mse, psnr
