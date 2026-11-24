import numpy as np


def median_numpy_3x3(gray: np.ndarray) -> np.ndarray:
  """Apply a 3x3 median filter in NumPy using reflect padding."""
  padded = np.pad(gray, pad_width=1, mode="reflect")
  patches = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
  return np.median(patches, axis=(-2, -1)).round().astype(np.uint8)
