import os
import cv2
import numpy as np


def load_grayscale_image(image_path: str) -> np.ndarray:
  """Load an image from disk and convert it to a grayscale uint8 array."""
  if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}")

  img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
  if img_bgr is None:
    raise ValueError(f"Failed to read image: {image_path}")

  return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def degrade_image(
    image: np.ndarray,
    noise_type: str = "gaussian",
    level: float = 0.15,
    seed: int = 42,
) -> np.ndarray:
  """Add controlled noise to an image using a fixed random seed."""
  rng = np.random.default_rng(seed)
  img = image.astype(np.float32) / 255.0

  if noise_type == "gaussian":
    # Add zero-mean Gaussian noise
    noisy = img + rng.normal(0.0, level, img.shape)
  elif noise_type == "speckle":
    # Add multiplicative speckle noise
    noisy = img + img * rng.normal(0.0, level, img.shape)
  elif noise_type == "salt_and_pepper":
    # Add salt (1.0) and pepper (0.0) impulse noise
    noisy = img.copy()
    prob = rng.random(img.shape[:2])
    noisy[prob < level / 2.0] = 0.0
    noisy[(prob >= level / 2.0) & (prob < level)] = 1.0
  else:
    raise ValueError(f"Unsupported noise type: {noise_type}")

  return (np.clip(noisy, 0.0, 1.0) * 255.0).round().astype(np.uint8)
