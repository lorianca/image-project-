import numpy as np


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
    noisy = img + rng.normal(0.0, level, img.shape)
  elif noise_type == "speckle":
    noisy = img + img * rng.normal(0.0, level, img.shape)
  elif noise_type == "salt_and_pepper":
    noisy = img.copy()
    prob = rng.random(img.shape[:2])
    noisy[prob < level / 2.0] = 0.0
    noisy[(prob >= level / 2.0) & (prob < level)] = 1.0
  else:
    raise ValueError(f"Unsupported noise type: {noise_type}")

  return (np.clip(noisy, 0.0, 1.0) * 255.0).round().astype(np.uint8)
