
import cv2
import numpy as np
from scipy import ndimage


def to_binary_mask(mask):
    """Extract the foreground region (pixel value 255)."""
    mask = np.asarray(mask)

    if mask.ndim != 2:
        raise ValueError("Mask must be a grayscale image.")

    return (mask == 255).astype(np.uint8)


def connected_components(mask, connectivity=8):
    """Analyze connected components in a binary mask."""
    binary = to_binary_mask(mask)

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=connectivity
        )
    )

    areas = stats[1:, cv2.CC_STAT_AREA]

    return {
        "num_components": len(areas),
        "labels": labels,
        "stats": stats,
        "centroids": centroids,
        "largest_area": int(areas.max()) if len(areas) else 0,
        "smallest_area": int(areas.min()) if len(areas) else 0,
        "average_area": float(areas.mean()) if len(areas) else 0
    }


def apply_opening(mask, kernel_size=3):
    """Apply Morphological Opening."""
    binary = to_binary_mask(mask)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    return cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, kernel
    )


def apply_closing(mask, kernel_size=3):
    """Apply Morphological Closing."""
    binary = to_binary_mask(mask)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    return cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, kernel
    )


def fill_holes(mask):
    """Apply Hole Filling to a binary mask."""
    binary = to_binary_mask(mask).astype(bool)

    return ndimage.binary_fill_holes(binary).astype(np.uint8)


def calculate_iou(original, processed):
    """Calculate IoU between original and processed masks."""
    original = np.asarray(original) > 0
    processed = np.asarray(processed) > 0

    if original.shape != processed.shape:
        raise ValueError("Mask dimensions must match.")

    intersection = np.logical_and(
        original, processed
    ).sum()

    union = np.logical_or(
        original, processed
    ).sum()

    return float(intersection / union) if union > 0 else 1.0
