# Segmentation Baselines (Checkpoint 2)

NumPy thresholding implementations, baseline segmentation methods, IoU evaluation, and CSV experiment logging for the segmentation work package.

## Input & Output Conventions

- **Input images:** Non-empty **2D `uint8`** grayscale arrays (`0..255`).
- **Output masks:** Binary `uint8` arrays (`0` or `1`, where `1` is foreground).
- **Foreground polarity (`foreground`):**
  - `"bright"` (default): pixels strictly greater than threshold (`> T`).
  - `"dark"`: pixels less than or equal to threshold (`<= T`).

## Key Functions

- `validate_image(image)` & `check_foreground(foreground)`: Input validation helpers.
- `global_threshold(image, threshold=127, foreground="bright")`: Fixed global thresholding (`0..255`).
- `otsu_threshold(image)`: Between-class variance maximization in NumPy (returns integer threshold `0..255`).
- `verify_otsu(image)`: Compares NumPy Otsu threshold and binary mask against `cv2.threshold(..., THRESH_OTSU)`.
- `trivial_baseline(image, foreground="bright")`: Fixed threshold `T = 127`.
- `simple_baseline(image, foreground="bright")`: Otsu thresholding followed by a `3x3` morphological opening (`cv2.MORPH_OPEN`).
- `iou(pred, truth)`: Computes Intersection over Union between two binary masks (returns `1.0` if both masks are empty).
- `log_run(path, row)` & `measure_baselines(...)`: Evaluates both baselines on the **development** set and appends runs to a CSV file using `LOG_FIELDS`.

## Self-Check & Usage

Run the built-in verification checks:

```bash
python segmentation.py
