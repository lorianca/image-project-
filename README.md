# Segmentation (Checkpoint 2)

This folder contains the segmentation-specific Checkpoint 2 artifacts for CSE281.

> **Scope:** This covers only the segmentation part of Checkpoint 2. It does not claim to complete the other work packages or the full project.

---

## Dataset Setup

Extract `Dataset.zip` so the `Dataset/` directory is placed under `data/`:

- `data/Dataset/cedar_apple_rust/`
- `data/Dataset/soybean_bacterial_blight/`
- `data/Dataset/wheat_stripe_rust/`

Paths in `data_split.csv` are relative to the extracted zip root (`Dataset/...`). If referencing paths directly, set `DATA_ROOT` to the directory containing `Dataset/`.

---

## Baseline Code & Artifacts

- `src/segmentation.py`: Input convention, trivial and simple baselines, IoU calculation, and mask loading.
- `experiments/checkpoint2_segmentation_baselines_development.csv`: Per-image evaluation results on the development set.
- `CHECKPOINT2_SEGMENTATION.md`: Checkpoint notes and aggregate baseline scores.
- `tests/test_metrics.py`: Unit test verifying IoU against hand-calculated values.
