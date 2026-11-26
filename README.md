Segmentation
This folder contains the segmentation-specific Checkpoint 2 artifacts.
Place dataset
Extract the provided `Dataset.zip` so the dataset's `Dataset/` directory is located at:
`data/Dataset/cedar_apple_rust/`, `data/Dataset/soybean_bacterial_blight/`,
and `data/Dataset/wheat_stripe_rust/`.
The `data_split.csv` manifest paths are relative to the extracted zip root (`Dataset/...`).
If using the paths directly, set `DATA_ROOT` to the directory containing `Dataset/`.
Baseline code
`src/segmentation.py`: input convention, two baselines, IoU and mask loader.
`experiments/checkpoint2_segmentation_baselines_development.csv`: per-development-image results.
`CHECKPOINT2_SEGMENTATION.md`: checkpoint notes and actual aggregate scores.
`tests/test_metrics.py`: hand-calculated IoU test.
This is only the segmentation part of Checkpoint 2. It does not claim to complete the other work packages or the full project.
