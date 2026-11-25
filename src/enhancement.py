from pathlib import Path
import cv2
import numpy as np
import pandas as pd


def gamma_correction(rgb: np.ndarray, gamma: float = 0.7) -> np.ndarray:
    """Apply power-law (gamma < 1) correction on an RGB uint8 image."""
    if gamma <= 0:
        raise ValueError("gamma must be greater than 0.")
    if abs(gamma - 1.0) < 1e-6:
        return rgb.copy()

    inv_gamma = 1.0 / gamma
    lut = np.array(
        [np.clip(((i / 255.0) ** inv_gamma) * 255.0, 0, 255) for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(rgb, lut)


def clahe_luminance(
    rgb: np.ndarray, clip_limit: float = 1.5, tile_grid_size: tuple = (8, 8)
) -> np.ndarray:
    """Optional CLAHE on the L channel of LAB color space."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_ch)
    return cv2.cvtColor(cv2.merge((l_enhanced, a_ch, b_ch)), cv2.COLOR_LAB2RGB)


def enhance_image(
    rgb: np.ndarray,
    gamma: float = 0.7,
    use_clahe: bool = False,
    clip_limit: float = 1.5,
    tile_grid_size: tuple = (8, 8),
) -> np.ndarray:
    """Enhance an RGB leaf image using gamma < 1 by default."""
    enhanced = gamma_correction(rgb, gamma=gamma)
    if use_clahe:
        enhanced = clahe_luminance(
            enhanced, clip_limit=clip_limit, tile_grid_size=tile_grid_size
        )
    return enhanced


def enhance_split_from_csv(
    split_csv: Path,
    orig_root: Path,
    out_root: Path,
    split_name: str = "val",
    gamma: float = 0.7,
    use_clahe: bool = False,
) -> int:
    """Enhance images belonging to `split_name` from `OriginalData` and save them."""
    df = pd.read_csv(split_csv)
    sub = df[df["split"].astype(str).str.lower() == split_name.lower()]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    by_rel = {}
    by_name = {}
    for p in orig_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            rel = str(p.relative_to(orig_root)).replace("\\", "/")
            by_rel[rel] = p
            by_name.setdefault(p.name, p)

    saved_count = 0
    for rel_img in sub["image"].astype(str):
        rel_clean = rel_img.strip().lstrip("./")
        src_path = by_rel.get(rel_clean) or by_name.get(Path(rel_clean).name)
        if src_path is None:
            continue

        bgr = cv2.imread(str(src_path))
        if bgr is None:
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out_rgb = enhance_image(rgb, gamma=gamma, use_clahe=use_clahe)
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

        dst_path = out_root / split_name / rel_clean
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst_path), out_bgr)
        saved_count += 1

    return saved_count
