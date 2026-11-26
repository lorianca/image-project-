import csv
import os
import time
from datetime import date

import cv2
import numpy as np

# Columns of the shared experiment log
LOG_FIELDS = ["run_id", "date", "author", "pipeline", "stage", "method",
              "parameters", "seed", "time_s", "metric_name", "metric_value"]


def validate_image(image):
    ## Return image as an array raise ValueError unless non-empty 2-D uint8
    image = np.asarray(image)
    if image.ndim != 2 or image.dtype != np.uint8 or image.size == 0:
        raise ValueError(f"Image must be a non-empty 2D uint8 array, got shape "
                         f"{image.shape} and dtype {image.dtype}.")
    return image


def check_foreground(foreground):
    #Raise ValueError unless ``foreground`` is 'bright' or 'dark'
    if foreground not in ("bright", "dark"):
        raise ValueError("foreground must be 'bright' or 'dark'")


def global_threshold(image, threshold=127, foreground="bright"):
    ##Fixed threshold: ``image > threshold`` (bright) or ``<= threshold`` (dark).
    image = validate_image(image)
    check_foreground(foreground)
    if not 0 <= threshold <= 255:
        raise ValueError("threshold must be between 0 and 255")
    return (image > threshold if foreground == "bright"
            else image <= threshold).astype(np.uint8)


def otsu_threshold(image):
   
    image = validate_image(image)
    p = np.bincount(image.ravel(), minlength=256) / image.size
    w, m = np.cumsum(p), np.cumsum(p * np.arange(256))
    denom = w * (1 - w)
    score = np.zeros(256)
    ok = denom > 0                         
    # skip t where one class is empty
    score[ok] = (m[-1] * w[ok] - m[ok]) ** 2 / denom[ok]
    return int(np.argmax(score))


def verify_otsu(image):
    ##True if the NumPy Otsu gives OpenCV's threshold and mask
    image = validate_image(image)
    t = otsu_threshold(image)
    cv_t, cv_mask = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return t == int(cv_t) and np.array_equal(global_threshold(image, t), cv_mask)


def trivial_baseline(image, foreground="bright"):
    ##Baseline 1: fixed threshold of 127 on the raw grayscale image
    return global_threshold(image, 127, foreground)


def simple_baseline(image, foreground="bright"):
    ##Baseline 2: Otsu threshold, then one 3 x 3 square opening
    mask = global_threshold(image, otsu_threshold(image), foreground)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def iou(pred, truth):
    ##IoU of two masks (1.0 if both empty) Fallback: use ``metrics.py`` IoU
    pred, truth = np.asarray(pred) > 0, np.asarray(truth) > 0
    if pred.shape != truth.shape:
        raise ValueError("Prediction and ground truth shapes differ.")
    union = np.logical_or(pred, truth).sum()
    return float(np.logical_and(pred, truth).sum() / union) if union else 1.0


def log_run(path, row):
    ##Append one run (dict keyed by ``LOG_FIELDS``) to the CSV, header once
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in LOG_FIELDS})


def measure_baselines(images, masks, foreground="bright", author="", log_path=None,
                      metric=iou, seed=0):
   
    check_foreground(foreground)
    if len(images) != len(masks) or len(images) == 0:
        raise ValueError("Images and masks must have equal non-zero lengths.")
    methods = {
        "trivial_T127"
        : (lambda im: trivial_baseline(im, foreground),
                         {"threshold": 127, "foreground": foreground}),
        "simple_otsu_open3x3"
        : (lambda im: simple_baseline(im, foreground),
                                {"method": "Otsu", "kernel": "3x3",
                                 "foreground": foreground}),
    }
    results = []
    for name, (fn, params) in methods.items():
        start = time.perf_counter()
        scores = [float(metric(fn(im), mk)) for im, mk in zip(images, masks)]
        elapsed = (time.perf_counter() - start) / len(images)
        mean, std = float(np.mean(scores)), float(np.std(scores))
        results.append({"method": name, "mean": mean, "std": std,
                        "per_image": scores})
        if log_path:
            log_run(log_path, {
                "run_id": f"baseline-{name}-{foreground}-{time.time_ns()}",
                "date": date.today().isoformat(), "author": author,
                "pipeline": "baseline", "stage": "segmentation", "method": name,
                "parameters": str(params), "seed": seed, "time_s": round(elapsed, 5),
                "metric_name": "IoU", "metric_value": round(mean, 5)})
    return results


def _self_check():
    ## Otsu vs OpenCV, hand-worked examples, and a guard that foreground is used
    for s in range(30):
        base = np.random.default_rng(s).integers(0, 256, (64, 64))
        img = np.clip(base // (s % 3 + 1) + 20 * s, 0, 255).astype(np.uint8)
        assert verify_otsu(img), f"Otsu differs from OpenCV (seed {s})"
    # 15 pixels = 10, 10 pixels = 90: p0 = 0.6, p1 = 0.4, sigma_b^2 = 0.6*0.4*80^2
    # = 1536 for every t in 10..89; the first maximum is t = 10.
    tiny = np.array([10] * 15 + [90] * 10, dtype=np.uint8).reshape(5, 5)
    assert otsu_threshold(tiny) == 10 and verify_otsu(tiny)
    row = np.array([[100, 127, 128]], dtype=np.uint8)       
    # T = 127 by hand
    assert global_threshold(row, 127, "bright").tolist() == [[0, 0, 1]]
    assert global_threshold(row, 127, "dark").tolist() == [[1, 1, 0]]
    scene = np.full((20, 20), 20, dtype=np.uint8)
    # lone noise pixel
    scene[5, 5] = 200   
    # real object                                     
    scene[10:16, 10:16] = 200                                
    out = simple_baseline(scene)
     # opening keeps object
    assert out[5, 5] == 0 and out[12, 12] == 1  
    # regression: foreground            
    dark = np.full((20, 20), 200, dtype=np.uint8)            
    dark[5:12, 5:12] = 30
    truth = np.zeros((20, 20), dtype=np.uint8)
    truth[5:12, 5:12] = 1
    for fg, want in (("dark", 1.0), ("bright", 0.0)):
        assert [r["mean"] for r in measure_baselines([dark], [truth], fg)] == [want] * 2
    for bad in (np.zeros((4, 4)), np.zeros((4, 4, 3), np.uint8),
                np.zeros((0, 4), np.uint8)):
        try:
            otsu_threshold(bad)
        except ValueError:
            continue
        raise AssertionError("invalid image accepted")
    print("all checks passed: Otsu == OpenCV (30 images), hand examples, opening, "
          "foreground used when scoring, invalid images rejected")


if __name__ == "__main__":
    _self_check()