import cv2
import numpy as np
import pandas as pd


def load_split(csv_path, subset):
    df = pd.read_csv(csv_path)
    return df[df["subset"] == subset].reset_index(drop=True)


def load_image(path, gray=False, size=None):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size is not None:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img.astype(np.uint8)


def load_mask(label_path, size=None):
    lab = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
    if lab is None:
        raise FileNotFoundError(label_path)
    if size is not None:
        lab = cv2.resize(lab, size, interpolation=cv2.INTER_NEAREST)
    lesion = (lab == 255).astype(np.uint8)
    leaf = (lab > 0).astype(np.uint8)
    return lesion, leaf


def load_sample(img_path, label_path, gray=False, size=None):
    img = load_image(img_path, gray=gray, size=size)
    lesion, leaf = load_mask(label_path, size=size)
    return img, lesion, leaf
