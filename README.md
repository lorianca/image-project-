# image-project-Field-PDD
### Classical Image Processing Pipeline for Field-PDD Leaf Lesion Quantification

## Problem Statement & Dataset

Using the Field-PDD dataset of real-world leaf images with pixel-level ground-truth masks, this project builds a classical image-processing pipeline to segment diseased regions and measure lesion area and infection ratio.

Real-world field images may suffer from practical visual issues, including uneven illumination or glare, low contrast, blur and varying sharpness, noise, and holes, gaps, or touching regions after thresholding. These challenges make a single global threshold or a single filter insufficient for reliable segmentation.

## Target User & Task

**Target Users:** Agronomists and crop monitoring engineers.

**Task:** Given an RGB field image of a plant leaf, the system aims to output a cleaned binary lesion mask (`0/1`) and quantify the lesion area and infection ratio.

## Primary Metric

**Intersection over Union (IoU)** between the predicted lesion mask and the pixel-level ground truth.

IoU is the primary evaluation metric because both missed lesions (false negatives) and false detections (false positives) affect segmentation quality and disease severity estimates.

## Dataset

**Field-PDD:** 240 real-world leaf images with pixel-level ground-truth masks.

- **Source:**
https://www.kaggle.com/datasets/alexzcheny/testdataset
- **Licence:**
  [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

## Results Table

The following table compares the performance of the two baseline methods and the three proposed processing pipelines on the Development (Dev) and Test datasets.

Results will be updated after completing the implementation and running the evaluation script.

| Method / Pipeline | Split | Mean IoU | PSNR | MSE | Lesion Area / Infection Ratio Error |
|---|---|---|---|---|---|
| Trivial Baseline | Dev | TBD | N/A | N/A | TBD |
| Trivial Baseline | Test | TBD | N/A | N/A | TBD |
| Simple Baseline | Dev | TBD | N/A | N/A | TBD |
| Simple Baseline | Test | TBD | N/A | N/A | TBD |
| Pipeline A | Dev | TBD | TBD | TBD | TBD |
| Pipeline A | Test | TBD | TBD | TBD | TBD |
| Pipeline B | Dev | TBD | TBD | TBD | TBD |
| Pipeline B | Test | TBD | TBD | TBD | TBD |
| Pipeline C | Dev | TBD | TBD | TBD | TBD |
| Pipeline C | Test | TBD | TBD | TBD | TBD |

*Note: PSNR and MSE are applicable only when image restoration quality is evaluated against a suitable reference image.*

## Pipeline Components

The following table summarizes the main components of the baselines and proposed pipelines. The final methods should match the actual implementation.

| Method | Enhancement | Restoration | Segmentation | Morphology | Measurement |
|---|---|---|---|---|---|
| Trivial Baseline | None | None | Global Threshold (127) | None | Pixel Count |
| Simple Baseline | None | None | Otsu Thresholding | 3×3 Opening | Pixel Count |
| Pipeline A | Gamma Correction / Histogram Equalization | None | Global / Otsu Thresholding | None | Pixel Count |
| Pipeline B | Gamma Correction / Histogram Equalization | Median / Gaussian Filtering | Adaptive Thresholding / Edge Detection | None | Pixel Count |
| Pipeline C | Enhancement from Pipeline B | Noise Removal | Adaptive / Edge-based Segmentation | Opening, Closing, Hole Filling | Connected Components / Area Filtering |

- **Degradation & Restoration (`src/data.py`, `src/restoration.py`):** Controlled noise generation (`degrade_image`) and 3x3 median filtering in NumPy (`median_numpy_3x3`).
- **Frequency Analysis (`src/frequency.py`):** Centred Fourier log-magnitude spectrum (`centred_spectrum`) and 2D Gaussian low-pass filter (`gaussian_lpf`).


## Experiments Notebook

An executed Google Colab notebook is available at `notebooks/degradation_median_frequency_experiments.ipynb`[cite: 8]. It covers:

- **Dataset Samples:** Representative `Apple` (Cedar Apple Rust) and `Wheat` (Wheat Stripe Rust) images across `train`, `val`, and `test` splits[cite: 1].
- **Noise & Restoration:** Salt-and-pepper impulse noise degradation and restoration using a 3x3 median filter, evaluated with `MSE` and `PSNR`[cite: 1].
- **Frequency Analysis:** Centered 2D FFT magnitude spectrum and low-pass filtering[cite: 1].

## Setup and Run Instructions



### 1. Clone the Repository

```bash
git clone [https://github.com/lorianca/image-project-.git](https://github.com/lorianca/image-project-.git)
cd image-project-
