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

## Image Enhancement Experiments & Results

Our standardized enhancement pipeline applies **Gamma Correction (`gamma = 0.7`)** followed by **CLAHE** on the **L channel** in `LAB` color space (`clipLimit = 2.0`, `tileGridSize = (8, 8)`) to lift shadow/illumination regions and improve local contrast while preserving leaf color fidelity.

### 1. Gamma (`0.7`) + CLAHE (`L`-channel, `clipLimit = 2.0`)+unsharpe
![Gamma and CLAHE ana unsharpe  Comparison](results/Apple Gamma (Unsharpe + clashe).png)


### 2. Gamma (`0.7`) + CLAHE (`L`-channel, `clipLimit = 2.0`)+unsharpe
![Gamma and CLAHE ana unsharpe  Comparison](results/wheat Gamma (Unsharpe + clashe).png)

### 3. Filters Comparison vs. Original (`Laplacian`)
![Filters Comparison](results/laplacian.png)
### 4 . Filters Comparison vs. Original ( `High-pass`)
![Filters Comparison](results/high pass.png)
### 5 . Synthetic Salt-and-Pepper Noise Experiment
![Salt and Pepper Noise Removal](results/median + gamma + clashe.png)

## Setup and Run Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/lorianca/image-project-.git
cd image-project-
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Evaluation

```bash
python src/evaluate.py
```
