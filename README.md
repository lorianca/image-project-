# image-project-Field-PDD (Plant Disease Dataset with pixel-level masks)
Classical image processing pipeline for Field-PDD leaf lesion quantification
# Problem Statement & Dataset:
Using the Field-PDD dataset of real-world leaf images with pixel-level ground-truth masks, this project builds a classical image-processing pipeline to segment diseased regions and measure lesion area and infection ratio. Real-world field images in this dataset suffer from practical visual issues—including uneven illumination or glare, low contrast, blur and varying sharpness, noise, and holes, gaps, or touching regions after thresholding—making a single global threshold or a single filter insufficient.

# Target User & Task:
Agronomists and crop monitoring engineers; given an RGB field image of a plant leaf, output a cleaned binary lesion mask (`0/1`) and quantify the lesion area and infection ratio.
# Primary Metric:
  Intersection over Union (`IoU`) between the predicted lesion mask and the pixel-level ground truth, because both missed lesions (false negatives) and false detections (false positives) directly distort disease severity estimates.
# Dataset:
Field-PDD (240 real-world leaf images with pixel-level ground-truth masks).
# Source:
`https://www.kaggle.com/datasets/alexzcheny/testdataset
# Licence
https://creativecommons.org/licenses/by-nc/4.0/
CC BY-NC 4.0 
