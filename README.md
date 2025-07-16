
# GCNN Nuclei Segmentation
## About 
This repository contains the models and training code for the attached thesis project.

Tumor Proliferation Assessment Challenge 2016 (TUPAC) dataset used for training and testing.

The Cancer Genome Atlas (TCGA) dataset used as an independent test dataset. 
# Architecture 
This work is inspired by [**Gated-SCNN: Gated Shape CNNs for Semantic Segmentation**](https://github.com/nv-tlabs/GSCNN).

[**Original Architecture**](https://openreview.net/pdf?id=fQDGt0RJkMu)
![test](assets/gcnn.png)
# Modifications
(1) Removal of gate 5 from original architecture.
(2) Replaced interpolation blocks with transpose convolution.

# Results

