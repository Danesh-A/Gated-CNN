
# GCNN Nuclei Segmentation
This repository contains the models and training code for the attached thesis project.
# Datasets 
TUPAC dataset used for training, testing. 
TCGA dataset used as out of distribution dataset. 
# Architecture 
This work is inspired by [**Gated-SCNN: Gated Shape CNNs for Semantic Segmentation**](https://github.com/nv-tlabs/GSCNN).

[**Original Architecture**](https://openreview.net/pdf?id=fQDGt0RJkMu)
!(assets/gcnn.png)
# Modifications
(1) Removal of gate 5 from original architecture.
(2) Replaced interpolation blocks with transpose convolution.

# Results

