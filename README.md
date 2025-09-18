# End-to-End MRI-Based Model for Glioma Subtype and Grade Prediction

## Overview

This repository contains the official implementation for the paper: **"Segmentation-Free End-to-End Multi-Sequence MRI-Based Model for Prediction of Molecular Subtypes and Grades in Adult-Type Gliomas"**.

Our project introduces the Multi-Sequence Multitask Deep Learning Model (MMDLM), an end-to-end deep learning framework that predicts key molecular subtypes (IDH mutation, 1p/19q co-deletion) and histological grade (LGG vs. GBM) in adult-type gliomas directly from multi-modal MRI scans, eliminating the need for manual tumor segmentation.

The model was trained and validated on a large dataset of 1982 patients from multiple institutions, demonstrating high accuracy and strong generalization capabilities across internal and external test sets. The MMDLM not only provides accurate, non-invasive glioma classification but also offers prognostic value that can enhance clinical decision-making and facilitate personalized treatment planning.

## Workflow

![Preprocessed MRI Image](figures/Fig1.png)
*<p align="center"><b>Figure 1.</b> Study design and patient flowchart for model development and validation.</p>*

![MODEL STRUCTURE](figures/Fig2.png)
*<p align="center"><b>Figure 2.</b> Overview of the Multi-Sequence Multitask Deep Learning Model (MMDLM).</p>*

The model architecture consists of two main stages:
1.  **Feature Extraction**: Three separate ResNet-34 models, pre-trained on T2, FLAIR, and CET1 sequences, act as feature extractors.
2.  **Prediction Head**: A gating-based attention mechanism performs multiple instance learning (MIL) on the extracted features. The outputs are then concatenated and passed through a fully connected layer to generate final predictions for the three tasks (Grade, IDH, and 1p/19q).

## Repository Structure

```
.
├── datasets/
│   └── dataset.py            # PyTorch Dataset classes for data loading
├── extract_features/
│   └── extract_features.py   # Script to extract deep features using trained ResNet models
├── model/
│   └── model.py              # Defines the neural network architectures (ResNet, Attention, Fusion)
├── train_utils/
│   └── train.py              # Core training and validation loops
├── utils/
│   ├── calculate.py          # Utility functions for metrics calculation and plotting
│   ├── h5py.py               # Script to create HDF5 datasets from NIfTI files
│   └── resampling.py         # Script for pre-processing and resampling NIfTI images
├── extract_feature.py        # Main script to run feature extraction
├── train_attention.py        # Main script to train the attention-based MIL model
├── train_resnet.py           # Main script to train the initial ResNet feature extractors
├── validation.ipynb          # Jupyter notebook to compute final results and generate figures
├── results/                  # Directory for model outputs, tables, and plots
└── figures/                  # Directory for figures used in the paper and this README
```

## Getting Started

### 1. Installation

First, clone the repository and navigate to the project directory:
```bash
git clone https://github.com/NannanXuesuper/End-to-End-MRI-Based-Model-for-Prediction-of-Molecular-Subtypes-and-Grades.git
cd End-to-End-MRI-Based-Model-for-Prediction-of-Molecular-Subtypes-and-Grades
```

Next, create and activate the Conda environment using the provided file:
```bash
conda env create -f environment.yml -n glioma_prediction
conda activate glioma_prediction
```

### 2. Data Preparation

The data preparation process involves two steps:
1.  **Resampling**: Use `utils/resampling.py` to standardize the slice thickness of your raw NIfTI files (e.g., to 2mm).
2.  **HDF5 Conversion**: Use `utils/h5py.py` to package the resampled NIfTI images into HDF5 files. This script also handles data splitting (train/validation/test) based on a metadata CSV file. The CSV file should contain patient IDs and their corresponding labels (Grade, IDH, 1p/19q).

### 3. Training Pipeline

The model is trained in three stages:

**Stage 1: Train ResNet Feature Extractors**
Run `train_resnet.py` to train the ResNet-34 models on individual MRI slices. This will produce separate models for each prediction task and MRI sequence. The best-performing model weights are saved automatically.
```bash
python train_resnet.py
```

**Stage 2: Extract Deep Features**
After training the ResNet models, use `extract_feature.py` to generate 512-dimensional feature vectors from the penultimate layer of each model. These features are saved into new HDF5 files for the next stage.
```bash
python extract_feature.py
```

**Stage 3: Train the Attention and Fusion Model**
Finally, run `train_attention.py` to train the gated attention and fusion model on the extracted features. This script performs 5-fold cross-validation to determine the best models for predicting Grade, IDH, and 1p/19q status.
```bash
python train_attention.py
```

## Evaluation

To evaluate the model's performance and reproduce the results from our paper, use the `validation.ipynb` notebook. This notebook will load the trained models, run inference, and save the final metrics and visualizations to the `results/` directory.

## Citation

If you find this code useful in your research, please consider citing our paper:

```
[Paper citation will be added here upon publication]
```

## Contact

For questions or issues, please open a GitHub Issue or contact the corresponding author at [fccliujq@zzu.edu.cn](mailto:fccliujq@zzu.edu.cn), [olga.sukocheva@sa.gov.au](mailto:olga.sukocheva@sa.gov.au), [fccfanrt@zzu.edu.cn](mailto:fccfanrt@zzu.edu.cn), or [nannanxue@163.com](mailto:nannanxue@163.com).

