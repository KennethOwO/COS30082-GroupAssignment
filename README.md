# 📑 Table of Contents

- [📁 AML_Resnet (Darren)](#-aml_resnet-darren)
  - [🌿 Mix-Stream CNN Baselines for Plant Species Classification](#-mix-stream-cnn-baselines-for-plant-species-classification)
  - [📁 Included Notebook](#-included-notebook)
  - [🧠 Models](#-models)
  - [🔧 Training Type](#-training-type)
  - [🧩 Classifier: Mix-Stream Head](#-classifier-mix-stream-head)
  - [📘 Description](#-description)
  - [🔧 Mix-Stream Classifier Architecture](#-mix-stream-classifier-architecture)
  - [🛠️ Training Flow](#️-training-flow)
  - [📊 Evaluation Output](#-evaluation-output)
  - [🛠️ Data Preprocessing](#-data-preprocessing)
  - [🧪 Training Summary](#-training-summary)

- [📁 Dinov2withTripletLoss (Eddie Pui)](#-dinov2withtripletloss-eddie-pui)
  - [🌿 Dinov2 with TripletLoss Herbarium–Field Cross-Domain Classification](#-dinov2-with-tripletloss-herbariumfield-cross-domain-classification)
  - [📁 Included Notebook](#-included-notebook-1)
  - [🛠️ Training Summary](#️-training-summary-1)

- [📁 DINOv2 Baseline (Xuan Yong)](#-dinov2-baseline-xuan-yong)
  - [🌿 DINOv2 Self-Supervised Baseline Experiments](#-dinov2-self-supervised-baseline-experiments)
  - [📁 Included Notebooks](#-included-notebooks)
    - [Baseline A](#1-baselinea_dinov2_feature_extractoripynb)
    - [Baseline B](#2-baselineb_dinov2_coloraugmentation_splitdataipynb)
    - [Baseline C](#3-baselinec_dinov2_nocoloraugmentation_nosplitipynb)
    - [Baseline D](#4-baselined_dinov2_finetuning_last2layersipynb)

- [📁 DINOv2 New Approach (Xuan Yong)](#-dinov2-new-approach-xuan-yong)
  - [🚀 Two-Stage Metric Learning (Triplet Loss + Classifier)](#-two-stage-metric-learning-triplet-loss--classifier)
  - [📁 Included Notebooks](#-included-notebooks-1)
    - [New Approach A](#1-dinov2_triplet_hf_1ipynb)
    - [New Approach B](#2-dinov2_triplet_f_1ipynb)
    - [New Approach C](#3-dinov2_triplet_hf_2ipynb)
    - [New Approach D](#4-dinov2_triplet_hf_4ipynb)
    - [New Approach E](#5-dinov2_triplet_hf_5ipynb)
    - [New Approach F](#6-dinov2_triplet_hf_3ipynb)
  - [📊 Overall Validation Accuracy Summary](#-overall-validation-accuracy-summary)

- [📁 MAE (TengYong & Voong)](#-mae-tengyong--voong)
  - [🌿 MAE Herbarium–Field Cross-Domain Classification](#-mae-herbariumfield-cross-domain-classification)
  - [📁 Included Notebooks](#-included-notebooks-2)
    - [MAE Base (End-to-End)](#1-mae_bipynb)
    - [MAE Base (Frozen)](#2-mae_freeze_bipynb)
    - [MAE Large (End-to-End)](#3-mae_lipynb)
    - [MAE Large (Frozen)](#4-mae_freeze_lipynb)
  - [🔧 Toggle Data Augmentation](#-toggle-data-augmentation)
  - [🛠️ Training Summary](#️-training-summary-2)
  - [📂 Runs Output Structure](#-runs-output-structure)
  - [✔ File Descriptions](#-file-descriptions)
    - [training_log.txt](#training_logtxt)
    - [classification_report.txt](#classification_reporttxt)
    - [confusion_matrix.png](#confusion_matrixpng)
    - [per_class_metrics.csv](#per_class_metricscsv)
    - [val_with_without_pairs_results.txt](#val_with_without_pairs_resultstxt)

---

# 📁 **AML_Resnet (Darren)**

## 🌿 **Mix-Stream CNN Baselines for Plant Species Classification**

This repository contains a single notebook that trains two CNN baselines —  
**ResNet50** and **EfficientNet-B0** — using a shared **Mix-Stream classifier**.  
The notebook handles data loading, preprocessing, training, fine-tuning, and evaluation on the herbarium dataset.


## 📁 **Included Notebook**
1. **AML_Resnet.ipynb**



## 🧠 **Models**
- **ResNet50** (ImageNet)
- **EfficientNet-B0** (ImageNet)



## 🔧 **Training Type**
- Two-stage training:  
  1. **Frozen backbone**  
  2. **Fine-tune last 120 layers**



## 🧩 **Classifier: Mix-Stream Head**
- Dense 512  
- Dense 256  


## 📘 **Description**
This notebook:
- Loads the dataset  
- Preprocesses images  
- Builds a `tf.data` pipeline  
- Trains two CNN backbones using the same architecture  
- Stage 1: Freeze backbone, train Mix-Stream head  
- Stage 2: Unfreeze last 120 layers for fine-tuning  

Outputs include:
- Final Top-1 / Top-5 accuracy  
- Loss  
- Confusion matrices  
- Pair / No-pair accuracy  



## 🔧 **Mix-Stream Classifier Architecture**

### **Branch A**
- Dense(512, ReLU)  
- Dropout(0.5)

### **Branch B**
- Dense(256, ReLU)  
- Dropout(0.3)

**Merged → Dense(NUM_CLASSES, softmax)**


## 🛠️ **Training Flow**

### **Stage 1 — Frozen Backbone**
- Backbone trainable = False  
- Optimizer: **Adam(1e-3)**  
- Epochs: **20**  
- Only Mix-Stream head learns

### **Stage 2 — Fine-tuning**
- Unfreeze **last 120 layers**  
- Optimizer: **Adam(1e-4)**  
- Epochs: **10**  
- Backbone adapts to plant features



## 📊 **Evaluation Output**
Each model reports:

- Loss  
- Top-1 accuracy  
- Top-5 accuracy  
- Pair accuracy (from `class_with_pairs.txt`)  
- No-pair accuracy  
- Confusion matrix (numbers only for readability)  
- Example predictions on test images  



## 🛠️ **Data Preprocessing**

ImageNet preprocessing:
- **ResNet50** → `resnet.preprocess_input`  
- **EfficientNet-B0** → `efficientnet.preprocess_input`

Augmentations applied:
- Random horizontal flip  
- Random brightness  
- Resize to **224×224**  
- Batched via `tf.data` with prefetching  



## 🧪 **Training Summary**

| Notebook           | Model              | Frozen Stage      | Fine-Tune Stage                        | Classifier   |
|--------------------|--------------------|-------------------|-----------------------------------------|--------------|
| AML_Resnet.ipynb   | EfficientNet-B0    | Yes (20 epochs)   | Unfreeze last 120 layers (10 epochs)    | Mix-Stream   |
| AML_Resnet.ipynb   | ResNet50           | Yes (20 epochs)   | Unfreeze last 120 layers (10 epochs)    | Mix-Stream   |



# 📁 **Dinov2withTripletLoss (Eddie Pui)**

## 🌿 **Dinov2 with TripletLoss Herbarium–Field Cross-Domain Classification**

Repository for **Dinov2 with Triplet Loss** training with **Multi Layer Perceptron (MLP)**.  
This repository contains **1 training pipeline** built using **DINOv2** for **cross-domain plant species classification (Herbarium → Field)**.

## 📁 **Included Notebook**

### **Dinov2 New Approach (Eddie Pui).ipynb**

- **Model:** Dinov2 ViT-Base  
- **Training Type:**  
  - **Stage 1:** Unfreeze 2 on Stage 1 fine tuning  
  - **Stage 2:** Freeze all layers + attach an MLP  

### **MLP Architecture**
- **BatchNorm**  
- **Linear(768 → 256)**  
- **ReLU**  
- **Linear(256 → 100)**  

### **Augmentations**
- RandomResizedCrop  
- Horizontal/Vertical Flip  
- Rotation  

## 🛠️ **Training Summary**

| Setting | Details |
|--------|---------|
| **Notebook** | Dinov2_new_approach.ipynb |
| **Model** | DINOv2 ViT-Base |
| **Stage 1 Config** | Unfreeze 2 transformer blocks (partial fine-tuning) |
| **Stage 2 Config** | Freeze entire backbone + attach MLP (BatchNorm → Linear → ReLU → Linear) |
| **Augmentation** | Yes |


# 📁 **DINOv2 Baseline (Xuan Yong)**

## 🌿 **DINOv2 Self-Supervised Baseline Experiments**

This folder contains the source code for the baseline experiments using the **DINOv2 ViT-Base** architecture.  
These experiments aim to establish benchmarks using both **frozen feature extraction** and **supervised fine-tuning** strategies.



## 📁 **Included Notebooks**

### **1. BaselineA_Dinov2_feature_extractor.ipynb**
- **Model:** DINOv2 Baseline A  
- **Strategy:** Feature Extraction (Frozen Backbone)  
- **Classifier:** Support Vector Machine (SVM)  
- **Description:**  
  The DINOv2 backbone is completely frozen.  
  Features are extracted from the **[CLS] token** and classified using an SVM to evaluate raw self-supervised feature quality.


### **2. BaselineB_Dinov2_colorAugmentation_SplitData.ipynb**
- **Model:** DINOv2 Baseline B  
- **Strategy:** Supervised Fine-Tuning  
- **Augmentation:** Heavy (includes ColorJitter)  
- **Split:** Train/Validation Split  
- **Description:**  
  Trains a Linear Head on top of the frozen backbone using heavy augmentation to test robustness against color variations.



### **3. BaselineC_Dinov2_NoColorAugmentation_NoSplit.ipynb**
- **Model:** DINOv2 Baseline C  
- **Strategy:** Supervised Fine-Tuning  
- **Augmentation:** Simplified (No ColorJitter)  
- **Split:** No Split (Uses All Data)  
- **Description:**  
  Removes ColorJitter for simpler augmentation and trains on the **full dataset**.  
  This model achieved the **best performance** among all baselines.



### **4. BaselineD_Dinov2_FineTuning_Last2Layers.ipynb**
- **Model:** DINOv2 Baseline D  
- **Strategy:** Partial Unfreezing  
- **Augmentation:** Heavy  
- **Description:**  
  Unfreezes the **last 2 Transformer blocks** to allow limited adaptation to the plant domain.



# 📁 **DINOv2 New Approach (Xuan Yong)**

## 🚀 **Two-Stage Metric Learning (Triplet Loss + Classifier)**

These experiments follow a **two-stage training pipeline**:

1. **Metric Learning** using Triplet Loss to align Herbarium & Field embeddings  
2. **Classifier Training** on aligned features  


## 📁 **Included Notebooks**

### **1. DINOv2_Triplet_H+F_1.ipynb**
- **Model:** New Approach A (Best Model)  
- **Margin:** 0.5  
- **Unfreezing:** ALL backbone layers  
- **Description:**  
  Full-backbone adaptation using Triplet Loss over Herbarium + Field images.  
  Achieved the **highest cross-domain (“Without Pair”) accuracy**.


### **2. DINOv2_Triplet_F_1.ipynb**
- **Model:** New Approach B  
- **Margin:** 0.2  
- **Data:** Field images only  
- **Description:**  
  Triplet Loss trained using only Field images.  
  Resulted in **weak domain alignment**.


### **3. DINOv2_Triplet_H+F_2.ipynb**
- **Model:** New Approach C  
- **Margin:** 0.5  
- **Unfreezing:** Last 2 Layers Only  
- **Description:**  
  Strict margin with minimal backbone updates.  
  Balanced performance but limited adaptation.


### **4. DINOv2_Triplet_H+F_4.ipynb**
- **Model:** New Approach D  
- **Margin:** 0.2  
- **Unfreezing:** Last 2 Layers Only  
- **Description:**  
  Uses a looser margin (0.2) and lightly adapts the last two layers during both Triplet and Classifier stages.



### **5. DINOv2_Triplet_H+F_5.ipynb**
- **Model:** New Approach E  
- **Margin:** 0.2  
- **Classifier:** Frozen  
- **Description:**  
  Triplet Loss (Margin 0.2) trains embeddings, but classifier stage **does not update the backbone**.



### **6. DINOv2_Triplet_H+F_3.ipynb**
- **Model:** New Approach F  
- **Margin:** 0.5  
- **Unfreezing:** Last 2 Layers (Both Stages)  
- **Description:**  
  Last two layers updated in **both** Triplet Loss and classifier stages for moderate adaptation.

---

# 📊 **Overall Validation Accuracy Summary**

The **Overall Validation Accuracy.doc** file summarises all DINOv2 model performances on the Test Set.

**"Without Pair" accuracy is the most important metric**  
→ It measures **true cross-domain generalisation**.

---

# 📁 **MAE (TengYong & Voong)**

## 🌿 **MAE Herbarium–Field Cross-Domain Classification**

Repository for **MAE-Base** & **MAE-Large** Experiments (**End-to-End** & **Frozen**)  
This repository contains four training pipelines built using **Masked Autoencoder (MAE)** Vision Transformers for **cross-domain plant species classification** (Herbarium → Field).  
All notebooks follow a unified structure and differ only in model size and training strategy.


## 📁 **Included Notebooks**



### **1. MAE_B.ipynb**
- **Model:** MAE-Base  
- **Training Type:** End-to-End fine-tuning (entire MAE backbone is trainable)  
- **Augmentation:** Controllable (True/False)  

**Description:**  
Trains the full MAE-Base encoder on herbarium + field images with optional augmentations.


### **2. MAE_freeze_B.ipynb**
- **Model:** MAE-Base  
- **Training Type:** Frozen backbone  
- **Augmentation:** Controllable (True/False)

**Description:**  
The MAE-Base encoder is frozen, and only the final classification head is trained.



### **3. MAE_L.ipynb**
- **Model:** MAE-Large  
- **Training Type:** End-to-End fine-tuning  
- **Augmentation:** Controllable (True/False)

**Description:**  
Trains the full large-scale MAE encoder for stronger cross-domain generalisation.



### **4. MAE_freeze_L.ipynb**
- **Model:** MAE-Large  
- **Training Type:** Frozen backbone  
- **Augmentation:** Controllable (True/False)

**Description:**  
Uses MAE-Large as a feature extractor, training only the classification head.



## 🔧 **Toggle Data Augmentation**

All notebooks contain a variable in **Section 4: Transform**:
```USE_AUG = True or False```


### **What It Does**
- **True** → Heavy/medium augmentation  
- **False** → Only center-crop + normalization  

### **Why It Matters**
Augmentation strongly affects cross-domain performance.

#### Quick Summary

| Setting           | Description |
|------------------|-------------|
| **USE_AUG = True**  | Tests whether augmentation improves Field generalisation |
| **USE_AUG = False** | Tests pure model capability without augmentation |



## 🛠️ **Training Summary**

| Notebook              | Model     | Frozen | Augmentation Toggle |
|-----------------------|-----------|--------|----------------------|
| MAE_B.ipynb           | MAE-Base  | No     | USE_AUG             |
| MAE_freeze_B.ipynb    | MAE-Base  | Yes    | USE_AUG             |
| MAE_L.ipynb           | MAE-Large | No     | USE_AUG             |
| MAE_freeze_L.ipynb    | MAE-Large | Yes    | USE_AUG             |



## 📂 **Runs Output Structure**

Each run directory contains:

### **/logs/**
- `training_log.txt`

### **/evaluation/**
- `classification_report.txt`  
- `confusion_matrix.png`  
- `per_class_metrics.csv`  
- `val_with_without_pairs_results.txt`



## ✔ **File Descriptions**

### **training_log.txt**
- Train/validation accuracy  
- Train/validation loss  
- Best validation accuracy  



### **classification_report.txt**
Contains:
- Precision  
- Recall  
- F1-score  
- Support  
- Micro / Macro / Weighted averages  



### **confusion_matrix.png**
Visual heatmap:
- True labels (rows)  
- Predicted labels (columns)



### **per_class_metrics.csv**
Per-class performance:
- Precision  
- Recall  
- F1-score  
- Support  



### **val_with_without_pairs_results.txt**
Final test accuracy on:
- **With-Pair** classes: classes that have Herbarium + Field pairs
- **Without-Pair** classes: unseen classes without pairs



