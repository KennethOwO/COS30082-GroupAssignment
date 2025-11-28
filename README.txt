# 📁 **MAE (TengYong & Raymond)**

## 🌿 **MAE Herbarium–Field Cross-Domain Classification**

Repository for **MAE-Base** & **MAE-Large** Experiments (**End-to-End** & **Frozen**)  
This repository contains four training pipelines built using **Masked Autoencoder (MAE)** Vision Transformers for **cross-domain plant species classification** (Herbarium → Field).  
All notebooks follow a unified structure and differ only in model size and training strategy.

---

## 📁 **Included Notebooks**

---

### **1. MAE_B.ipynb**
- **Model:** MAE-Base  
- **Training Type:** End-to-End fine-tuning (entire MAE backbone is trainable)  
- **Augmentation:** Controllable (True/False)  

**Description:**  
Trains the full MAE-Base encoder on herbarium + field images with optional augmentations.

---

### **2. MAE_freeze_B.ipynb**
- **Model:** MAE-Base  
- **Training Type:** Frozen backbone  
- **Augmentation:** Controllable (True/False)

**Description:**  
The MAE-Base encoder is frozen, and only the final classification head is trained.

---

### **3. MAE_L.ipynb**
- **Model:** MAE-Large  
- **Training Type:** End-to-End fine-tuning  
- **Augmentation:** Controllable (True/False)

**Description:**  
Trains the full large-scale MAE encoder for stronger cross-domain generalisation.

---

### **4. MAE_freeze_L.ipynb**
- **Model:** MAE-Large  
- **Training Type:** Frozen backbone  
- **Augmentation:** Controllable (True/False)

**Description:**  
Uses MAE-Large as a feature extractor, training only the classification head.

---

## 🔧 **Toggle Data Augmentation**

All notebooks contain a variable in **Section 4: Transform**:
- USE_AUG = True or False


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

---

## 🛠️ **Training Summary**

| Notebook              | Model     | Frozen | Augmentation Toggle |
|-----------------------|-----------|--------|----------------------|
| MAE_B.ipynb           | MAE-Base  | No     | USE_AUG             |
| MAE_freeze_B.ipynb    | MAE-Base  | Yes    | USE_AUG             |
| MAE_L.ipynb           | MAE-Large | No     | USE_AUG             |
| MAE_freeze_L.ipynb    | MAE-Large | Yes    | USE_AUG             |

---

## 📂 **Runs Output Structure**

Each run directory contains:

### **/logs/**
- `training_log.txt`

### **/evaluation/**
- `classification_report.txt`  
- `confusion_matrix.png`  
- `per_class_metrics.csv`  
- `val_with_without_pairs_results.txt`

---

## ✔ **File Descriptions**

### **training_log.txt**
- Train/validation accuracy  
- Train/validation loss  
- Best validation accuracy  

---

### **classification_report.txt**
Contains:
- Precision  
- Recall  
- F1-score  
- Support  
- Micro / Macro / Weighted averages  

---

### **confusion_matrix.png**
Visual heatmap:
- True labels (rows)  
- Predicted labels (columns)

---

### **per_class_metrics.csv**
Per-class performance:
- Precision  
- Recall  
- F1-score  
- Support  

---

### **val_with_without_pairs_results.txt**
Final test accuracy on:
- **With-Pair** classes: classes that have Herbarium + Field pairs
- **Without-Pair** classes: unseen classes without pairs


