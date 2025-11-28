------------------
📁MAE(TengYong&Raymond)
------------------
🌿 MAE Herbarium–Field Cross-Domain Classification
Repository for MAE-Base & MAE-Large Experiments (End-to-End & Frozen)
This repository contains four training pipelines built using Masked Autoencoder (MAE) Vision Transformers for cross-domain plant species classification (Herbarium → Field). All notebooks follow a unified structure and differ only in model size and training strategy.

📁 Included Notebooks
1. MAE_B.ipynb
* Model: MAE-Base
* Training Type: End-to-End fine-tuning (entire MAE backbone is trainable)
* Augmentation: Controllable (True/False)
Description:
This notebook trains the full MAE-Base encoder on herbarium + field images with optional augmentations.

2. MAE_freeze_B.ipynb
* Model: MAE-Base
* Training Type: Frozen backbone
* Augmentation: Controllable (True/False)
Description:
The MAE-Base encoder is frozen, and only the final classification head is trained.

3. MAE_L.ipynb
* Model: MAE-Large
* Training Type: End-to-End fine-tuning
* Augmentation: Controllable (True/False)
Description:
Trains the full large-scale MAE encoder for stronger cross-domain generalisation.

4. MAE_freeze_L.ipynb
* Model: MAE-Large
* Training Type: Frozen backbone
* Augmentation: Controllable (True/False)
Description:
Uses MAE-Large as a feature extractor, training only the classification head for efficient training.

🔧 Toggle Data Augmentation

All four notebooks contain a variable inside Section 4: Transform:
* USE_AUG = True or False

What it does
* True → use heavy/medium augmentation
* RandomResizedCrop
* ColorJitter
* Horizontal/Vertical flip
* Rotation
* False → use only center-crop + normalization

Why it matters
* Augmentation strongly affects cross-domain learning.

You can easily compare:
Setting         : Description
---------------------------------------------------------------
USE_AUG = True  : Tests whether augmentation helps the model
                   generalise to Field photos
USE_AUG = False : Tests pure model capability without
                   augmentation

🛠️ Training Summary
Notebook              | Model     | Frozen | Augmentation Toggle
-----------------------------------------------------------------
MAE_B.ipynb           | MAE-Base  | No     | USE_AUG
MAE_freeze_B.ipynb    | MAE-Base  | Yes    | USE_AUG
MAE_L.ipynb           | MAE-Large | No     | USE_AUG
MAE_freeze_L.ipynb    | MAE-Large | Yes    | USE_AUG

📂 Runs Output Structure
This project generates six experiment runs, each corresponding to one MAE model configuration (Base/Large × Frozen/End-to-End × With/Without Augmentation).
All runs share the same internal folder structure and evaluation artefacts.
* runs_mae_base_AUG
* runs_mae_base_NOAUG
* runs_mae_freeze_base_AUG
* runs_mae_freeze_base_NOAUG
* runs_mae_large_AUG
* runs_mae_freeze_large_AUG

Each run represents a unique experimental setup based on:
* MAE model size (Base or Large)
* Training strategy (End-to-End or Frozen)
* Data preparation (AUG or NOAUG)

Each run contains two main directories:
/logs
	/training_log.txt
/evaluation
	/classification_report.txt
	/confusion_matrix.png
	/per_class_metrics.csv
	/val_with_without_pairs_results.txt

✔ training_log.txt
* Train accuracy and loss for every epoch
* Validation accuracy and loss
* Best validation accuracy

✔ classification_report.txt
Contains model performance metrics:
* Precision
* Recall
* F1-score
* Support
* Micro / Macro / Weighted averages

✔ confusion_matrix.png
A heatmap visualising:
* True labels (rows)
* Predicted labels (columns)
* Useful for identifying misclassification patterns.

✔ per_class_metrics.csv
A CSV file listing metrics for each class:
* Per-class precision / recall / F1
* Number of samples
* Useful for analysing long-tail performance

✔ val_with_without_pairs_results.txt
This file reports the final testing accuracy on two splits:
* With-Pair → classes that have Herbarium + Field pairs
* Without-Pair → unseen classes without pairs

