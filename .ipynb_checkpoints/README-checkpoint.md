# Predicting-Insurance-Type-from-Normal-Chest-Xrays
## Introduction
This is the official repository of the ML4H 2025 finding paper (under submission): "The Unawareness of AI Looking for Health Insurance Type from Normal Chest X-ray Images".

### Abstract
Due to the broad use of artificial intelligence in healthcare settings, one should be more cautious about the spurious correlation that has been learned by the model. In this study, we focused on the health insurance type feature, which is highly correlated to patients' socioeconomic status, hidden in the chest X-ray images. We demonstrated that common deep vision models are able to learn insurance type information unperceivable to humans in the subtle textures of the medical images. The result serves as a calling to re-examine the current trained chest X-ray classifiers and ensure that they treat economically different populations equally.

## Installation

### Environment
```
pip install requirements.txt
```
### Rebuild the dataset
Due to the privacy policies of both MIMIC and CheXpert datasets, we are not allowed to provide our parsed dataset, but we provide our train/val/test patient ids ("MIMIC_split.pickle" and "CheXpert_split.pkl") in both MIMIC and CheXpert datasets to replicate our reults.

### Experiment 1: Health Insurance Prediction from CXRs
#### Training
##### MIMIC
```
python run_exp1.py --dataset MIMIC --mode train --train_path TRAIN_PATH --val_path VAL_PATH --experiment_name EXPERIMENT_NAME --weight_dir WEIGHT_DIR
```
train_path: the train csv dataset file location
val_path: the val csv dataset file location
experiment_name: name your experiment as you wish
weight_dir: the directory where you saved your weights

##### CheXpert
```
python run_exp1.py --dataset CheXpert --mode train --train_path TRAIN_PATH --val_path VAL_PATH --experiment_name EXPERIMENT_NAME --weight_dir WEIGHT_DIR
```
train_path: the train csv dataset file location
val_path: the val csv dataset file location
experiment_name: name your experiment as you wish
weight_dir: the directory where you saved your weights








