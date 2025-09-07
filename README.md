# Predicting-Insurance-Type-from-Normal-Chest-Xrays
## Introduction
This is the official repository of the ML4H 2025 finding paper (under submission): "The Unawareness of AI Looking for Health Insurance Type from Normal Chest X-ray Images".

## Abstract
Due to the broad use of artificial intelligence in healthcare settings, one should be more cautious about the spurious correlation that has been learned by the model. In this study, we focused on the health insurance type feature, which is highly correlated to patients' socioeconomic status, hidden in the chest X-ray images. We demonstrated that common deep vision models are able to learn insurance type information unperceivable to humans in the subtle textures of the medical images. The result serves as a calling to re-examine the current trained chest X-ray classifiers and ensure that they treat economically different populations equally.

## Installation

### Environment
```
pip install requirements.txt
```
### Rebuild the dataset
Due to the privacy policies of both MIMIC and CheXpert datasets, we are not allowed to provide our parsed dataset, but we provide our train/val/test patient ids ("MIMIC_split.pickle" and "CheXpert_split.pkl") in both MIMIC and CheXpert datasets to replicate our reults.

#### MIMIC
Make sure you include column names: "dicom_id", "subject_id_x", "study_id", "new_insurance_type", "gender", "anchor_age", "race" in your parsed csv. You could find all those corresponding data in the MIMIC official website as long as you signed up for the privacy agreement.

#### CheXpert
Make sure you include at least these keys in your parsed tfrecord file: "jpg_bytes" for CXR, "insurance_type", "age", "sex", "race". You could find all those corresponding data in the CheXpert official website as long as you signed up for the privacy agreement.

### Experiment 1: Health Insurance Prediction from CXRs
#### Training
**MIMIC**
```
python run_exp1.py --dataset MIMIC --mode train --train_path TRAIN_PATH --val_path VAL_PATH --experiment_name EXPERIMENT_NAME --weight_dir WEIGHT_DIR
```

**CheXpert**
```
python run_exp1.py --dataset CheXpert --mode train --train_path TRAIN_PATH --val_path VAL_PATH --experiment_name EXPERIMENT_NAME --weight_dir WEIGHT_DIR
```

#### Testing
**MIMIC**
```
python run_exp1.py --dataset MIMIC --mode test --val_path TEST_PATH --experiment_name EXPERIMENT_NAME --weight_dir WEIGHT_DIR
```

**CheXpert**
```
python run_exp1.py --dataset CheXpert --mode test --val_path VAL_PATH --experiment_name EXPERIMENT_NAME --weight_dir WEIGHT_DIR
```
train_path: The train csv/tfrecord dataset file location
val_path: The test csv/tfrecord dataset file location
experiment_name: Name your experiment as you wish
weight_dir: The directory where you saved your weights

### Experiment 2: Localization of insurance information on Xray - Patch-based training
#### Remove-One-Patch 
**Train**
```
python run_exp2.py --mode train --method remove --train_path TRAIN_PATH --val_path VAL_PATH --experiment_name EXPERIMENT_NAME --weight_dir WEIGHT_DIR --idx INDEX
```

**Test**
```
python run_exp2.py --mode test --method remove --train_path TRAIN_PATH --val_path VAL_PATH --experiment_name EXPERIMENT_NAME --weight_dir WEIGHT_DIR --idx INDEX
```

#### Keep-One-Patch 
**Train**
```
python run_exp2.py --mode train --method keep --train_path TRAIN_PATH --val_path VAL_PATH --experiment_name EXPERIMENT_NAME --weight_dir WEIGHT_DIR --idx INDEX
```

**Test**
```
python run_exp2.py --mode test --method keep --train_path TRAIN_PATH --val_path VAL_PATH --experiment_name EXPERIMENT_NAME --weight_dir WEIGHT_DIR --idx INDEX
```
train_path: The train csv dataset file location
val_path: The test csv dataset file location
experiment_name: Name your experiment as you wish
weight_dir: The directory where you saved your weights
idx: Enter a number from 1-9. Each number corresponds to a specific patch in a 3x3 divided image. (from left to right, from up to bottom)

### Experiment 3: Experiments on Demographic Mediators
#### Health insurance type prediction performance across multiple machine learning methods given the combination of age, race, and sex attributes.
Refer to ml_analysis.ipynb file

#### DenseNet121 trained on isolated White people
**Train**
```
python train_insurance_fullimgsize_densenet.py --mode train --method keep --train_path TRAIN_PATH --val_path VAL_PATH --experiment_name EXPERIMENT_NAME --weight_dir WEIGHT_DIR --idx INDEX
```

**Test**
```
python train_insurance_fullimgsize_densenet.py --mode test --method keep --train_path TRAIN_PATH --val_path VAL_PATH --experiment_name EXPERIMENT_NAME --weight_dir WEIGHT_DIR --idx INDEX
```
train_path: The train csv dataset file location (Only with White people)
val_path: The test csv dataset file location (Only with White people)
experiment_name: Name your experiment as you wish
weight_dir: The directory where you saved your weights

## Code References
MedMamba: [Link](https://github.com/YubiaoYue/MedMamba)
SwinTransformer V2: [Link](https://github.com/ChristophReich1996/Swin-Transformer-V2)













