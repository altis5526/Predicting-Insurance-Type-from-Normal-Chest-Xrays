# Predicting-Insurance-Type-from-Normal-Chest-Xrays
## Introduction
This is the official repository of the MIDL 2026 paper (under review): "Algorithms Trained on Normal Chest X-rays Can Predict
Health Insurance Types".

## Abstract
Artificial intelligence is revealing what medicine never intended to encode. Deep vision models, trained on chest X-rays, can now detect not only disease but also invisible traces of social inequality. In this study, we show that state-of-the-art architectures (DenseNet121, SwinV2-B*, MedMamba) can predict a patient’s health insurance type, a strong proxy for socioeconomic status, from normal chest X-rays with significant accuracy (AUC $\approx$ 0.70 on MIMIC-CXR-JPG, 0.68 on CheXpert). The signal was unlikely contributed by demographic features by our machine learning study combining age, race, and sex labels to predict health insurance types. The signal also remains detectable when the model is trained exclusively on a single racial group. Patch-based occlusion reveals that the signal is diffuse rather than localized, embedded in the upper and mid-thoracic regions. This suggests that deep networks may be internalizing subtle traces of clinical environments, equipment differences, or care pathways; learning socioeconomic segregation itself. These findings challenge the assumption that medical images are neutral biological data. By uncovering how models perceive and exploit these hidden social signatures, this work reframes fairness in medical AI: the goal is no longer only to balance datasets or adjust thresholds, but to interrogate and disentangle the social fingerprints embedded in clinical data itself. 

## Installation

### Environment
```
pip install requirements.txt
```
### Rebuild the dataset
Due to the privacy policies of both MIMIC and CheXpert datasets, we are not allowed to provide our parsed dataset, but we provide our train/val/test patient ids ("MIMIC_split.pickle" and "Exclude_support_device_train_test_split_CheXpert.pkl") in both MIMIC and CheXpert datasets to replicate our reults.

#### MIMIC
Make sure you include column names: "dicom_id", "subject_id_x", "study_id", "new_insurance_type", "gender", "anchor_age", "race" in your parsed csv. You could find all those corresponding data in the MIMIC official website as long as you signed up for the privacy agreement.

#### CheXpert
Make sure you include at least these keys in your parsed tfrecord file: "jpg_bytes" for CXR, "insurance_type", "age", "sex", "race". You could find all those corresponding data in the CheXpert official website as long as you signed up for the privacy agreement.

### Experiment 1: Health Insurance Prediction from CXRs
**MIMIC**
```
python run_exp0.py --dataset MIMIC --model MODEL --train_path TRAIN_PATH --val_path VAL_PATH --test_path TEST_PATH --experiment_name NAME --weight_dir WEIGHT_DIR
```

**CheXpert**
```
python run_exp0.py --dataset CheXpert --model MODEL --train_path TRAIN_PATH --val_path VAL_PATH --test_path TEST_PATH --experiment_name NAME --weight_dir WEIGHT_DIR
```

**MIMIC_Random**
```
python run_exp0-1.py --dataset CheXpert --model MODEL --train_path TRAIN_PATH --val_path VAL_PATH --test_path TEST_PATH --experiment_name NAME --weight_dir WEIGHT_DIR
```
model: choose one of the models: "densenet", "swinTF", "mamba"
train_path: The train csv/tfrecord dataset file location
val_path: The test csv/tfrecord dataset file location
experiment_name: Name your experiment as you wish
weight_dir: The directory where you saved your weights

### Experiment 2: Localization of insurance information on Xray - Patch-based training
#### Remove-One-Patch 
```
python run_exp1.py --method remove --train_path TRAIN_PATH --val_path VAL_PATH --test_path TEST_PATH --experiment_name NAME --weight_dir WEIGHT_DIR
```

#### Keep-One-Patch 
```
python run_exp1.py --method keep --train_path TRAIN_PATH --val_path VAL_PATH --test_path TEST_PATH --experiment_name NAME --weight_dir WEIGHT_DIR
```

train_path: The train csv dataset file location
val_path: The test csv dataset file location
experiment_name: Name your experiment as you wish
weight_dir: The directory where you saved your weights

### Experiment 3: Experiments on Demographic Mediators
#### Health insurance type prediction performance across multiple machine learning methods given the combination of age, race, and sex attributes.
Refer to ml_analysis.ipynb file

#### DenseNet121 trained on isolated White people
Basically the same as experiment 1, but remember to change your train_path, val_path, and test_path to the curated isolated White people dataset.

**MIMIC**
```
python run_exp0.py --dataset MIMIC --model MODEL --train_path TRAIN_PATH --val_path VAL_PATH --test_path TEST_PATH --experiment_name NAME --weight_dir WEIGHT_DIR
```

**CheXpert**
```
python run_exp0.py --dataset CheXpert --model MODEL --train_path TRAIN_PATH --val_path VAL_PATH --test_path TEST_PATH --experiment_name NAME --weight_dir WEIGHT_DIR
```

train_path: The train csv dataset file location (Only with White people)
val_path: The test csv dataset file location (Only with White people)
experiment_name: Name your experiment as you wish
weight_dir: The directory where you saved your weights

## Code References
MedMamba: [Link](https://github.com/YubiaoYue/MedMamba)

SwinTransformer V2: [Link](https://github.com/ChristophReich1996/Swin-Transformer-V2)













