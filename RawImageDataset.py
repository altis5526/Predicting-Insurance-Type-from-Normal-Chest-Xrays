import torch
from tfrecord.torch.dataset import TFRecordDataset
from torch.utils.data import DataLoader
import cv2
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
import random

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from PIL import Image
import csv
import random
import os
import pandas as pd
import cv2
from torchvision import transforms
from transformers import AutoTokenizer, BioGptModel
from low_pass_func import low_pass, high_pass

class MIMIC_raw(Dataset):
    def __init__(self, dataset_csv, resize=448, transform=True):
        with open(dataset_csv , newline='') as csvfile:
            data = list(csv.reader(csvfile))
            self.title = data[0]
            self.rows = data[1::]

        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            # transforms.RandomResizedCrop(size=448),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        self.resize = resize
        self.transform = transform
    
    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        img_root_dir = "/mnt/new_usb/jupyter-altis5526/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"
        img_dir = f"p{self.rows[idx][1][:2]}/p{self.rows[idx][1]}/s{self.rows[idx][2]}/{self.rows[idx][0]}.jpg"
        img_dir = img_root_dir + img_dir
        full_img = Image.open(img_dir).convert('RGB')
        full_img = full_img.resize((self.resize,self.resize))
        full_img = np.transpose(full_img, (2, 0, 1))/255.0
        full_img = torch.from_numpy(full_img.copy()).float()
        if self.transform:
            full_img = self.augmentation(full_img)

        insurance_index = self.title.index("new_insurance_type")
        insurance_type = self.rows[idx][insurance_index]
        gender_index = self.title.index("gender")
        gender = self.rows[idx][gender_index]
        age_index = self.title.index("anchor_age")
        age = float(self.rows[idx][age_index])
        race_index = self.title.index("race")
        race = self.rows[idx][race_index]
        
        if insurance_type == "Private":
            output_insurance = torch.Tensor([0., 1.])
        elif insurance_type == "Medicaid" or insurance_type == "Medicare":
            output_insurance = torch.Tensor([1., 0.])

        if gender == "M":
            gender = torch.Tensor([1., 0.])
        elif gender == "F":
            gender = torch.Tensor([0., 1.])

        if age < 40:
            age = torch.Tensor([1., 0., 0.])
        elif age >= 40 and age < 50:
            age = torch.Tensor([0., 1., 0.])
        elif age >= 50 and age < 65:
            age = torch.Tensor([0., 0., 1.])

        if race == "WHITE":
            race = torch.Tensor([1., 0., 0.])
        elif race == "BLACK":
            race = torch.Tensor([0., 1., 0.])
        else:
            race = torch.Tensor([0., 0., 1.])
        
        output = {'full_img': full_img, 'insurance': output_insurance, 'img_id': self.rows[idx][0], 'gender': gender, 'age': age, 'race': race}
        return output

class MIMIC_raw_ICD(Dataset):
    def __init__(self, dataset_csv):
        with open(dataset_csv , newline='') as csvfile:
            data = list(csv.reader(csvfile))
            self.title = data[0]
            self.rows = data[1::]

        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            # transforms.RandomResizedCrop(size=448),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        # self.tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
        # self.gptmodel = BioGptModel.from_pretrained("microsoft/biogpt")
    
    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        img_root_dir = "/mnt/new_usb/jupyter-altis5526/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"
        img_dir = f"p{self.rows[idx][1][:2]}/p{self.rows[idx][1]}/s{self.rows[idx][2]}/{self.rows[idx][0]}.jpg"
        img_dir = img_root_dir + img_dir
        full_img = Image.open(img_dir).convert('RGB')
        full_img = full_img.resize((448,448))
        full_img = np.transpose(full_img, (2, 0, 1))/255.0
        full_img = torch.from_numpy(full_img.copy()).float()
        full_img = self.augmentation(full_img)
        
        insurance_index = self.title.index("new_insurance_type")
        insurance_type = self.rows[idx][insurance_index]
        gender_index = self.title.index("gender")
        gender = self.rows[idx][gender_index]
        age_index = self.title.index("anchor_age")
        age = float(self.rows[idx][age_index])
        race_index = self.title.index("race")
        race = self.rows[idx][race_index]
        icd_text_index = self.title.index("disease_list")
        icd_text = self.rows[idx][icd_text_index]
        
        if insurance_type == "Private":
            output_insurance = torch.Tensor([0., 1.])
        elif insurance_type == "Medicaid" or insurance_type == "Medicare":
            output_insurance = torch.Tensor([1., 0.])

        if gender == "M":
            gender = 0
        elif gender == "F":
            gender = 1

        if age < 40:
            age = torch.Tensor([1., 0., 0.])
        elif age >= 40 and age < 50:
            age = torch.Tensor([0., 1., 0.])
        elif age >= 50 and age < 65:
            age = torch.Tensor([0., 0., 1.])

        if race == "WHITE":
            race = torch.Tensor([1., 0., 0.])
        elif race == "BLACK":
            race = torch.Tensor([0., 1., 0.])
        else:
            race = torch.Tensor([0., 0., 1.])

        subject_id = self.rows[idx][1]
        icd_embedding_path = "/mnt/new_usb/jupyter-altis5526/icd_tensors/"
        icd_feature = torch.load(icd_embedding_path+subject_id+'.pt')
        
        output = {'img_dir': img_dir, 'full_img': full_img, 'insurance': output_insurance, 'img_id': self.rows[idx][0], 'gender': gender, 'age': age, 'race': race, 'icd_feature': icd_feature}
        return output


class MIMIC_raw_ICD_mask_image(Dataset):
    def __init__(self, dataset_csv, mask_start, mask_size_y, mask_size_x):
        with open(dataset_csv , newline='') as csvfile:
            data = list(csv.reader(csvfile))
            self.title = data[0]
            self.rows = data[1::]

        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            # transforms.RandomResizedCrop(size=448),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])        

        self.start_y = mask_start[0]
        self.start_x = mask_start[1]
        self.mask_size_x = mask_size_x
        self.mask_size_y = mask_size_y
    
    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        img_root_dir = "/mnt/new_usb/jupyter-altis5526/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"
        img_dir = f"p{self.rows[idx][1][:2]}/p{self.rows[idx][1]}/s{self.rows[idx][2]}/{self.rows[idx][0]}.jpg"
        img_dir = img_root_dir + img_dir
        full_img = Image.open(img_dir).convert('RGB')
        full_img = full_img.resize((448,448))
        full_img = np.transpose(full_img, (2, 0, 1))/255.0
        full_img[:, self.start_y:self.start_y+self.mask_size_y, self.start_x:self.start_x+self.mask_size_x] = 0.

        # full_img = full_img * 255.0
        # cv2.imwrite("maskone_image.png", full_img[0])
        
        full_img = torch.from_numpy(full_img.copy()).float()
        full_img = self.augmentation(full_img)

        insurance_index = self.title.index("new_insurance_type")
        insurance_type = self.rows[idx][insurance_index]
        gender_index = self.title.index("gender")
        gender = self.rows[idx][gender_index]
        age_index = self.title.index("anchor_age")
        age = float(self.rows[idx][age_index])
        race_index = self.title.index("race")
        race = self.rows[idx][race_index]
        icd_text_index = self.title.index("disease_list")
        icd_text = self.rows[idx][icd_text_index]
        
        if insurance_type == "Private":
            output_insurance = torch.Tensor([0., 1.])
        elif insurance_type == "Medicaid" or insurance_type == "Medicare":
            output_insurance = torch.Tensor([1., 0.])

        if gender == "M":
            gender = 0
        elif gender == "F":
            gender = 1

        if age < 40:
            age = torch.Tensor([1., 0., 0.])
        elif age >= 40 and age < 50:
            age = torch.Tensor([0., 1., 0.])
        elif age >= 50 and age < 65:
            age = torch.Tensor([0., 0., 1.])

        if race == "WHITE":
            race = torch.Tensor([1., 0., 0.])
        elif race == "BLACK":
            race = torch.Tensor([0., 1., 0.])
        else:
            race = torch.Tensor([0., 0., 1.])
        
        output = {'full_img': full_img, 'insurance': output_insurance, 'img_id': self.rows[idx][0], 'gender': gender, 'age': age, 'race': race}
        return output
    
class MIMIC_raw_ICD_mask_mostimage(Dataset):
    def __init__(self, dataset_csv, mask_start, mask_size_y, mask_size_x):
        with open(dataset_csv , newline='') as csvfile:
            data = list(csv.reader(csvfile))
            self.title = data[0]
            self.rows = data[1::]

        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            # transforms.RandomResizedCrop(size=448),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])        

        self.start_y = mask_start[0]
        self.start_x = mask_start[1]
        self.mask_size_x = mask_size_x
        self.mask_size_y = mask_size_y
    
    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        img_root_dir = "/mnt/new_usb/jupyter-altis5526/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"
        img_dir = f"p{self.rows[idx][1][:2]}/p{self.rows[idx][1]}/s{self.rows[idx][2]}/{self.rows[idx][0]}.jpg"
        img_dir = img_root_dir + img_dir
        full_img = Image.open(img_dir).convert('RGB')
        full_img = full_img.resize((448,448))
        full_img = np.transpose(full_img, (2, 0, 1))/255.0
        mask_img = Image.new('L', (full_img.shape[1], full_img.shape[1]), 0)
        draw = ImageDraw.Draw(mask_img)
        draw.rectangle([self.start_x, self.start_y, self.start_x+self.mask_size_x, self.start_y+self.mask_size_y], fill=255)
        full_img = cv2.bitwise_and(full_img[0, :, :], full_img[0, :, :], mask=np.array(mask_img))
        full_img = np.tile(full_img, (3, 1, 1))

        # full_img = full_img * 255.0
        # cv2.imwrite("mask_image.png", full_img[0])
        
        full_img = torch.from_numpy(full_img.copy()).float()
        full_img = self.augmentation(full_img)

        insurance_index = self.title.index("new_insurance_type")
        insurance_type = self.rows[idx][insurance_index]
        gender_index = self.title.index("gender")
        gender = self.rows[idx][gender_index]
        age_index = self.title.index("anchor_age")
        age = float(self.rows[idx][age_index])
        race_index = self.title.index("race")
        race = self.rows[idx][race_index]
        icd_text_index = self.title.index("disease_list")
        icd_text = self.rows[idx][icd_text_index]
        
        if insurance_type == "Private":
            output_insurance = torch.Tensor([0., 1.])
        elif insurance_type == "Medicaid" or insurance_type == "Medicare":
            output_insurance = torch.Tensor([1., 0.])

        if gender == "M":
            gender = 0
        elif gender == "F":
            gender = 1

        if age < 40:
            age = torch.Tensor([1., 0., 0.])
        elif age >= 40 and age < 50:
            age = torch.Tensor([0., 1., 0.])
        elif age >= 50 and age < 65:
            age = torch.Tensor([0., 0., 1.])

        if race == "WHITE":
            race = torch.Tensor([1., 0., 0.])
        elif race == "BLACK":
            race = torch.Tensor([0., 1., 0.])
        else:
            race = torch.Tensor([0., 0., 1.])
        
        output = {'full_img': full_img, 'insurance': output_insurance, 'img_id': self.rows[idx][0], 'gender': gender, 'age': age, 'race': race}
        return output
    
class MIMIC_raw_low_pass(Dataset):
    def __init__(self, dataset_csv, resize=448, diameter=50):
        with open(dataset_csv , newline='') as csvfile:
            data = list(csv.reader(csvfile))
            self.title = data[0]
            self.rows = data[1::]

        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            # transforms.RandomResizedCrop(size=448),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        self.resize = resize
        self.diameter = diameter
    
    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        img_root_dir = "/mnt/new_usb/jupyter-altis5526/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"
        img_dir = f"p{self.rows[idx][1][:2]}/p{self.rows[idx][1]}/s{self.rows[idx][2]}/{self.rows[idx][0]}.jpg"
        img_dir = img_root_dir + img_dir
        full_img = Image.open(img_dir).convert('RGB')
        full_img = full_img.resize((self.resize,self.resize))
        full_img = np.array(full_img)
        full_img = low_pass(full_img, self.diameter) / 255.0

        full_img_show = (np.transpose(full_img, (1,2,0)) * 255.0).astype(np.uint8)
        full_img_show = cv2.cvtColor(full_img_show, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"low_pass_image{self.diameter}.png", full_img_show)
        
        full_img = torch.from_numpy(full_img.copy()).float()
        full_img = self.augmentation(full_img)

        insurance_index = self.title.index("new_insurance_type")
        insurance_type = self.rows[idx][insurance_index]
        gender_index = self.title.index("gender")
        gender = self.rows[idx][gender_index]
        age_index = self.title.index("anchor_age")
        age = float(self.rows[idx][age_index])
        race_index = self.title.index("race")
        race = self.rows[idx][race_index]
        
        if insurance_type == "Private":
            output_insurance = torch.Tensor([0., 1.])
        elif insurance_type == "Medicaid" or insurance_type == "Medicare":
            output_insurance = torch.Tensor([1., 0.])

        if gender == "M":
            gender = torch.Tensor([1., 0.])
        elif gender == "F":
            gender = torch.Tensor([0., 1.])

        if age < 40:
            age = torch.Tensor([1., 0., 0.])
        elif age >= 40 and age < 50:
            age = torch.Tensor([0., 1., 0.])
        elif age >= 50 and age < 65:
            age = torch.Tensor([0., 0., 1.])

        if race == "WHITE":
            race = torch.Tensor([1., 0., 0.])
        elif race == "BLACK":
            race = torch.Tensor([0., 1., 0.])
        else:
            race = torch.Tensor([0., 0., 1.])
        
        output = {'full_img': full_img, 'insurance': output_insurance, 'img_id': self.rows[idx][0], 'gender': gender, 'age': age, 'race': race}
        return output

class MIMIC_raw_high_pass(Dataset):
    def __init__(self, dataset_csv, resize=448, diameter=50):
        with open(dataset_csv , newline='') as csvfile:
            data = list(csv.reader(csvfile))
            self.title = data[0]
            self.rows = data[1::]

        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            # transforms.RandomResizedCrop(size=448),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        self.resize = resize
        self.diameter = diameter
    
    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        img_root_dir = "/mnt/new_usb/jupyter-altis5526/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"
        img_dir = f"p{self.rows[idx][1][:2]}/p{self.rows[idx][1]}/s{self.rows[idx][2]}/{self.rows[idx][0]}.jpg"
        img_dir = img_root_dir + img_dir
        full_img = Image.open(img_dir).convert('RGB')
        full_img = full_img.resize((self.resize,self.resize))
        full_img = np.array(full_img)
        full_img = high_pass(full_img, self.diameter) / 255.0

        full_img_show = (np.transpose(full_img, (1,2,0)) * 255.0).astype(np.uint8)
        full_img_show = cv2.cvtColor(full_img_show, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"high_pass_image{self.diameter}.png", full_img_show)
        
        full_img = torch.from_numpy(full_img.copy()).float()
        full_img = self.augmentation(full_img)

        insurance_index = self.title.index("new_insurance_type")
        insurance_type = self.rows[idx][insurance_index]
        gender_index = self.title.index("gender")
        gender = self.rows[idx][gender_index]
        age_index = self.title.index("anchor_age")
        age = float(self.rows[idx][age_index])
        race_index = self.title.index("race")
        race = self.rows[idx][race_index]
        
        if insurance_type == "Private":
            output_insurance = torch.Tensor([0., 1.])
        elif insurance_type == "Medicaid" or insurance_type == "Medicare":
            output_insurance = torch.Tensor([1., 0.])

        if gender == "M":
            gender = torch.Tensor([1., 0.])
        elif gender == "F":
            gender = torch.Tensor([0., 1.])

        if age < 40:
            age = torch.Tensor([1., 0., 0.])
        elif age >= 40 and age < 50:
            age = torch.Tensor([0., 1., 0.])
        elif age >= 50 and age < 65:
            age = torch.Tensor([0., 0., 1.])

        if race == "WHITE":
            race = torch.Tensor([1., 0., 0.])
        elif race == "BLACK":
            race = torch.Tensor([0., 1., 0.])
        else:
            race = torch.Tensor([0., 0., 1.])
        
        output = {'full_img': full_img, 'insurance': output_insurance, 'img_id': self.rows[idx][0], 'gender': gender, 'age': age, 'race': race}
        return output


class MIMIC_raw_random_label(Dataset):
    def __init__(self, dataset_csv, resize=448, diameter=50):
        with open(dataset_csv , newline='') as csvfile:
            data = list(csv.reader(csvfile))
            self.title = data[0]
            self.rows = data[1::]

        self.labels = torch.randn(len(self.rows), 2)

        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            # transforms.RandomResizedCrop(size=448),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        self.resize = resize
        self.diameter = diameter
    
    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        img_root_dir = "/mnt/new_usb/jupyter-altis5526/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"
        img_dir = f"p{self.rows[idx][1][:2]}/p{self.rows[idx][1]}/s{self.rows[idx][2]}/{self.rows[idx][0]}.jpg"
        img_dir = img_root_dir + img_dir
        full_img = Image.open(img_dir).convert('RGB')
        full_img = full_img.resize((self.resize,self.resize))
        full_img = np.array(full_img)
        full_img = high_pass(full_img, self.diameter) / 255.0
    
        full_img = torch.from_numpy(full_img.copy()).float()
        full_img = self.augmentation(full_img)

        gender_index = self.title.index("gender")
        gender = self.rows[idx][gender_index]
        age_index = self.title.index("anchor_age")
        age = float(self.rows[idx][age_index])
        race_index = self.title.index("race")
        race = self.rows[idx][race_index]

        
        output_insurance = torch.Tensor([0., 0.])
        _, random_index = torch.max(torch.unsqueeze(self.labels[idx, :], 0), dim=1)
        output_insurance[random_index] = 1.

        if gender == "M":
            gender = torch.Tensor([1., 0.])
        elif gender == "F":
            gender = torch.Tensor([0., 1.])

        if age < 40:
            age = torch.Tensor([1., 0., 0.])
        elif age >= 40 and age < 50:
            age = torch.Tensor([0., 1., 0.])
        elif age >= 50 and age < 65:
            age = torch.Tensor([0., 0., 1.])

        if race == "WHITE":
            race = torch.Tensor([1., 0., 0.])
        elif race == "BLACK":
            race = torch.Tensor([0., 1., 0.])
        else:
            race = torch.Tensor([0., 0., 1.])
        
        output = {'full_img': full_img, 'insurance': output_insurance, 'img_id': self.rows[idx][0], 'gender': gender, 'age': age, 'race': race}
        return output

if __name__ == "__main__":
    train_path = "insurance_dataset_8_1_1_PMMthree_test_addRaceICD.csv"
    dataset = MIMIC_raw_ICD_mask_mostimage(train_path, [0,0], 150, 150)
    train_loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
    for batch in train_loader:
        age = batch["age"]
        break



