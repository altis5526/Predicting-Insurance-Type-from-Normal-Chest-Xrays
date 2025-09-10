import torch
from torch.utils.data import DataLoader
import cv2
import numpy as np
import random
from torch.nn.utils.rnn import pad_sequence


class CombinedLoader(DataLoader):
    def __init__(self, dataset, batch_size, num_workers, shuffle, pin_memory=False):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.dataset = dataset
            
        g = torch.Generator()
        g.manual_seed(0)
             
        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'worker_init_fn': self.seed_worker,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'generator': g,
        }
        super().__init__(**self.init_kwargs)
        
    
    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    @staticmethod
    def collate_fn(batch):
        features = []
        img_dirs = []
        full_imgs = []
        insurances = []
        img_ids = []
        genders = []
        ages = []
        races = []
        max_length = -1
        for data in batch:
            feature = data["icd_feature"]
            if feature.size(1) > max_length:
                max_length = feature.size(1)
        for data in batch:
            feature = data["icd_feature"]
            padding_length = max_length - feature.size(1)
            if padding_length > 0:
                pad_tensor = torch.zeros((1, padding_length, feature.size(2)))
                feature = torch.cat((feature, pad_tensor), dim=1)
            features.append(feature)
            img_dirs.append(data["img_dir"])
            full_imgs.append(data["full_img"])
            insurances.append(torch.Tensor(data["insurance"]))
            genders.append(data["gender"])
            ages.append(torch.Tensor(data["age"]))
            races.append(torch.Tensor(data["race"]))
        lengths = torch.tensor([feature.size(0) for feature in features])

        output = {"img_dirs": img_dirs, 'full_img': torch.stack(full_imgs), 'insurance': torch.stack(insurances), 'gender': torch.Tensor(genders), 'age': torch.stack(ages), "race": torch.stack(races), "icd_feature": torch.stack(features), "lengths": lengths}

        return output
