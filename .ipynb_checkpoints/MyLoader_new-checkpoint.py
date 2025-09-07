import torch
from tfrecord.torch.dataset import TFRecordDataset
from torch.utils.data import DataLoader
import cv2
from torchvision import transforms
from PIL import Image
import numpy as np
import random



class CheXpertLoader(DataLoader):
    def __init__(self, data_path, index_path, batch_size, num_workers, shuffle, pin_memory=False):
        self.data_path = data_path
        self.index_path = index_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        
        self.description = {
            'fracture': 'int',
            'enlarged_cardiomediastinum': 'int',
            'dataset': 'int',
            'edema': 'int',
            'patient': 'int',
            'support_devices': 'int',
            'pleural_other': 'int',
            'consolidation': 'int',
            'cardiomegaly': 'int',
            'view': 'int',
            'pneumonia': 'int',
            'airspace_opacity': 'int',
            'jpg_bytes': 'byte',
            'pleural_effusion': 'int',
            'atelectasis': 'int',
            'no_finding': 'int',
            'pneumothorax': 'int',
            'study': 'int',
            'image': 'int',
            'lung_lesion': 'int',
            'age': 'float',
            'race': 'byte',
            'insurance_type': 'byte',
            'sex': 'byte'
        }
        
        if self.shuffle == True:
            self.dataset = TFRecordDataset(self.data_path, self.index_path, self.description, shuffle_queue_size = self.batch_size, transform = self.train_decode)
        else:
            self.dataset = TFRecordDataset(self.data_path, self.index_path, self.description, shuffle_queue_size = self.batch_size, transform = self.val_decode)
            
        g = torch.Generator()
        g.manual_seed(0)
             
        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'worker_init_fn': self.seed_worker,
            'generator': g,
        }
        super().__init__(**self.init_kwargs)
        
    def train_decode(self, features):
        # decode_img = np.fromstring(features['jpg_bytes'], dtype=np.uint8)
        # decode_img = np.reshape(decode_img, (448,448))
        decode_img = cv2.imdecode(np.fromstring(features['jpg_bytes'], dtype=np.uint8), -1)
        features['jpg_bytes'] = Image.fromarray(decode_img).convert('RGB')
        transform = transforms.Compose([
#             transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            # transforms.RandomResizedCrop(size=224),
            transforms.ToTensor(),
            # transforms.Resize(192),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        features['jpg_bytes'] = transform(features['jpg_bytes'])

        return features

    def val_decode(self, features):
        decode_img = cv2.imdecode(np.fromstring(features['jpg_bytes'], dtype=np.uint8), -1)
        features['jpg_bytes'] = Image.fromarray(decode_img).convert('RGB')
        transform = transforms.Compose([
#             transforms.ToPILImage(),
            transforms.ToTensor(),
            # transforms.Resize(192),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        features['jpg_bytes'] = transform(features['jpg_bytes'])

        return features
    
    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        
        
                
    @staticmethod    
    def collate_fn(data):
        imgs = []
        final_labels = []
        age = []
        gender = []
        race = []
        label_name = {b'Private Insurance': 0, b'Medicaid': 1, b'Medicare': 1}
        gender_name = {b'Male': 0, b'Female': 1, b'Unknown': 2}
        race_name = {b'Pacific Islander': 2, b'Other': 2, b'Unknown': 2, b'Black': 0, b'Native American': 2, b'Patient Refused': 2, b'Asian': 2, b'White': 1}
        
        for example in data:
            imgs.append(example['jpg_bytes'])
            
            insurance_type = np.zeros(2)
            insurance_type[label_name[example["insurance_type"]]] = 1
            final_labels.append(insurance_type)
            age.append(example['age'])
            gender.append(gender_name[example['sex']])
            race.append(race_name[example['race']])
            
        final_labels = np.stack(final_labels)
        age = np.array(age)
        gender = np.array(gender)
        race = np.array(race)
            
        return torch.stack(imgs, 0), torch.Tensor(final_labels), torch.Tensor(age), torch.Tensor(gender), torch.Tensor(race)
    