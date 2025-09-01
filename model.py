from __future__ import absolute_import, division, print_function
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
from collections import OrderedDict
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import numpy as np
import os, math
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize

class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        return A, x

class CLAM_SB(nn.Module):
    def __init__(self, size_arg = "small", dropout = 0., n_classes=1, n_tasks = 1, embed_dim=1024):
        super().__init__()
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = n_tasks)
        
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        A, h = self.attention_net(h)  # NxK
        # A = torch.transpose(A, 2, 1)  # KxN
        h = torch.transpose(h, 2, 1)
        h = torch.squeeze(h, dim=2)
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=2)  # softmax over N
        A = torch.squeeze(A, dim=3)
        M = torch.bmm(A, h)
        
        return A_raw, M

class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers=18, isPretrained=False, isGrayscale=False, embDimension=128, poolSize=32):
        super(ResnetEncoder, self).__init__()
        self.path_to_model = '/tmp/models'
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.isGrayscale = isGrayscale
        self.isPretrained = isPretrained
        self.embDimension = embDimension
        self.poolSize = poolSize
        self.featListName = ['layer0', 'layer1', 'layer2', 'layer3', 'layer4']
        
        resnets = {
            18: models.resnet18, 
            34: models.resnet34,
            50: models.resnet50, 
            101: models.resnet101,
            152: models.resnet152}
        
        resnets_pretrained_path = {
            18: 'resnet18-5c106cde.pth', 
            34: 'resnet34.pth',
            50: 'resnet50-19c8e357.pth',
            101: 'resnet101.pth',
            152: 'resnet152.pth'}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(
                num_layers))

        self.encoder = resnets[num_layers](pretrained=True)
        
        if self.isPretrained:
            print("using pretrained model")
            self.encoder.load_state_dict(
                torch.load(os.path.join(self.path_to_model, resnets_pretrained_path[num_layers])))
            
        if self.isGrayscale:
            self.encoder.conv1 = nn.Conv2d(
                1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        if num_layers > 34:
            self.num_ch_enc[1:] = 2048
        else:
            self.num_ch_enc[1:] = 512
                    
        if self.embDimension>0:
            self.encoder.fc =  nn.Linear(self.num_ch_enc[-1], self.embDimension)
            

    def forward(self, input_image):
        self.features = []
        
        x = self.encoder.conv1(input_image)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        self.features.append(x)
        
        x = self.encoder.layer1(x)
        self.features.append(x)
        
        x = self.encoder.layer2(x)
        self.features.append(x)
        
        x = self.encoder.layer3(x) 
        self.features.append(x)
        
        x = self.encoder.layer4(x)
        self.features.append(x)
        
        x = F.avg_pool2d(x, self.poolSize)
        
        self.x = x.view(x.size(0), -1)
        
        x = self.encoder.fc(self.x)
        return x
    
    
# class DenseNetWithDropout(nn.Module):
#     def __init__(self, num_classes=1000, dropout_prob=0.25):
#         super(DenseNetWithDropout, self).__init__()
        
#         # Load the pre-trained DenseNet-169 model
#         self.densenet = models.densenet121(pretrained=True)
        
#         # Add a new classification head
#         self.classifier = nn.Linear(self.densenet.classifier.in_features, num_classes, bias=True)
        
#         # Remove the original classification head
#         self.densenet.classifier = nn.Identity()
        
#         # Add dropout layers after each dense block
#         for name, module in self.densenet.features.named_children():
#             if 'denseblock' in name:
#                 setattr(self.densenet.features, name, nn.Sequential(module, nn.Dropout(p=dropout_prob)))

#     def forward(self, x):
#         features = self.densenet(x)
#         features = features.view(features.size(0), -1)
#         output = self.classifier(features)
#         return output

class DenseNetClassification(nn.Module):
    def __init__(self, num_classes=1000, dropout_prob=0.25):
        super(DenseNetClassification, self).__init__()
        
        # Load the pre-trained DenseNet-121 model
        self.densenet = models.densenet121(pretrained=True)
        
        # Add a new classification head
        self.classifier = nn.Linear(self.densenet.classifier.in_features, num_classes, bias=True)
        
        # Remove the original classification head
        self.densenet.classifier = nn.Identity()
        
        for name, module in self.densenet.features.named_children():
            if 'denseblock' in name:
                setattr(self.densenet.features, name, nn.Sequential(module, nn.Dropout(p=dropout_prob)))
        
    def forward(self, x):
        features = self.densenet(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
    
        return output

class DenseNetClassification_addAge(nn.Module):
    def __init__(self, num_classes=1000, dropout_prob=0.25):
        super(DenseNetClassification_addAge, self).__init__()
        
        # Load the pre-trained DenseNet-121 model
        self.densenet = models.densenet121(pretrained=True)
        
        # Add a new classification head
        self.classifier = nn.Linear(self.densenet.classifier.in_features+3, num_classes, bias=True)
        
        # Remove the original classification head
        self.densenet.classifier = nn.Identity()
        
        for name, module in self.densenet.features.named_children():
            if 'denseblock' in name:
                setattr(self.densenet.features, name, nn.Sequential(module, nn.Dropout(p=dropout_prob)))
        
    def forward(self, x, age):
        features = self.densenet(x)
        features = features.view(features.size(0), -1)
        features = torch.cat((features, age), dim=-1)
        output = self.classifier(features)
    
        return output

class DenseNetClassification_addAge5then2(nn.Module):
    def __init__(self, num_classes=1000, dropout_prob=0.25):
        super(DenseNetClassification_addAge5then2, self).__init__()
        
        # Load the pre-trained DenseNet-121 model
        self.densenet = models.densenet121(pretrained=True)
        
        # Add a new classification head
        self.classifier1 = nn.Linear(self.densenet.classifier.in_features, num_classes, bias=True)
        self.classifier2 = nn.Linear(5, 2, bias=True)
        
        # Remove the original classification head
        self.densenet.classifier = nn.Identity()
        
        for name, module in self.densenet.features.named_children():
            if 'denseblock' in name:
                setattr(self.densenet.features, name, nn.Sequential(module, nn.Dropout(p=dropout_prob)))
        
    def forward(self, x, age):
        features = self.densenet(x)
        features = features.view(features.size(0), -1)
        features = self.classifier1(features)
        features = torch.cat((features, age), dim=-1)
        output = self.classifier2(features)
    
        return output

class DenseNetClassification_addSex5then2(nn.Module):
    def __init__(self, num_classes=1000, dropout_prob=0.25):
        super(DenseNetClassification_addSex5then2, self).__init__()
        
        # Load the pre-trained DenseNet-121 model
        self.densenet = models.densenet121(pretrained=True)
        
        # Add a new classification head
        self.classifier1 = nn.Linear(self.densenet.classifier.in_features, num_classes, bias=True)
        self.classifier2 = nn.Linear(4, 2, bias=True)
        
        # Remove the original classification head
        self.densenet.classifier = nn.Identity()
        
        for name, module in self.densenet.features.named_children():
            if 'denseblock' in name:
                setattr(self.densenet.features, name, nn.Sequential(module, nn.Dropout(p=dropout_prob)))
        
    def forward(self, x, sex):
        features = self.densenet(x)
        features = features.view(features.size(0), -1)
        features = self.classifier1(features)
        features = torch.cat((features, sex), dim=-1)
        output = self.classifier2(features)
    
        return output

class DenseNetClassification_addSex(nn.Module):
    def __init__(self, num_classes=1000, dropout_prob=0.25):
        super(DenseNetClassification_addSex, self).__init__()
        
        # Load the pre-trained DenseNet-121 model
        self.densenet = models.densenet121(pretrained=True)
        
        # Add a new classification head
        self.classifier = nn.Linear(self.densenet.classifier.in_features+2, num_classes, bias=True)
        
        # Remove the original classification head
        self.densenet.classifier = nn.Identity()
        
        for name, module in self.densenet.features.named_children():
            if 'denseblock' in name:
                setattr(self.densenet.features, name, nn.Sequential(module, nn.Dropout(p=dropout_prob)))
        
    def forward(self, x, sex):
        features = self.densenet(x)
        features = features.view(features.size(0), -1)
        features = torch.cat((features, sex), dim=-1)
        output = self.classifier(features)
    
        return output

class DenseNetClassification_addAgeRace(nn.Module):
    def __init__(self, num_classes=1000, dropout_prob=0.25):
        super(DenseNetClassification_addAgeRace, self).__init__()
        
        # Load the pre-trained DenseNet-121 model
        self.densenet = models.densenet121(pretrained=True)
        
        # Add a new classification head
        self.classifier = nn.Linear(self.densenet.classifier.in_features+6, num_classes, bias=True)
        
        # Remove the original classification head
        self.densenet.classifier = nn.Identity()
        
        for name, module in self.densenet.features.named_children():
            if 'denseblock' in name:
                setattr(self.densenet.features, name, nn.Sequential(module, nn.Dropout(p=dropout_prob)))
        
    def forward(self, x, age, race):
        features = self.densenet(x)
        features = features.view(features.size(0), -1)
        features = torch.cat((features, age, race), dim=-1)
        output = self.classifier(features)
    
        return output

class DenseNetClassification_addAgeRace8then2(nn.Module):
    def __init__(self, num_classes=1000, dropout_prob=0.25):
        super(DenseNetClassification_addAgeRace8then2, self).__init__()
        
        # Load the pre-trained DenseNet-121 model
        self.densenet = models.densenet121(pretrained=True)
        
        # Add a new classification head
        self.classifier1 = nn.Linear(self.densenet.classifier.in_features, num_classes, bias=True)
        self.classifier2 = nn.Linear(8, 2, bias=True)
        
        # Remove the original classification head
        self.densenet.classifier = nn.Identity()
        
        for name, module in self.densenet.features.named_children():
            if 'denseblock' in name:
                setattr(self.densenet.features, name, nn.Sequential(module, nn.Dropout(p=dropout_prob)))
        
    def forward(self, x, age, race):
        features = self.densenet(x)
        features = features.view(features.size(0), -1)
        features = self.classifier1(features)
        features = torch.cat((features, age, race), dim=-1)
        output = self.classifier2(features)
    
        return output

class DenseNetClassification_addAgeRaceSex(nn.Module):
    def __init__(self, num_classes=1000, dropout_prob=0.25):
        super(DenseNetClassification_addAgeRaceSex, self).__init__()
        
        # Load the pre-trained DenseNet-121 model
        self.densenet = models.densenet121(pretrained=True)
        
        # Add a new classification head
        self.classifier = nn.Linear(self.densenet.classifier.in_features+8, num_classes, bias=True)
        
        # Remove the original classification head
        self.densenet.classifier = nn.Identity()
        
        for name, module in self.densenet.features.named_children():
            if 'denseblock' in name:
                setattr(self.densenet.features, name, nn.Sequential(module, nn.Dropout(p=dropout_prob)))
        
    def forward(self, x, age, race, sex):
        features = self.densenet(x)
        features = features.view(features.size(0), -1)
        features = torch.cat((features, age, race, sex), dim=-1)
        output = self.classifier(features)
    
        return output

class DenseNetClassification_addICD(nn.Module):
    def __init__(self, num_classes=1000, dropout_prob=0.25):
        super(DenseNetClassification_addICD, self).__init__()
        
        # Load the pre-trained DenseNet-121 model
        self.densenet = models.densenet121(pretrained=True)
        
        # Add a new classification head
        self.classifier = nn.Linear(self.densenet.classifier.in_features+512, num_classes, bias=True)
        self.attention = CLAM_SB()
        
        # Remove the original classification head
        self.densenet.classifier = nn.Identity()
        
        for name, module in self.densenet.features.named_children():
            if 'denseblock' in name:
                setattr(self.densenet.features, name, nn.Sequential(module, nn.Dropout(p=dropout_prob)))
        
    def forward(self, x, icd_embed):
        _, icd_embed = self.attention(icd_embed)
        features = self.densenet(x)
        features = features.view(features.size(0), -1)
        icd_embed = torch.squeeze(icd_embed, dim=1)
        features = torch.cat((features, icd_embed), dim=-1)
        output = self.classifier(features)
    
        return output

class DenseNetClassification_addAgeRaceICD(nn.Module):
    def __init__(self, num_classes=1000, dropout_prob=0.25):
        super(DenseNetClassification_addAgeRaceICD, self).__init__()
        
        # Load the pre-trained DenseNet-121 model
        self.densenet = models.densenet121(pretrained=True)
        
        # Add a new classification head
        self.classifier = nn.Linear(self.densenet.classifier.in_features+6+512, num_classes, bias=True)
        self.attention = CLAM_SB()
        
        # Remove the original classification head
        self.densenet.classifier = nn.Identity()
        
        for name, module in self.densenet.features.named_children():
            if 'denseblock' in name:
                setattr(self.densenet.features, name, nn.Sequential(module, nn.Dropout(p=dropout_prob)))
        
    def forward(self, x, age, race, icd_embed):
        _, icd_embed = self.attention(icd_embed)
        features = self.densenet(x)
        features = features.view(features.size(0), -1)
        icd_embed = torch.squeeze(icd_embed, dim=1)
        features = torch.cat((features, age, race, icd_embed), dim=-1)
        output = self.classifier(features)
    
        return output
    
class DenseNetWithDoubleLinear(nn.Module):
    def __init__(self, num_classes=1000, dropout_prob=0.25):
        super(DenseNetWithDoubleLinear, self).__init__()
        
        # Load the pre-trained DenseNet-121 model
        self.densenet = models.densenet121(pretrained=True)
        
        # Add a new classification head
        self.classifier = nn.Linear(self.densenet.classifier.in_features, 512, bias=True)
        self.linear_probe1 = nn.Linear(512, 128, bias=True)
        self.linear_probe2 = nn.Linear(128, num_classes, bias=True)
        
        # Remove the original classification head
        self.densenet.classifier = nn.Identity()
        
        for name, module in self.densenet.features.named_children():
            if 'denseblock' in name:
                setattr(self.densenet.features, name, nn.Sequential(module, nn.Dropout(p=dropout_prob)))
        
    def forward(self, x):
        features = self.densenet(x)
        features = features.view(features.size(0), -1)
        features = self.classifier(features)
        features = self.linear_probe1(features)
        output = self.linear_probe2(features)
        return output


class DenseNetWithDoubleLinear_addAge(nn.Module):
    def __init__(self, num_classes=1000, dropout_prob=0.25):
        super(DenseNetWithDoubleLinear_addAge, self).__init__()
        
        # Load the pre-trained DenseNet-121 model
        self.densenet = models.densenet121(pretrained=True)
        
        # Add a new classification head
        self.classifier = nn.Linear(self.densenet.classifier.in_features+6, 512, bias=True)
        self.linear_probe1 = nn.Linear(512, 128, bias=True)
        self.linear_probe2 = nn.Linear(128, num_classes, bias=True)
        
        # Remove the original classification head
        self.densenet.classifier = nn.Identity()
        
        for name, module in self.densenet.features.named_children():
            if 'denseblock' in name:
                setattr(self.densenet.features, name, nn.Sequential(module, nn.Dropout(p=dropout_prob)))
        
    def forward(self, x, age, race):
        features = self.densenet(x)
        features = features.view(features.size(0), -1)
        features = torch.cat((features, age, race), dim=-1)
        
        features = features.to(torch.float16)
        features = self.classifier(features)
        features = self.linear_probe1(features)
        output = self.linear_probe2(features)
        return output

class DenseNetWithDoubleLinear_addICD(nn.Module):
    def __init__(self, num_classes=1000, dropout_prob=0):
        super(DenseNetWithDoubleLinear_addICD, self).__init__()
        
        # Load the pre-trained DenseNet-121 model
        self.densenet = models.densenet121(pretrained=True)
        self.attention = CLAM_SB()
        
        # Add a new classification head
        self.classifier = nn.Linear(self.densenet.classifier.in_features+6+512, 512, bias=True)
        self.linear_probe1 = nn.Linear(512, 128, bias=True)
        self.linear_probe2 = nn.Linear(128, num_classes, bias=True)
        
        # Remove the original classification head
        self.densenet.classifier = nn.Identity()
        
        for name, module in self.densenet.features.named_children():
            if 'denseblock' in name:
                setattr(self.densenet.features, name, nn.Sequential(module, nn.Dropout(p=dropout_prob)))
        
    def forward(self, x, age, race, icd_embed):
        _, icd_embed = self.attention(icd_embed)
        features = self.densenet(x)
        features = features.view(features.size(0), -1)
        icd_embed = torch.squeeze(icd_embed, dim=1)
        features = torch.cat((features, age, race, icd_embed), dim=-1)
        
        features = features.to(torch.float16)
        features = self.classifier(features)
        features = self.linear_probe1(features)
        output = self.linear_probe2(features)
        return output


class ICD_for_insurnce(nn.Module):
    def __init__(self, num_classes=2):
        super(ICD_for_insurnce, self).__init__()
        self.linear = nn.Linear(512, num_classes)
        self.attention = CLAM_SB()

    def forward(self, x):
        _, icd_embed = self.attention(x)
        icd_embed = torch.squeeze(icd_embed, dim=1)
        output = self.linear(icd_embed)

        return output


    