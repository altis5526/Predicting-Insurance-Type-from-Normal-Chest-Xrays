import torch
from model import *
import numpy as np
import os
import random
import torch.optim as optim
import torch.nn as nn
from MyLoader_new import CheXpertLoader
import time
import torchvision.models as models
from torchmetrics.classification import MultilabelAveragePrecision, MulticlassAUROC, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
import wandb
from torch.optim.lr_scheduler import ExponentialLR
from model import DenseNetClassification
from lion_pytorch import Lion

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    
def evaluate(model, val_loader, num_classes):
    model.eval()
    test_running_loss = 0.0
    test_total = 0
    correct = 0
    medicaid = 0
    
    with torch.no_grad():
        record_target_label = torch.zeros(1, num_classes).to(device)
        record_predict_label = torch.zeros(1, num_classes).to(device)
        
        for (test_imgs, test_labels, test_age, test_sex, test_race) in val_loader:
            test_imgs = test_imgs.to(device)
            test_labels = test_labels.to(device)
            test_labels = test_labels.squeeze(-1)

            # filter = torch.bitwise_and(test_age >= 50, test_age < 65)
            # filter = (test_sex == 1)
            # filter = (test_race == 2)
            # test_imgs = test_imgs[filter.flatten()]
            # test_labels = test_labels[filter.flatten()]
            
            test_output = model(test_imgs)
            loss = criterion(test_output, test_labels)
            loss = loss.mean()
            
            test_running_loss += loss.item() * test_imgs.size(0)
            test_total += test_imgs.size(0)
            
            record_target_label = torch.cat((record_target_label, test_labels), 0)
            record_predict_label = torch.cat((record_predict_label, test_output), 0)
            
            _, test_label_transform = torch.max(test_labels, 1)
            _, test_output_transform = torch.max(test_output, 1)

            correct += (test_label_transform==test_output_transform).sum()
            
        
        record_target_label = record_target_label[1::]
        record_predict_label = record_predict_label[1::]
        
        _, one_label_target = torch.max(record_target_label, 1)
        
        auc_metric = MulticlassAUROC(num_classes=num_classes, average="macro", thresholds=None).to(device)
        precision_metric = MulticlassPrecision(num_classes=num_classes, average="macro").to(device)
        recall_metric = MulticlassRecall(num_classes=num_classes, average="macro").to(device)
        f1_metric = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)
        
        auc = auc_metric(record_predict_label, one_label_target)
        precision = precision_metric(record_predict_label, one_label_target)
        recall = recall_metric(record_predict_label, one_label_target)
        f1 = f1_metric(record_predict_label, one_label_target)
        acc = correct / test_total


    return auc, precision, recall, f1, acc, test_running_loss, test_total

def compute_group_avg(losses, group_idx, ngroups):
    # compute observed counts and mean loss for each group
    group_map = (group_idx == torch.arange(ngroups).unsqueeze(1).long().to(device)).float().to(device)
    group_count = group_map.sum(1)
    group_denom = group_count + (group_count==0).float() # avoid nans
    
    group_loss = (group_map @ losses.view(-1))/group_denom
    return group_loss, group_count

def compute_robust_loss(group_loss, group_count, group_weights, step_size):
        adjusted_loss = group_loss
        adjusted_loss = adjusted_loss/(adjusted_loss.sum())
        group_weights = group_weights * torch.exp(step_size*adjusted_loss.data)
        group_weights = group_weights/group_weights.sum()

        robust_loss = group_loss @ group_weights
        return robust_loss, group_weights
            

if __name__ == "__main__":
    set_seed(123)
    torch.cuda.set_device(0)
    weight_dir = "/mnt/new_usb/jupyter-altis5526/new_insurancetype_weight/CheXpert_BS32_PMMthree_FULLIMAGE448_densenet121_SingleLinear_Lion4e-5_20250713"
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
        
    epochs = 200
    batch_size = 64
    num_classes = 2
    train_path = "/mnt/new_usb/jupyter-altis5526/Resplit_CheXpert_train_insurance.tfrecord"
    train_index = None
    val_path = "/mnt/new_usb/jupyter-altis5526/Resplit_CheXpert_test_insurance.tfrecord"
    val_index = None
    opt_lr = 4e-5
    weight_decay = 0
    training = False
    train_wandb_name = "CheXpert_BS32_PMMthree_FULLIMAGE448_densenet121_SingleLinear_Lion4e-5_20250713"
    val_wandb_name = "Test_CheXpert_BS32_PMMthree_FULLIMAGE448_densenet121_SingleLinear_Lion4e-5_20250713"
    groupDRO = False
    DRO_step_size = 1e-5
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder = DenseNetClassification(num_classes=num_classes, dropout_prob=0)
    encoder.to(device)
    
    opt = Lion(encoder.parameters(), lr=opt_lr, weight_decay = weight_decay)
    train_loader = CheXpertLoader(train_path, train_index, batch_size, num_workers=1, shuffle=True)
    val_loader = CheXpertLoader(val_path, val_index, batch_size, num_workers=1, shuffle=False)
    
    if groupDRO == True:
        criterion = nn.CrossEntropyLoss(reduction="none")
    else:
        # criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, 1.88]).to(device))
        criterion = nn.CrossEntropyLoss()
#     scheduler = ExponentialLR(opt, gamma=0.9)
    
    testing_weight_path = f"{weight_dir}/{train_wandb_name}_model_best.pt"
    if training == False:
        encoder.load_state_dict(torch.load(testing_weight_path)["model_state_dict"])
        
    if groupDRO == True:
        group_weights = torch.ones(1).to(device) / 2
    if training == True:
        wandb.init(
            project='insurance_classification',
            name= train_wandb_name, 
            settings=wandb.Settings(start_method="fork"))
        config = wandb.config
        config.batch_size = batch_size
        config.opt_lr = opt_lr
        config.weight_decay = weight_decay
        config.weight_path = weight_dir
        config.train_path = train_path
        config.val_path = val_path
        max_auc = 0
        total = 0
        scaler = torch.cuda.amp.GradScaler()
        
        for epoch in range(epochs):
            encoder.train()
            running_loss = 0.0
            start_time = time.time()
            count = 0
            
            for (imgs, labels, age, sex, race) in train_loader:
                encoder.zero_grad()
                opt.zero_grad()
                
                imgs = imgs.to(device)
                labels = labels.to(device)
                labels = labels.squeeze(-1)
                
                _, target_label = torch.max(labels, 1)
                
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    output = encoder(imgs)
                    loss = criterion(output, labels)
                    if groupDRO == True:
                        group_loss, group_count = compute_group_avg(loss, target_label, 2)
                        loss, new_weights = compute_robust_loss(group_loss, group_count, group_weights, DRO_step_size)
                        group_weights = new_weights
                
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                
                running_loss += loss.item() * imgs.size(0)
                count += imgs.size(0)
                
                if count != 0 and count % 2560 == 0 and total == 0:
                    print(f"epoch {epoch}: {count}/unknown finished / train loss: {running_loss / count}")
                
                elif count != 0 and count % 2560 == 0 and total != 0:
                    print(f"epoch {epoch}: {count}/{total} (%.2f %%) finished / train loss: {running_loss / count}" % (count/total))
                
            total = count
            auc, precision, recall, f1, acc, test_running_loss, test_total = evaluate(encoder, val_loader, num_classes)
#             scheduler.step()
            
            if auc > max_auc:
                max_auc = auc
                torch.save({
                    'model_state_dict': encoder.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                }, f"{weight_dir}/{train_wandb_name}_model_best.pt")
                
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"epoch {epoch} / AUC: {auc} / precision: {precision} / recall: {recall} / f1: {f1} / acc: {acc} / test loss: {test_running_loss / test_total} / duration: {duration}")
            
            wandb.log({'auc': auc, 'precision': precision, 'recall': recall, 'f1': f1, 'acc': acc, 'testing_loss': test_running_loss / test_total})
            
    if training == False:
        # wandb.init(
        #     project='insurance_classification',
        #     name= val_wandb_name, 
        #     settings=wandb.Settings(start_method="fork"))
        # config = wandb.config
        # config.batch_size = batch_size
        # config.test_weight = testing_weight_path
        # config.train_path = train_path
        # config.test_path = val_path
        
        auc, precision, recall, f1, acc, test_running_loss, test_total = evaluate(encoder, val_loader, num_classes)

        print(f"AUC: {auc} / precision: {precision} / recall: {recall} / f1: {f1} / acc: {acc} / test loss: {test_running_loss / test_total} / test total: {test_total}")
        
        wandb.log({'auc': auc, 'precision': precision, 'recall': recall, 'f1': f1, 'acc': acc, 'testing_loss': test_running_loss / test_total})
                
                
                