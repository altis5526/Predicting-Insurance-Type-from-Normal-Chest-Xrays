import torch
from model import *
import numpy as np
import os
import random
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import torchvision.models as models
from torchmetrics.classification import MultilabelAveragePrecision, MulticlassAUROC, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
import wandb
from torch.optim.lr_scheduler import ExponentialLR
from model import DenseNetClassification, DenseNetWithDoubleLinear
from lion_pytorch import Lion
from RawImageDataset import MIMIC_raw_ICD_mask_mostimage

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
    
def evaluate(model, val_loader, num_classes):
    model.eval()
    test_running_loss = 0.0
    test_total = 0
    correct = 0
    
    with torch.no_grad():
        record_target_label = torch.zeros(1, num_classes).to(device)
        record_predict_label = torch.zeros(1, num_classes).to(device)
        
        for batch in val_loader:
            test_imgs = batch["full_img"]
            test_labels = batch["insurance"]
            age = batch["age"]
            gender = batch["gender"]
            race = batch["race"]
            test_imgs = test_imgs.to(device)
            test_labels = test_labels.to(device)
            test_labels = test_labels.squeeze(-1)

            # _, age_label = torch.max(age, 1)
            # _, race_label = torch.max(race, 1)
            # filter = (age_label == 2)
            # filter = (race_label == 2)
            # filter = (gender == 1)
            # test_imgs = test_imgs[filter.flatten()]
            # test_labels = test_labels[filter.flatten()]

            if test_imgs.size(0) == 0:
                continue
            
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

if __name__ == "__main__":
    set_seed(123)
    weight_dir = "/data/insurance/weights/Remain_area[2_2]_8_1_1split_BS32_PMMthree_FULLIMAGE448_densenet121_SingleLinear_Lion4e-5_20250419"
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
        
    epochs = 100
    batch_size = 32
    num_classes = 2
    train_path = "insurance_dataset_8_1_1_PMMthree_train_addRaceICD.csv"
    val_path = "insurance_dataset_8_1_1_PMMthree_test_addRaceICD.csv"
    opt_lr = 4e-5
    weight_decay = 0
    training = False
    train_wandb_name = "Remain_area[2_2]_8_1_1split_BS32_PMMthree_FULLIMAGE448_densenet121_SingleLinear_Lion4e-5_20250419"
    val_wandb_name = "Test_Remain_area[2_2]_8_1_1split_BS32_PMMthree_FULLIMAGE448_densenet121_SingleLinear_Lion4e-5_20250419"
    dropout_prob = 0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder = DenseNetClassification(num_classes=num_classes, dropout_prob=dropout_prob)
    encoder.to(device)
    
    g = torch.Generator()
    g.manual_seed(0)
    opt = Lion(encoder.parameters(), lr=opt_lr, weight_decay = weight_decay)
    train_dataset = MIMIC_raw_ICD_mask_mostimage(train_path, [300,300], 148, 148)
    val_dataset = MIMIC_raw_ICD_mask_mostimage(val_path, [300,300], 148, 148)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, worker_init_fn=seed_worker, num_workers=0, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, worker_init_fn=seed_worker, num_workers=0, shuffle=False, generator=g)
    
    criterion = nn.CrossEntropyLoss()
    
    testing_weight_path = f"{weight_dir}/{train_wandb_name}_model_aucbest.pt"
    
    if training == False:
        encoder.load_state_dict(torch.load(testing_weight_path)["model_state_dict"])
        
    if training == True:
        max_auc = 0
        max_acc = 0
        total = 0
        scaler = torch.cuda.amp.GradScaler()
        
        for epoch in range(epochs):
            encoder.train()
            running_loss = 0.0
            start_time = time.time()
            count = 0
            
            for batch in train_loader:
                encoder.zero_grad()
                opt.zero_grad()
                imgs = batch["full_img"]
                labels = batch["insurance"]
                # dicom_id = batch["img_id"]
                imgs = imgs.to(device)
                labels = labels.to(device)
                labels = labels.squeeze(-1)

                _, target_label = torch.max(labels, 1)
                
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    output = encoder(imgs)
                    loss = criterion(output, labels)
                
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            
                
                running_loss += loss.item() * imgs.size(0)
                count += imgs.size(0)
                
                if count != 0 and count % 2560 == 0 and total == 0:
                    print(f"epoch {epoch}: {count}/unknown finished / train loss: {running_loss / count}")
                
                elif count != 0 and count % 2560 == 0 and total != 0:
                    print(f"epoch {epoch}: {count}/{total} (%.2f %%) finished / train loss: {running_loss / count}" % (count/total*100))
                
            total = count
            auc, precision, recall, f1, acc, test_running_loss, test_total = evaluate(encoder, val_loader, num_classes)
            # scheduler.step()
            
            if auc > max_auc:
                max_auc = auc
                torch.save({
                    'model_state_dict': encoder.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                }, f"{weight_dir}/{train_wandb_name}_model_aucbest.pt")

            if acc > max_acc:
                max_acc = acc
                torch.save({
                    'model_state_dict': encoder.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                }, f"{weight_dir}/{train_wandb_name}_model_accbest.pt")
                
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"epoch {epoch} / AUC: {auc} / precision: {precision} / recall: {recall} / f1: {f1} / acc: {acc} / test loss: {test_running_loss / test_total} / duration: {duration}")
            
            wandb.log({'auc': auc, 'precision': precision, 'recall': recall, 'f1': f1, 'acc': acc, 'testing_loss': test_running_loss / test_total})
            
    if training == False:
        auc, precision, recall, f1, acc, test_running_loss, test_total = evaluate(encoder, val_loader, num_classes)
        
        print(f"AUC: {auc} / precision: {precision} / recall: {recall} / f1: {f1} / acc: {acc} / test loss: {test_running_loss / test_total}")
        
        wandb.log({'auc': auc, 'precision': precision, 'recall': recall, 'f1': f1, 'acc': acc, 'testing_loss': test_running_loss / test_total})
                
                
                