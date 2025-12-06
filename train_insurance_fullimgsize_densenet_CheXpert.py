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
import argparse

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    
def evaluate(model, val_loader, num_classes, subgroup):
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

            if subgroup == "Male":
                filter = (test_sex == 0)

            elif subgroup == "Female":
                filter = (test_sex == 1)

            elif subgroup == "Young":
                filter = torch.bitwise_and(test_age < 40, test_age < 40)

            elif subgroup == "Middle":
                filter = torch.bitwise_and(test_age >= 40, test_age < 50)

            elif subgroup == "Old":
                filter = torch.bitwise_and(test_age >= 50, test_age < 65)

            elif subgroup == "White":
                filter = (test_race == 1)

            elif subgroup == "Black":
                filter = (test_race == 0)

            elif subgroup == "Race_Others":
                filter = (test_race == 2)

            if subgroup != "all" and subgroup != "All":
                test_imgs = test_imgs[filter.flatten()]
                test_labels = test_labels[filter.flatten()]
            
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", help='', type=str)
    parser.add_argument("--val_path", help='', type=str)
    parser.add_argument("--experiment_name", help='', type=str)
    parser.add_argument("--weight_dir", help='', type=str)
    parser.add_argument("--mode", help='', type=str)
    parser.add_argument("--seed", help='', type=int)
    parser.add_argument("--root_dir", help='', type=str, default="/mnt/new_usb/jupyter-altis5526/new_insurancetype_weight/")
    parser.add_argument("--subgroup", help='', type=str, default="all")
    
    args = parser.parse_args()

    set_seed(args.seed)
    torch.cuda.set_device(1)
    weight_dir = os.path.join(args.root_dir+args.weight_dir, f'Rand{str(args.seed)}')
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    train_path = args.train_path
    val_path = args.val_path
    train_wandb_name = f'Rand{str(args.seed)}' + '_' + args.experiment_name
    val_wandb_name = f'Test_Rand{str(args.seed)}' + '_' + args.experiment_name

    if args.mode == "train":
        training = True
    elif args.mode == "test":
        training = False
        
    epochs = 200
    batch_size = 128
    num_classes = 2
    train_index = None
    val_index = None
    opt_lr = 5e-6
    weight_decay = 0
    dropout_prob = 0
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder = DenseNetWithDoubleLinear(num_classes=num_classes, dropout_prob=dropout_prob)
    encoder.to(device)
    
    opt = Lion(encoder.parameters(), lr=opt_lr, weight_decay = weight_decay)

    if args.mode == "train":
        train_loader = CheXpertLoader(train_path, train_index, batch_size, num_workers=1, dataset_type="train")
        val_loader = CheXpertLoader(val_path, val_index, batch_size, num_workers=1, dataset_type="val")

    elif args.mode == "test":
        train_loader = CheXpertLoader(train_path, train_index, batch_size, num_workers=1, dataset_type="train")
        val_loader = CheXpertLoader(val_path, val_index, batch_size, num_workers=1, dataset_type="test")
        
    criterion = nn.CrossEntropyLoss()
    
    testing_weight_path = f"{weight_dir}/{train_wandb_name}_model_best.pt"
    if training == False:
        encoder.load_state_dict(torch.load(testing_weight_path)["model_state_dict"])
        
    if training == True:
        wandb.init(
            project='insurance_classification',
            name= train_wandb_name, 
            settings=wandb.Settings(start_method="fork"))
        config = wandb.config
        config.batch_size = batch_size
        config.opt_lr = opt_lr
        config.weight_decay = weight_decay
        config.dropout = dropout_prob
        config.weight_path = weight_dir
        max_auc = 0
        total = 0
        scaler = torch.cuda.amp.GradScaler()

        stop_criteria = 0
        
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
            auc, precision, recall, f1, acc, test_running_loss, test_total = evaluate(encoder, val_loader, num_classes, args.subgroup)
#             scheduler.step()
            stop_criteria += 1
            
            if auc > max_auc:
                max_auc = auc
                torch.save({
                    'model_state_dict': encoder.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                }, f"{weight_dir}/{train_wandb_name}_model_best.pt")
                stop_criteria = 0

            if stop_criteria >= 10:
                break
                
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"epoch {epoch} / AUC: {auc} / precision: {precision} / recall: {recall} / f1: {f1} / acc: {acc} / test loss: {test_running_loss / test_total} / duration: {duration}")

            wandb.log({'auc': auc, 'precision': precision, 'recall': recall, 'f1': f1, 'acc': acc, 'testing_loss': test_running_loss / test_total})
            
            
    if training == False:
        wandb.init(
            project='insurance_classification',
            name= val_wandb_name, 
            settings=wandb.Settings(start_method="fork"))
        config = wandb.config
        config.batch_size = batch_size
        config.test_weight = testing_weight_path
        
        auc, precision, recall, f1, acc, test_running_loss, test_total = evaluate(encoder, val_loader, num_classes, args.subgroup)
        
        print(f"{args.subgroup} analysis AUC: {auc} / precision: {precision} / recall: {recall} / f1: {f1} / acc: {acc} / test loss: {test_running_loss / test_total}")
        
        if args.subgroup == "all":
            wandb.log({'auc': auc, 'precision': precision, 'recall': recall, 'f1': f1, 'acc': acc, 'testing_loss': test_running_loss / test_total})
        
                
                
                