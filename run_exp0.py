import subprocess
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help='', type=str)
    parser.add_argument("--model", help='', type=str)
    parser.add_argument("--train_path", help='', type=str)
    parser.add_argument("--val_path", help='', type=str)
    parser.add_argument("--test_path", help='', type=str)
    parser.add_argument("--experiment_name", help='', type=str)
    parser.add_argument("--weight_dir", help='', type=str)
    
    args = parser.parse_args()

    seeds = ["123"]

    if args.dataset == "MIMIC":
        if args.model == "densenet":
            for seed in seeds:
                subprocess.run(f'python train_insurance_fullimgsize_densenet.py --mode train --train_path {args.train_path} --val_path {args.val_path} --experiment_name {args.experiment_name} --weight_dir {args.weight_dir} --seed {seed}', shell=True)
    
                subprocess.run(f'python train_insurance_fullimgsize_densenet.py --mode test --train_path {args.train_path} --val_path {args.test_path} --experiment_name {args.experiment_name} --weight_dir {args.weight_dir} --seed {seed}', shell=True)

        elif args.model == "mamba":
            for seed in seeds:
                subprocess.run(f'python train_insurance_fullimgsize_mamba.py --mode train --train_path {args.train_path} --val_path {args.val_path} --experiment_name {args.experiment_name} --weight_dir {args.weight_dir} --seed {seed}', shell=True)
    
                subprocess.run(f'python train_insurance_fullimgsize_mamba.py --mode test --train_path {args.train_path} --val_path {args.test_path} --experiment_name {args.experiment_name} --weight_dir {args.weight_dir} --seed {seed}', shell=True)

        elif args.model == "swinTF":
            for seed in seeds:
                subprocess.run(f'python train_insurance_fullimgsize_swintransformer_csv.py --mode train --train_path {args.train_path} --val_path {args.val_path} --experiment_name {args.experiment_name} --weight_dir {args.weight_dir} --seed {seed}', shell=True)
    
                subprocess.run(f'python train_insurance_fullimgsize_swintransformer_csv.py --mode test --train_path {args.train_path} --val_path {args.test_path} --experiment_name {args.experiment_name} --weight_dir {args.weight_dir} --seed {seed}', shell=True)


    if args.dataset == "CheXpert":
        if args.model == "densenet":
            for seed in seeds:
                subprocess.run(f'python train_insurance_fullimgsize_densenet_CheXpert.py --mode train --train_path {args.train_path} --val_path {args.val_path} --experiment_name {args.experiment_name} --weight_dir {args.weight_dir} --seed {seed}', shell=True)
    
                subprocess.run(f'python train_insurance_fullimgsize_densenet_CheXpert.py --mode test --train_path {args.train_path} --val_path {args.test_path} --experiment_name {args.experiment_name} --weight_dir {args.weight_dir} --seed {seed}', shell=True)

        elif args.model == "swinTF":
            for seed in seeds:
                subprocess.run(f'python train_insurance_fullimgsize_swintransformer_CheXpert.py --mode train --train_path {args.train_path} --val_path {args.val_path} --experiment_name {args.experiment_name} --weight_dir {args.weight_dir} --seed {seed}', shell=True)
    
                subprocess.run(f'python train_insurance_fullimgsize_swintransformer_CheXpert.py --mode test --train_path {args.train_path} --val_path {args.test_path} --experiment_name {args.experiment_name} --weight_dir {args.weight_dir} --seed {seed}', shell=True)

        elif args.model == "mamba":
            for seed in seeds:
                subprocess.run(f'python train_insurance_fullimgsize_mamba_CheXpert.py --mode train --train_path {args.train_path} --val_path {args.val_path} --experiment_name {args.experiment_name} --weight_dir {args.weight_dir} --seed {seed}', shell=True)
    
                subprocess.run(f'python train_insurance_fullimgsize_mamba_CheXpert.py --mode test --train_path {args.train_path} --val_path {args.test_path} --experiment_name {args.experiment_name} --weight_dir {args.weight_dir} --seed {seed}', shell=True)


        