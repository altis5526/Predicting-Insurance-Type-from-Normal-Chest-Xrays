import subprocess
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", help='', type=str)
    parser.add_argument("--train_path", help='', type=str)
    parser.add_argument("--val_path", help='', type=str)
    parser.add_argument("--test_path", help='', type=str)
    parser.add_argument("--experiment_name", help='', type=str)
    parser.add_argument("--weight_dir", help='', type=str)
    args = parser.parse_args()

    seeds = [123]
    
    
    if args.method == "remove":
        for seed in seeds:
            for idx in range(1,10):
                subprocess.run(f'python train_insurance_fullimgsize_mamba_mask_area.py --mode train --train_path {args.train_path} --val_path {args.val_path} --experiment_name {args.experiment_name} --weight_dir {args.weight_dir} --seed {seed} --patch_idx {idx}', shell=True)
                subprocess.run(f'python train_insurance_fullimgsize_mamba_mask_area.py --mode test --train_path {args.train_path} --val_path {args.test_path} --experiment_name {args.experiment_name} --weight_dir {args.weight_dir} --seed {seed} --patch_idx {idx}', shell=True)

    if args.method == "keep":
        for seed in seeds:
            for idx in range(1,10):
                subprocess.run(f'python train_insurance_fullimgsize_mamba_mask_mostarea.py --mode train --train_path {args.train_path} --val_path {args.val_path} --experiment_name {args.experiment_name} --weight_dir {args.weight_dir} --seed {seed} --patch_idx {idx}', shell=True)
            
                subprocess.run(f'python train_insurance_fullimgsize_mamba_mask_mostarea.py --mode test --train_path {args.train_path} --val_path {args.test_path} --experiment_name {args.experiment_name} --weight_dir {args.weight_dir} --seed {seed} --patch_idx {idx}', shell=True)
