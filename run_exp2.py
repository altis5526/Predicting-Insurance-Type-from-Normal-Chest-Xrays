import subprocess
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help='', type=str)
    parser.add_argument("method", help='', type=str)
    parser.add_argument("train_path", help='', type=str)
    parser.add_argument("val_path", help='', type=str)
    parser.add_argument("experiment_name", help='', type=str)
    parser.add_argument("weight_dir", help='', type=str)
    parser.add_argument("idx", help='', type=int)
    args = parser.parse_args()

    if args.method == "remove":
        if args.mode == "train":
            subprocess.run(f'python train_insurance_fullimgsize_mask_area.py --mode train --train_path {args.train_path} --val_path {args.val_path} --experiment_name {args.experiment_name} --weight_dir {args.weight_dir} --patch_idx {args.idx}', shell=True)
            
        if args.mode == "test":
            subprocess.run(f'python train_insurance_fullimgsize_mask_area.py --mode test --train_path {args.train_path} --val_path {args.val_path} --experiment_name {args.experiment_name} --weight_dir {args.weight_dir} --patch_idx {args.idx}', shell=True)

    if args.method == "keep":
        if args.mode == "train":
            subprocess.run(f'python train_insurance_fullimgsize_mask_mostarea.py --mode train --train_path {args.train_path} --val_path {args.val_path} --experiment_name {args.experiment_name} --weight_dir {args.weight_dir} --patch_idx {args.idx}', shell=True)
            
        if args.mode == "test":
            subprocess.run(f'python train_insurance_fullimgsize_mask_mostarea.py --mode test --train_path {args.train_path} --val_path {args.val_path} --experiment_name {args.experiment_name} --weight_dir {args.weight_dir} --patch_idx {args.idx}', shell=True)
