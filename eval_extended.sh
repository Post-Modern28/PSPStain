#!/bin/bash
#SBATCH --job-name=eval_psp_filtered
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval_psp_filtered.out
#SBATCH --error=logs/eval_psp_filtered.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate PSPStain
nvidia-smi
cd ~/benchmarking/PSPStain

echo "Evaluating all"
python eval_extended.py  --results_dir ~/benchmarking/PSPStain/results/MIST/val_pretrained/images --output_csv ~/benchmarking/PSPStain/results/MIST/val_pretrained/metrics.csv --batch_size 1 --num_workers 2

echo "Evaluating good"
python eval_extended.py  --results_dir ~/benchmarking/PSPStain/results/MIST/val_pretrained/good --output_csv ~/benchmarking/PSPStain/results/MIST/val_pretrained/metrics_good.csv --batch_size 1 --num_workers 2

echo "Evaluating norm"
python eval_extended.py  --results_dir ~/benchmarking/PSPStain/results/MIST/val_pretrained/norm --output_csv ~/benchmarking/PSPStain/results/MIST/val_pretrained/metrics_norm.csv --batch_size 1 --num_workers 2

echo "Evaluating combined"
python eval_extended.py  --results_dir ~/benchmarking/PSPStain/results/MIST/val_pretrained/comb --output_csv ~/benchmarking/PSPStain/results/MIST/val_pretrained/metrics_comb.csv --batch_size 1 --num_workers 2
echo "Done"