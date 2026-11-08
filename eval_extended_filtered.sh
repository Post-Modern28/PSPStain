#!/bin/bash
#SBATCH --job-name=eval_psp_filtered
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval_retccl_25.out
#SBATCH --error=logs/eval_retccl_25.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate PSPStain
nvidia-smi
cd ~/benchmarking/PSPStain

mkdir -p 
echo "Evaluating all"

python eval_extended.py  --results_dir ~/benchmarking/PSPStain/results/retccl_25/val_latest/images --output_csv ~/benchmarking/PSPStain/results/retccl_25/val_latest/metrics.csv --batch_size 1 --num_workers 2

echo "Evaluating good"
python eval_extended.py  --results_dir ~/benchmarking/PSPStain/results/retccl_25/val_latest/good --output_csv ~/benchmarking/PSPStain/results/retccl_25/val_latest/metrics_good.csv --batch_size 1 --num_workers 2

echo "Evaluating norm"
python eval_extended.py  --results_dir ~/benchmarking/PSPStain/results/retccl_25/val_latest/norm --output_csv ~/benchmarking/PSPStain/results/retccl_25/val_latest/metrics_norm.csv --batch_size 1 --num_workers 2

echo "Evaluating combined"
python eval_extended.py  --results_dir ~/benchmarking/PSPStain/results/retccl_25/val_latest/comb --output_csv ~/benchmarking/PSPStain/results/retccl_25/val_latest/metrics_comb.csv --batch_size 1 --num_workers 2
echo "Done"