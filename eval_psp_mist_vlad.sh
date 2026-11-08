#!/bin/bash
#SBATCH --job-name=eval_psp
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_psp_mist_vlad.out
#SBATCH --error=logs/eval_psp_mist_vlad.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate PSPStain
nvidia-smi
cd ~/benchmarking/PSPStain
echo "Evaluating MIST"
python eval_by_vlad.py  --results_dir ~/benchmarking/PSPStain/results/MIST/val_pretrained/images --output_csv ~/benchmarking/PSPStain/results/MIST/val_pretrained/metrics.csv --batch_size 1 --num_workers 2
echo "Done"
