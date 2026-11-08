#!/bin/bash
#SBATCH --job-name=eval_psp
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval_psp_bci.out
#SBATCH --error=logs/eval_psp_bci.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ASP
nvidia-smi
cd ~/benchmarking/PSPStain
echo "Evaluating MIST"
python evaluate.py --epoch pretrained --dataset MIST
echo "Done"

