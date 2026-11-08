#!/bin/bash
#SBATCH --job-name=train_psp
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_psp_bci.out
#SBATCH --error=logs/train_psp_bci.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate PSPStain
nvidia-smi
cd ~/benchmarking/PSPStain
echo "Training..."
python -m experiments --name PSPStain --cmd train --id 0
echo "Done!"

