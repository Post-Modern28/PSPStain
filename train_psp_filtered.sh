#!/bin/bash
#SBATCH --job-name=train_psp
#SBATCH --partition=ais-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=26:00:00
#SBATCH --output=logs/train_retccl_25.out
#SBATCH --error=logs/train_retccl_25.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate PSPStain
nvidia-smi
cd ~/benchmarking/PSPStain
echo "Training..."
python -m experiments --name PSPStain --cmd train --id 0
echo "Testing..."
python -m experiments --name PSPStain --cmd test --id 0
echo "Done!"

