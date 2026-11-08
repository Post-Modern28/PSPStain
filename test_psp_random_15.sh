#!/bin/bash
#SBATCH --job-name=psp_r30t
#SBATCH --partition=ais-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=26:00:00
#SBATCH --output=logs/train_random_30.out
#SBATCH --error=logs/train_pandom_30.err

unset PYTHONHOME
unset PYTHONPATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate PSPStain
nvidia-smi
cd ~/benchmarking/PSPStain
echo "Training..."
python -m experiments --name PSPStain --cmd train --id 0
echo "Testing..."
python -m experiments --name PSPStain --cmd test --id 0
echo "Done!"

