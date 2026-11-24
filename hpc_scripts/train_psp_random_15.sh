#!/bin/bash
#SBATCH --job-name=15_1t
#SBATCH --partition=ais-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=26:00:00
#SBATCH --output=../logs/train_random_15_seed_1.out
#SBATCH --error=../logs/train_pandom_15_seed_1.err

unset PYTHONHOME
unset PYTHONPATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate PSPStain
nvidia-smi
cd ~/benchmarking/PSPStain

export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "Training..."
python -m experiments --name PSPStain_R15_S1 --cmd train --id 0 
echo "Testing..."
python -m experiments --name PSPStain_R15_S1 --cmd test --id 0
echo "Done!"

