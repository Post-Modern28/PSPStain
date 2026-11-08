#!/bin/bash
#SBATCH --job-name=psp_r_te
#SBATCH --partition=ais-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/test_psp_random.out
#SBATCH --error=logs/test_psp_random.err

unset PYTHONHOME
unset PYTHONPATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate PSPStain
nvidia-smi
cd ~/benchmarking/PSPStain
echo "Testing 15..."
python -m experiments --name PSPStainR15 --cmd test --id 0

echo "Testing 20..."
python -m experiments --name PSPStainR20 --cmd test --id 0

echo "Testing 25..."
python -m experiments --name PSPStainR25 --cmd test --id 0

echo "Testing 30..."
python -m experiments --name PSPStainR30 --cmd test --id 0
echo "Done!"


