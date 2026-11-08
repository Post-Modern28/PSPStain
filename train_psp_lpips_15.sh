#!/bin/bash
#SBATCH --job-name=psp_lp_15t
#SBATCH --partition=ais-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=26:00:00
#SBATCH --output=logs/train_lpips_15.out
#SBATCH --error=logs/train_lpips_15.err

unset PYTHONHOME
unset PYTHONPATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate PSPStain
nvidia-smi
cd ~/benchmarking/PSPStain
echo "Training..."
python -m experiments --name PSPStainLPIPS15 --cmd train --id 0
echo "Testing..."
python -m experiments --name PSPStainLPIPS15 --cmd test --id 0
echo "Done!"

EXPERIMENTS=(
    "lpips_15_patches"
)

for exp in "${EXPERIMENTS[@]}"; do
    echo "========================================="
    echo "Processing: $exp"
    echo "========================================="

    python ~/benchmarking/metrics/evaluate_metrics.py \
        --results_dir ./results/${exp}/val_latest/images \
        --batch_size 20 \
        --num_workers 4 \
        --image_format subdirs

    echo "Done: $exp"
    echo
done