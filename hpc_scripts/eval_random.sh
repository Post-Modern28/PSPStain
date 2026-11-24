#!/bin/bash
#SBATCH --job-name=psp_r15e
#SBATCH --partition=ais-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --output=../logs/eval_random_seeds.out
#SBATCH --error=../logs/eval_seeds.err

unset PYTHONHOME
unset PYTHONPATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate PSPStain
nvidia-smi
cd ~/benchmarking/PSPStain


EXPERIMENTS=(
    "random_15_seed_1"
    "random_15_seed_2"
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