#!/bin/bash
#SBATCH --job-name=psp_l_e
#SBATCH --partition=ais-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=5:00:00
#SBATCH --output=logs/eval_psp_lpips.out
#SBATCH --error=logs/eval_psp_lpips.err


unset PYTHONHOME
unset PYTHONPATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate PSPStain
nvidia-smi

cd ~/benchmarking/PSPStain

EXPERIMENTS=(
    "lpips_15"
    "lpips_25"
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

echo "All experiments completed"
