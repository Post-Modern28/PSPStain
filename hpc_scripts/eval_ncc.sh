#!/bin/bash
#SBATCH --job-name=psp_eval_ncc
#SBATCH --partition=ais-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=60:00:00
#SBATCH --output=logs/eval/psp_eval_ncc_epochs.out
#SBATCH --error=logs/eval/psp_eval_ncc_epochs.err

unset PYTHONHOME
unset PYTHONPATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate PSPStain
nvidia-smi
cd ~/benchmarking/PSPStain
mkdir -p logs/eval

NAMES=("ncc_15_patches_rerun" "ncc_25_patches")
EPOCHS=($(seq 5 5 80))

for name in "${NAMES[@]}"; do
    for epoch in "${EPOCHS[@]}"; do
        results_dir="./results/${name}/val_${epoch}/images"

        if [ ! -d "$results_dir" ] || [ -z "$(ls -A $results_dir 2>/dev/null)" ]; then
            echo "⚠ Пропускаем (нет результатов): $name | epoch $epoch"
            continue
        fi

        echo "========================================="
        echo "Evaluating: $name | Epoch: $epoch"
        echo "========================================="

        python $HOME/benchmarking/metrics/evaluate_metrics.py \
            --results_dir "$results_dir" \
            --batch_size 8 \
            --num_workers 4 \
            --image_format subdirs

        echo "Done: $name | epoch $epoch"
        echo
    done
done

echo "All evaluations completed (NCC)"
