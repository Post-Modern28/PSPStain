#!/bin/bash
#SBATCH --job-name=psp_test_lpips
#SBATCH --partition=ais-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=60:00:00
#SBATCH --output=../logs/test/psp_test_lpips.out
#SBATCH --error=../logs/test/psp_test_lpips.err

unset PYTHONHOME
unset PYTHONPATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate PSPStain
nvidia-smi
cd ~/benchmarking/PSPStain
mkdir -p logs/test

# ⚠️ Уточни name и dataroot для LPIPS если отличаются
declare -A NAMES=(
    ["PSPStainLPIPS15"]="lpips_15_patches"
    ["PSPStainLPIPS25"]="lpips_25_patches"
)

declare -A DATAROOTS=(
    ["PSPStainLPIPS15"]="$HOME/benchmarking/datasets/MIST_HER2_LPIPS_MPP_PATCHES_15"
    ["PSPStainLPIPS25"]="$HOME/benchmarking/datasets/MIST_HER2_LPIPS_MPP_PATCHES_25"
)


EXPERIMENTS=("PSPStainLPIPS15" "PSPStainLPIPS25")
EPOCHS=($(seq 5 5 80))

for exp in "${EXPERIMENTS[@]}"; do
    name="${NAMES[$exp]}"
    dataroot="${DATAROOTS[$exp]}"

    for epoch in "${EPOCHS[@]}"; do
        results_dir="./results/${name}/val_${epoch}/images"
        if [ -d "$results_dir" ] && [ "$(ls -A $results_dir 2>/dev/null)" ]; then
            echo "⏭ Пропускаем (уже есть): $name | epoch $epoch"
            continue
        fi

        echo "========================================="
        echo "Testing: $name | Epoch: $epoch"
        echo "========================================="

        python test.py \
            --dataroot "$dataroot" \
            --name "$name" \
            --checkpoints_dir checkpoints \
            --model PSPStain \
            --CUT_mode FastCUT \
            --netD n_layers \
            --ndf 32 \
            --netG resnet_6blocks \
            --n_layers_D 5 \
            --normG instance \
            --normD instance \
            --weight_norm spectral \
            --lambda_GAN 1.0 \
            --lambda_NCE 1.0 \
            --nce_layers 0,4,8,12,16 \
            --nce_T 0.07 \
            --num_patches 256 \
            --unet_seg MIST_unet_seg \
            --lambda_gp 10.0 \
            --gp_weights "[0.015625,0.03125,0.0625,0.125,0.25,1.0]" \
            --dataset_mode aligned \
            --direction AtoB \
            --load_size 1024 \
            --crop_size 1024 \
            --preprocess resize_and_crop \
            --display_winsize 512 \
            --phase val \
            --num_test 1000 \
            --epoch "$epoch"

        echo "Done: $name | epoch $epoch"
        echo
    done
done

echo "All testing completed (LPIPS)"
