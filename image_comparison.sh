#!/bin/bash
#SBATCH --job-name=image_comparison
#SBATCH --partition=mem
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=34:00:00
#SBATCH --output=logs/streamlit_%j.out
#SBATCH --error=logs/streamlit_%j.err

unset PYTHONHOME
unset PYTHONPATH

source ~/miniconda3/etc/profile.d/conda.sh
conda activate PSPStain

cd ~/benchmarking/PSPStain/image_comparison

export RESULTS_DIR=~/benchmarking/PSPStain/results
export PORT=18501

echo "Starting image comparison app"
echo "Node: $(hostname)"
echo "Port: ${PORT}"
echo "Results directory: ${RESULTS_DIR}"

bash run.sh