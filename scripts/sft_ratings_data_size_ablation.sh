#!/bin/bash
#SBATCH -c 2  # Number of Cores per Task
#SBATCH --mem=30G  # Requested Memory
#SBATCH -p gpu-preempt # Partition
#SBATCH --gres=gpu:1  # Number of GPUs
#SBATCH -t 2-00:00:00  # Job time limit
#SBATCH -o ./jobs-exp/%j.out  # %j = job ID
#SBATCH --constraint="vram40"


# Create the jobs-exp directory if it doesn't exist
if [ ! -d "./jobs-exp" ]; then
    echo "Creating ./jobs directory..."
    mkdir -p ./jobs-exp
fi

model_repo="$1"
dataset_path="$2"
testset_path="$3"
output_base="$4"


# Run the Python script
export PYTHONPATH=$(pwd):$PYTHONPATH
python train/ratings_dataset_size_ablation.py --model_repo $model_repo --dataset_path $dataset_path --testset_path $testset_path --output_base $output_base 