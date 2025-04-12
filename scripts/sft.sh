#!/bin/bash
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=80G  # Requested Memory
#SBATCH -p gpu-preempt # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 2-00:00:00  # Job time limit
#SBATCH -o ./jobs-exp/%j.out  # %j = job ID
#SBATCH --constraint=[bf16]

#Example Run
# Set Hugging Face cache path (replace with your desired path)
export TRANSFORMERS_CACHE=/project/pi_wenlongzhao_umass_edu/1/models

# Create the jobs-exp directory if it doesn't exist
if [ ! -d "./jobs-exp" ]; then
    echo "Creating ./jobs directory..."
    mkdir -p ./jobs-exp
fi

model_repo="$1"
dataset_path="$2"
save_path="$3"
epochs="$4"

# Run the Python script
export PYTHONPATH=$(pwd):$PYTHONPATH
python train/sft.py --model_repo "$model_repo" --dataset_path "$dataset_path" --save_path "$save_path" --epochs "$4"
