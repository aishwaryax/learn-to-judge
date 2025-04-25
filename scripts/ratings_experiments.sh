#!/bin/bash
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=40G  # Requested Memory
#SBATCH -p gpu,superpod-a100 # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 23:00:00  # Job time limit
#SBATCH -o ./jobs-exp/%j.out  # %j = job ID
#SBATCH --constraint=[vram40]

# Set Hugging Face cache path (replace with your desired path)
export TRANSFORMERS_CACHE=/project/pi_wenlongzhao_umass_edu/1/models

# Create the jobs-exp directory if it doesn't exist
if [ ! -d "./jobs-exp" ]; then
    echo "Creating ./jobs directory..."
    mkdir -p ./jobs-exp
fi

train_path="$1"
test_path="$2"
train_emb_path="$3"
test_emb_path="$4"
experiment_folder="$5"

export PYTHONPATH=$(pwd):$PYTHONPATH
python experiments/ratings.py --train_path $train_path --test_path $test_path --train_emb_path $train_emb_path --test_emb_path $test_emb_path --experiment_folder $experiment_folder