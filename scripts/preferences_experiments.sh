#!/bin/bash
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=40G  # Requested Memory
#SBATCH -p gpu # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 23:00:00  # Job time limit
#SBATCH -o ./jobs-exp/%j.out  # %j = job ID
#SBATCH --constraint=[vram23]

# Set Hugging Face cache path (replace with your desired path)
export TRANSFORMERS_CACHE=/project/pi_wenlongzhao_umass_edu/1/models

# Create the jobs-exp directory if it doesn't exist
if [ ! -d "./jobs-exp" ]; then
    echo "Creating ./jobs directory..."
    mkdir -p ./jobs-exp
fi

relative_train_path="$1"
relative_test_path="$2"
relative_train_emb_path="$3"
relative_test_emb_path="$4"
absolute_train_path="$5"
absolute_test_path="$6"
absolute_train_emb1_path="$7"
absolute_train_emb2_path="$8"
absolute_test_emb1_path="$9"
absolute_test_emb2_path="${10}"
experiment_folder="${11}"

export PYTHONPATH=$(pwd):$PYTHONPATH
python experiments/preferences.py --relative_train_path $relative_train_path --relative_test_path $relative_test_path --relative_train_emb_path $relative_train_emb_path --relative_test_emb_path $relative_test_emb_path --absolute_train_path $absolute_train_path --absolute_test_path $absolute_test_path --absolute_train_emb1_path $absolute_train_emb1_path --absolute_train_emb2_path $absolute_train_emb2_path --absolute_test_emb1_path $absolute_test_emb1_path --absolute_test_emb2_path $absolute_test_emb2_path --experiment_folder $experiment_folder