#!/bin/bash
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=40G  # Requested Memory
#SBATCH -p gpu # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 2-00:00:00  # Job time limit
#SBATCH -o ./jobs-exp/%j.out  # %j = job ID
#SBATCH --constraint="vram80"

#sbatch scripts/create_embedding.sh /project/pi_wenlongzhao_umass_edu/1/models/models--prometheus-eval--prometheus-7b-v2.0/snapshots/66ffb1fc20beebfb60a3964a957d9011723116c5 /project/pi_wenlongzhao_umass_edu/1/absahoo/experiment_results/prometheusv2/nectar/absolute_test.csv train_embeddings 2

# Set Hugging Face cache path (replace with your desired path)
export TRANSFORMERS_CACHE=/project/pi_wenlongzhao_umass_edu/1/models

# Create the jobs-exp directory if it doesn't exist
if [ ! -d "./jobs-exp" ]; then
    echo "Creating ./jobs directory..."
    mkdir -p ./jobs-exp
fi

train_path1="$1"
test_path1="$2"
train_path2="$3"
test_path2="$4"
train_output1="$5"
test_output1="$6"
train_output2="$7"
test_output2="$8"

python utils/create_pca_embedding_pair.py \
      --train_path1 "$train_path1" \
      --test_path1 "$test_path1" \
      --train_path2 "$train_path2" \
      --test_path2 "$test_path2" \
      --train_output1 "$train_output1" \
      --test_output1 "$test_output1" \
      --train_output2 "$train_output2" \
      --test_output2 "$test_output2"
