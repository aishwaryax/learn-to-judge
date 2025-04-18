#!/bin/bash
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=40G  # Requested Memory
#SBATCH -p gpu,superpod-a100 # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 23:00:00  # Job time limit
#SBATCH -o ./jobs-exp/%j.out  # %j = job ID
#SBATCH --constraint=[vram80]

#sbatch scripts/create_embedding_pair.sh /project/pi_wenlongzhao_umass_edu/1/models/models--prometheus-eval--prometheus-7b-v2.0/snapshots/66ffb1fc20beebfb60a3964a957d9011723116c5 /project/pi_wenlongzhao_umass_edu/1/absahoo/experiment_results/prometheusv2/nectar/absolute_test.csv train_embeddings 2

# Set Hugging Face cache path (replace with your desired path)
export TRANSFORMERS_CACHE=/project/pi_wenlongzhao_umass_edu/1/models

# Create the jobs-exp directory if it doesn't exist
if [ ! -d "./jobs-exp" ]; then
    echo "Creating ./jobs directory..."
    mkdir -p ./jobs-exp
fi

model_repo="$1"
input_file="$2"
output_prefix="$3"
batch_size="$4"

python utils/create_sentence_embedding_pair.py \
      --model_repo "$model_repo" \
      --input_file "$input_file" \
      --output_prefix "$output_prefix" \
      --batch_size "$batch_size"
