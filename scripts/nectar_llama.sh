#!/bin/bash
#SBATCH -c 2  # Number of Cores per Task
#SBATCH --mem=30G  # Requested Memory
#SBATCH -p gpu-preempt # Partition
#SBATCH --gres=gpu:1  # Number of GPUs
#SBATCH -t 2-00:00:00  # Job time limit
#SBATCH -o ./jobs-exp/%j.out  # %j = job ID
#SBATCH --constraint="vram80"

#Example Run
#sbatch scripts/nectar.sh train /project/pi_wenlongzhao_umass_edu/1/absahoo/experiment_results/prometheusv2/nectar/relative_test.csv /project/pi_wenlongzhao_umass_edu/1/models/models--prometheus-eval--prometheus-7b-v2.0/snapshots/66ffb1fc20beebfb60a3964a957d9011723116c5 relative
#sbatch scripts/nectar.sh train /project/pi_wenlongzhao_umass_edu/1/absahoo/experiment_results/prometheusv2/nectar/asbolute_test.csv /project/pi_wenlongzhao_umass_edu/1/models/models--prometheus-eval--prometheus-7b-v2.0/snapshots/66ffb1fc20beebfb60a3964a957d9011723116c5 absolute

# Set Hugging Face cache path (replace with your desired path)
export TRANSFORMERS_CACHE=/project/pi_wenlongzhao_umass_edu/1/models

# Create the jobs-exp directory if it doesn't exist
if [ ! -d "./jobs-exp" ]; then
    echo "Creating ./jobs directory..."
    mkdir -p ./jobs-exp
fi

dataset_fold="$1"
output_file="$2"
model_repo="$3"
baseline_type="$4"

# Run the Python script
export PYTHONPATH=$(pwd):$PYTHONPATH
python datasets/nectar_llama.py --dataset_fold $dataset_fold --output_file $output_file --model_repo $model_repo --baseline_type $baseline_type

# sbatch scripts/nectar.sh train /project/pi_wenlongzhao_umass_edu/1/jkarnuthala/experiment_results/prometheusv2/nectar/asbolute_test.csv /datasets/ai/llama3/hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/33101ce6ccc08fa6249c10a543ebfcac65173393 absolute