#!/bin/bash
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=120G  # Requested Memory
#SBATCH -p gpu-preempt # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 23:00:00  # Job time limit
#SBATCH -o ./jobs-exp/%j.out  # %j = job ID
#SBATCH --constraint=[vram80]

#Example Run
#sbatch scripts/judgelm.sh /project/pi_wenlongzhao_umass_edu/1/tbudhwani/experiment_results/prometheusv2/judgelm/relative_train.csv /project/pi_wenlongzhao_umass_edu/1/models/models--prometheus-eval--prometheus-7b-v2.0/snapshots/66ffb1fc20beebfb60a3964a957d9011723116c5 relative
#sbatch scripts/judgelm.sh /project/pi_wenlongzhao_umass_edu/1/tbudhwani/experiment_results/prometheusv2/judgelm/absolute_train.csv /project/pi_wenlongzhao_umass_edu/1/models/models--prometheus-eval--prometheus-7b-v2.0/snapshots/66ffb1fc20beebfb60a3964a957d9011723116c5 absolute

# Set Hugging Face cache path (replace with your desired path)
export TRANSFORMERS_CACHE=/project/pi_wenlongzhao_umass_edu/1/models

# Create the jobs-exp directory if it doesn't exist
if [ ! -d "./jobs-exp" ]; then
    echo "Creating ./jobs directory..."
    mkdir -p ./jobs-exp
fi

output_file="$1"
model_repo="$2"
baseline_type="$3"

# Run the Python script
export PYTHONPATH=$(pwd):$PYTHONPATH
python datasets/judgelm.py --output_file $output_file --model_repo $model_repo --baseline_type $baseline_type
