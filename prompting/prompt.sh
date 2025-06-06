#!/bin/bash
#SBATCH --time=12:30:00
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --mem=16GB
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=v100:1

module purge
module load Python/3.9.6-GCCcore-11.2.0

source $HOME/LTPProject/.venv/bin/activate

pip install --no-cache-dir -r $HOME/LTPProject/requirements.txt

export TRANSFORMERS_CACHE=/scratch/s4680340/huggingface/transformers
export HF_HOME=/scratch/s4680340/huggingface


export HF_TOKEN="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"

huggingface-cli login --token $HF_TOKEN

python huggingfacemodel.py 