#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --mem=16GB
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=v100:1

module purge
module load Python/3.9.6-GCCcore-11.2.0

source $HOME/LTP/.venv/bin/activate

pip install --no-cache-dir -r $HOME/LTP/requirements.txt

MODEL_KEY=${1:-llama}  # Default model is 'llama' unless passed as argument

export HUGGINGFACE_TOKEN="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"

python huggingfacemodel.py llama