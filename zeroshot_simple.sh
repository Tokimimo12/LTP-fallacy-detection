#!/bin/bash
#SBATCH --job-name=zeroshot_simple       # Job name
#SBATCH --output=zeroshot_simple-%j.log
#SBATCH --nodes=1                     # Number of nodes (use 1 node)
#SBATCH --ntasks=1                    # One task
#SBATCH --gpus-per-node=v100:1
#SBATCH --time=06:00:00              # Time limit for the job
#SBATCH --mem=12GB

# Remove all previously loaded modules
module purge

# Load Python module
module load Python/3.11.5-GCCcore-13.2.0

# Activate virtual environment
source $HOME/venvs/ltp/bin/activate

# Create result directory
mkdir -p /scratch/$USER/LTP/LTP-fallacy-detection/prompting_results/$SLURM_JOB_ID

############ GETTING THE CODE
mkdir -p $TMPDIR 
mkdir -p $TMPDIR/results

# Copy code into TMPDIR
echo "Started copying at $(date)"
cp -r /scratch/$USER/LTP/LTP-fallacy-detection $TMPDIR
echo "Finished copying at $(date)"

############ RUN CODE
cd $TMPDIR/LTP-fallacy-detection/prompting

# Set HuggingFace cache to TMPDIR which should have more space
export HF_HOME=$TMPDIR/huggingface
mkdir -p $HF_HOME

# export token
export HF_TOKEN="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"

huggingface-cli login --token $HF_TOKEN

# Run training with parameter file
echo "About to run Python script at $(date)"
python3 -u new_prompting.py --mode zero-shot --model llama
python3 -u new_prompting.py --mode zero-shot --model llama-instruct
python3 -u new_prompting.py --mode zero-shot --model menda
python3 -u new_prompting.py --mode zero-shot --model phi-4
python3 -u new_prompting.py --mode zero-shot --model mistralai
python3 -u new_prompting.py --mode zero-shot --model tinyllama

############ SAVING RESULTS
# Save results to permanent storage
cp -r $TMPDIR/LTP-fallacy-detection/prompting/results /scratch/$USER/LTP/LTP-fallacy-detection/prompting_results/$SLURM_JOB_ID
