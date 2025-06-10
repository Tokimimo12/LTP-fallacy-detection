#!/bin/bash
#SBATCH --job-name=train_ltp_model        # Job name
#SBATCH --output=results-augmented-model-%j.log
#SBATCH --nodes=1                     # Number of nodes (use 1 node)
#SBATCH --ntasks=1                    # One task
#SBATCH --gpus-per-node=v100:1
#SBATCH --time=06:00:00              # Time limit for the job
#SBATCH --mem=20GB

# Remove all previously loaded modules
module purge

# Load Python module
module load Python/3.11.5-GCCcore-13.2.0

# Activate virtual environment
source $HOME/venvs/ltp/bin/activate

# Create result directory
mkdir -p /scratch/$USER/LTP/LTP-fallacy-detection/Saved_Plots/$SLURM_JOB_ID
mkdir -p /scratch/$USER/LTP/LTP-fallacy-detection/Saved_Models/$SLURM_JOB_ID

############ GETTING THE CODE
mkdir -p $TMPDIR 
mkdir -p $TMPDIR/Saved_Plots
mkdir -p $TMPDIR/Saved_Models

# Copy code into TMPDIR
echo "Started copying at $(date)"
cp -r /scratch/$USER/LTP/LTP-fallacy-detection $TMPDIR
echo "Finished copying at $(date)"

############ RUN CODE
cd $TMPDIR/LTP-fallacy-detection/Model

# Run training with parameter file
echo "About to run Python script at $(date)"
python3 -u train.py
####
# Run training with data from scratch
# python3 train.py --data_dir /scratch/$USER/DLP/'Task 2'/Data/splits --model_dir Models
####

############ SAVING RESULTS
# Save results to permanent storage
cp -r $TMPDIR/Saved_Plots /scratch/$USER/LTP/LTP-fallacy-detection/Saved_Plots/$SLURM_JOB_ID
cp -r $TMPDIR/Saved_Models /scratch/$USER/LTP/LTP-fallacy-detection/Saved_Models/$SLURM_JOB_ID
