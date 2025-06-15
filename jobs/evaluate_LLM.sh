#!/bin/bash
#SBATCH --job-name=train_stl_ltp_model        # Job name
#SBATCH --output=results-simple-model-%j.log
#SBATCH --nodes=1                     # Number of nodes (use 1 node)
#SBATCH --ntasks=1                    # One task
#SBATCH --gpus-per-node=v100:1
#SBATCH --time=0:20:00              # Time limit for the job

# Remove all previously loaded modules
module purge

# Load Python module
module load Python/3.9.6-GCCcore-11.2.0

# Activate virtual environment
source $HOME/venvs/ltp/bin/activate

# Create result directory
mkdir -p /scratch/$USER/LTP/LTP-fallacy-detection/Saved_Plots/$SLURM_JOB_ID
mkdir -p /scratch/$USER/LTP/LTP-fallacy-detection/Saved_Models/$SLURM_JOB_ID
mkdir -p /scratch/$USER/LTP/LTP-fallacy-detection/Saved_Test_Metrics/$SLURM_JOB_ID

############ GETTING THE CODE
mkdir -p $TMPDIR 
mkdir -p $TMPDIR/Saved_Plots
mkdir -p $TMPDIR/Saved_Models
mkdir -p $TMPDIR/Saved_Test_Metrics

# Copy code into TMPDIR
echo "Started copying at $(date)"
cp -r /scratch/$USER/LTP/LTP-fallacy-detection $TMPDIR
echo "Finished copying at $(date)"

cd $TMPDIR/LTP-fallacy-detection/Model

# Run training with parameter file
echo "About to run Python script at $(date)"
python3 -u test_eval.py --job_id 17869137 --head_type_list "HTC" --augment_list "LLM"
# python3 -u train.py


############ SAVING RESULTS
# Save results to permanent storage
cp -r $TMPDIR/Saved_Plots /scratch/$USER/LTP/LTP-fallacy-detection/Saved_Plots/$SLURM_JOB_ID
cp -r $TMPDIR/Saved_Models /scratch/$USER/LTP/LTP-fallacy-detection/Saved_Models/$SLURM_JOB_ID
cp -r $TMPDIR/Saved_Test_Metrics /scratch/$USER/LTP/LTP-fallacy-detection/Saved_Test_Metrics/$SLURM_JOB_ID