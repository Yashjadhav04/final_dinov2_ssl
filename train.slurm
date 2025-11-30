#!/bin/bash
#SBATCH --account=csci_ga_2572-2025fa
#SBATCH --partition=c24m170-a100-2
#SBATCH --gres=gpu:2
#SBATCH --time=40:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=170G
#SBATCH --requeue
#SBATCH --output=train.out
#SBATCH --error=train.err

# Load environment
module load anaconda
conda activate ssl_env

echo "========= JOB STARTED ========="
echo "Running on host: $(hostname)"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"
echo "==============================="

# Run training
python train.py

echo "========= JOB FINISHED ========="
echo "Time: $(date)"
