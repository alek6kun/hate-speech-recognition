#!/bin/bash -l
#SBATCH --chdir /scratch/izar/limozin/hate-speech-recognition
#SBATCH --job-name smart-deberta-training
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 8G
#SBATCH --time 1:00:00
#SBATCH --gres gpu:1
#SBATCH --account ee-559
#SBATCH --qos ee-559

echo "Starting job on $(hostname)"
echo "Loading modules..."
module load gcc/11.3.0 intel/2021.6.0 python/3.10.4 cuda/12.1.1

echo "Activating virtual environment..."
source /home/limozin/venvs/course_py-3.10/bin/activate  # Ensure this points to your virtual environment

echo "Running Python script..."
python Training.py

echo "Job completed"