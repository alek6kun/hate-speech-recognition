#!/bin/bash -l
#SBATCH --chdir /home/limozin/hate-speech-recognition
#SBATCH --job-name smart-deberta-training
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 32G
#SBATCH --time 12:00:00
#SBATCH --gres gpu:1
#SBATCH --account ee-559
#SBATCH --qos ee-559
#SBATCH --partition <specify-your-partition-if-needed>

echo "Starting job on $(hostname)"
echo "Loading modules..."
module load python/3.10.4
module load cuda/12.1.1

echo "Activating virtual environment..."
source /home/limozin/venvs/course_py-3.10/bin/activate  # Ensure this points to your virtual environment

echo "Running Python script..."
python train_model.py

echo "Job completed"
