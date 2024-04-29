#!/bin/bash

#SBATCH --job-name=VO
#SBATCH --nodes=1
#SBATCH --gres=gpu:4

#SBATCH --mem=16g
#SBATCH --partition=academic
#SBATCH --output=sbatch/VIO2_output.txt
#SBATCH -t 02:00:00

# Load the module system
source /etc/profile.d/modules.sh

# Load CUDA toolkit
module load cuda12.1/toolkit

# Initialize and activate Conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pylight

python train.py --data_file data/trajectories.json --mode 'VIO' --epochs 50 --batch_size 4 --gpus 4