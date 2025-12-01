#!/bin/bash
#SBATCH --job-name=Adaptive_ICL
#SBATCH --partition=ampere    
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/rds/user/tm681/hpc-work/Adaptive_ICL/%x.%j.out
#SBATCH --error=/rds/user/tm681/hpc-work/Adaptive_ICL/%x.%j.err

module load cuda/12.2        # or the CUDA version CSD3 offers
source ~/venvs/wavelet/bin/activate   # if you use a virtualenv

python my_script.py
