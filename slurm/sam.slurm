#!/bin/bash
#SBATCH --job-name=dev
#SBATCH --output=./slurm/dev.out
#SBATCH --error=./slurm/dev.out
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --hint=nomultithread
#SBATCH --time=05:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --cpus-per-task=8
#SBATCH --account=ads@v100

# Install segment-anything in a .local environment
# pip install git+https://github.com/facebookresearch/segment-anything.git --user --no-cache-dir

module purge

module load pytorch-gpu/py3/2.0.1

srun python lynx_id/preprocess/remove_background.py