#!/bin/bash
#SBATCH --job-name=dev
#SBATCH --output=slurm/cluster.out
#SBATCH --error=slurm/cluster.out
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --hint=nomultithread
#SBATCH --time=01:15:00
#SBATCH --qos=qos_gpu-dev
#SBATCH --cpus-per-task=8
#SBATCH --account=ads@v100

module purge

module load pytorch-gpu/py3/2.0.1

srun python cluster.py