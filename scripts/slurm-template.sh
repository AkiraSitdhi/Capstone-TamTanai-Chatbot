#!/bin/bash
#SBATCH -p dgx                 # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1                   # Specify number of GPU per task, Specify tasks per node
#SBATCH -t 120:00:00           # Specify maximum time limit (hour: minute: second)
#SBATCH -A <>            # Specify project name
#SBATCH -J <>           # Specify job name

source <HPC Conda ENV Path>
conda activate <Conda ENV name>
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 <filename>.py