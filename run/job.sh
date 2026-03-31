#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --hint=nomultithread
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH -J PaScaLTDMA
#SBATCH -o log_%j.out

source $MODULESHOME/init/bash
module purge
module load PrgEnv-nvidia
module load craype-accel-nvidia90

export MPICH_GPU_SUPPORT_ENABLED=1

cd $SLURM_SUBMIT_DIR

srun -N 1 -n 4 ./a.out
