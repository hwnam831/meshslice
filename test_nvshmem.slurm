#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --gpu-bind=closest    
#SBATCH --cpus-per-task=16    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bcku-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --time=00:05:00      # hh:mm:ss for the job
#SBATCH -e test-nvshmem-A40x4-%j.err
#SBATCH -o test-nvshmem-A40x4-%j.out
### GPU options ###

module load nccl
module load nvhpc  # ... or any appropriate modules
module list  # job documentation and metadata
export NVSHMEM_HOME=$NVHPC_HOME/Linux_x86_64/22.11/comm_libs/nvshmem
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NVSHMEM_HOME/lib
cd /u/hwnam831/systolicparallelism/gpu_simulator
make
echo "job is starting on `hostname`"
##srun ./TestNVSHMEM 4096 512
srun ./TestNVSHMEM
