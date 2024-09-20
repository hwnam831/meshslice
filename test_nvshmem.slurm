#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2
#SBATCH --gpu-bind=closest    
#SBATCH --cpus-per-task=16    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA100x4      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bcku-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --time=00:20:00      # hh:mm:ss for the job
#SBATCH -e test-nvshmem-A100x2-%j.err
#SBATCH -o test-nvshmem-A100x2-%j.out
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
srun ./TestNVSHMEM 104857600 4096
srun ./TestNVSHMEM 104857600 8192
srun ./TestNVSHMEM 104857600 16384
srun ./TestNVSHMEM 104857600 32768
srun ./TestNVSHMEM 104857600 65536
srun ./TestNVSHMEM 104857600 131072
