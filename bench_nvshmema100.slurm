#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --gpu-bind=closest    
#SBATCH --cpus-per-task=16    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA100x4      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bcku-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --time=00:10:00      # hh:mm:ss for the job
#SBATCH -e bench-nvshmem-A100x4-%j.err
#SBATCH -o bench-nvshmem-A100x4-%j.out
### GPU options ###

module load nccl nvhpc  # ... or any appropriate modules
module list  # job documentation and metadata
export NVSHMEM_HOME=$NVHPC_HOME/Linux_x86_64/22.11/comm_libs/nvshmem
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NVSHMEM_HOME/lib
cd /u/hwnam831/systolicparallelism/gpu_simulator
make
echo "job is starting on `hostname`"
##srun ./TestNVSHMEM 4096 512
srun ./BenchNVSHMEM 209715200 8192 4
srun ./BenchNVSHMEM 209715200 8192 8
srun ./BenchNVSHMEM 209715200 8192 16
srun ./BenchNVSHMEM 209715200 8192 32
srun ./BenchNVSHMEM 209715200 16384 4
srun ./BenchNVSHMEM 209715200 16384 8
srun ./BenchNVSHMEM 209715200 16384 16
srun ./BenchNVSHMEM 209715200 16384 32
srun ./BenchNVSHMEM 209715200 32768 4
srun ./BenchNVSHMEM 209715200 32768 8
srun ./BenchNVSHMEM 209715200 32768 16
srun ./BenchNVSHMEM 209715200 32768 32
srun ./BenchNVSHMEM 209715200 65536 4
srun ./BenchNVSHMEM 209715200 65536 8
srun ./BenchNVSHMEM 209715200 65536 16
srun ./BenchNVSHMEM 209715200 65536 32
srun ./BenchNVSHMEM 209715200 131072 4
srun ./BenchNVSHMEM 209715200 131072 8
srun ./BenchNVSHMEM 209715200 131072 16
##srun ./BenchNVSHMEM 104857600 32768
##srun ./BenchNVSHMEM 104857600 65536
##srun ./BenchNVSHMEM 104857600 131072
##srun ./BenchNVSHMEM 104857600 $((1024*256))
##srun ./BenchNVSHMEM 104857600 $((1024*512))
##srun ./BenchNVSHMEM 104857600 $((1024*1024))
##srun ./BenchNVSHMEM $((2*104857600)) $((1024*1024))
