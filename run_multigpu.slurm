#!/bin/bash
#SBATCH --account=bcku-delta-gpu
#SBATCH --job-name=mutihost-jax-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --time=00:20:00

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --chdir=/u/hwnam831/systolicparallelism/
#SBATCH --output=/u/hwnam831/logoutputs/%x-%j.out



### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=16137
export WORLD_SIZE=$SLURM_NNODES
export MY_NODE_ID=$SLURM_NODEID
echo "my node id="$MY_NODE_ID

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

### init virtual environment if needed
module load anaconda3_gpu

### the command to run
srun python Multihost.py
exit