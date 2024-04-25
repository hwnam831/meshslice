#!/bin/bash
#SBATCH --job-name=mutihost-jax-cpu
#SBATCH --partition=cpu
#SBATCH --time=72:00:00

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --chdir=/scratch/shared/beegfs/your_dir/
#SBATCH --output=/scratch/shared/beegfs/your_dir/%x-%j.out

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=16130
export WORLD_SIZE=4
export MY_NODE_ID=$SLURM_NODEID
echo "my node id="$MY_NODE_ID

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

### init virtual environment if needed
source ~/anaconda3/etc/profile.d/conda.sh
conda activate myenv

### the command to run
srun python main.py --net resnet18 \
--lr 1e-3 --epochs 50 --other_args