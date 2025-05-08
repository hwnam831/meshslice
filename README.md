# MeshSlice

This repository contains Jax implementation of MeshSlice: Efficient 2D Tensor Parallelism for Distributed DNN Training ([paper](github.com)).

MeshSlice is a framework with two components: 2D GeMM algorithm and autotuner to optimize the hyperparameters of the 2D GeMM.

See `TensorParallel.py` for the MeshSlice 2D GeMM implementation, and `Autotuner.py` for the autotuner implementation.

## Instruction - CPU

You can run the MeshSlice 2D GeMM algorithms by emulating the 2D device mesh with CPU.

Install CPU version of Jax with `pip install jax` in any supported system.

Then, follow `MeshFlowCPU.ipynb` to verify the correctness of MeshSlice 2D GeMM algorithms.

## Instructions - TPU

Most of the codes are written for Google Cloud TPU cluster.  
To run the code, please set-up the Cloud TPU ([link](https://cloud.google.com/tpu/docs/setup-gcp-account)).  
We recommend using tpuv4-32 (4 nodes of 4 TPUs) instance to run multi-host experiments (`BenchTransformer.py` and `BenchCollectives2D.py`) and tpuv4-8 (single node of 4 TPUS) to run the Autotuner (`Autotuner.py`).

Once a Cloud TPU cluster is up and running, run `./setup-tpupod.sh [tpupodname]` to install Jax and copy the source code to the TPU cluster.  
Then, execute `./runexp.sh [tpupodname] [NROW] [NCOL]` to benchmark GPT-3-like Transformer in $NROW\times NCOL$ device mesh.  
The execution profiles are available as tensorboard profiles under `/tmp/tensorboard` directory in the TPU cluster nodes.  
See Jax profiling [instructions](https://jax.readthedocs.io/en/latest/profiling.html) for accessing the profile data.  
You can run the experiments with different configurations. See `BenchTransformer.py` for available options.

The autotuner is designed to run in single-node TPU VM (tpuv4-8).
Set-up the single-node TPU VM using `setup-tpupod.sh` and run `./runautotune.sh [tpupodname]`.  
The autotuner's communication cost model is parametrized for TPUv4.  
For other HW architecture, use `BenchCollective2D.py` to benchmark the collective communications.  
Once you have the benchmark results, update `latencies`, `bws`, `base_overheads` in `Autotuner.py`.
