# MeshSlice

This repository contains Jax implementation of MeshSlice: Efficient 2D Tensor Parallelism for Distributed DNN Training ([paper](https://iacoma.cs.uiuc.edu/iacoma-papers/isca25_1.pdf)).
To cite the paper, please use the following bibtex entry.

```bibtex
@inproceedings{nam2025meshslice,
  title={MeshSlice: Efficient 2D Tensor Parallelism for Distributed DNN Training},
  author={Nam, Hyoungwook and Gerogiannis, Gerasimos and Torrellas, Josep},
  booktitle={2025 ACM/IEEE 52nd Annual International Symposium on Computer Architecture (ISCA)},
  year={2025}
}
```

MeshSlice is a framework with two components: 2D GeMM algorithm and autotuner to optimize the hyperparameters of the 2D GeMM.

See `TensorParallel.py` for the MeshSlice 2D GeMM implementation, and `Autotuner.py` for the autotuner implementation.

## Instruction - CPU

You can run the MeshSlice 2D GeMM algorithms by emulating the 2D device mesh with CPU.

Install CPU version of Jax with `pip install jax` in any supported system.

Then, follow `MeshSliceCPU.ipynb` to verify the correctness of MeshSlice 2D GeMM algorithms.

To run the autotuner, please run `Autotuner.py`.  
The autotuner will give the best mesh shape of the 2D cluster, the dataflow of each feed-forward layer, and the slice count (ksplit) for each FF layer as the example below.  
```
Best mesh shape is : (32, 8)
Layer: FeedForward:2     Dataflow: os    Transpose: False        ksplit: 8
Layer: FeedForward:4     Dataflow: ls    Transpose: False        ksplit: 8
Layer: FeedForward:7     Dataflow: os    Transpose: False        ksplit: 8
Layer: FeedForward:9     Dataflow: ls    Transpose: False        ksplit: 8
```
See `runautotune.sh` for examples and run `Autotuner.py --help` for the detailed options.

## Instructions - TPU

Most of the codes are written for Google Cloud TPU cluster.  
To run the code, please set-up the Cloud TPU ([link](https://cloud.google.com/tpu/docs/setup-gcp-account)).  
We recommend using tpuv4-32 (4 nodes of 4 TPUs) instance to run multi-host experiments (`BenchTransformer.py` and `BenchCollectives2D.py`).

Once a Cloud TPU cluster is up and running, run `./setup-tpupod.sh [tpupodname]` to install Jax and copy the source code to the TPU cluster.  
Then, execute `./runexp.sh [tpupodname] [NROW] [NCOL]` to benchmark GPT-3-like Transformer in $NROW\times NCOL$ device mesh.  
The execution profiles are available as tensorboard profiles under `/tmp/tensorboard` directory in the TPU cluster nodes.  
See Jax profiling [instructions](https://jax.readthedocs.io/en/latest/profiling.html) for accessing the profile data.  
You can run the experiments with different configurations. See `BenchTransformer.py` for available options.

The autotuner is configured using parameters collected from TPUv4.
For other HW architecture, use `BenchCollective2D.py` to benchmark the collective communications.  
Once you have the benchmark results, update `latencies`, `bws`, `base_overheads` and `eff_flops` in `Autotuner.py`.
