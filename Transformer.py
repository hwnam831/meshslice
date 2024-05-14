import os
from functools import partial

from TensorParallel import SPMD
from Autotuner import ComputeGraph, build_transformerBlock, Autotuner

import jax
import jax.numpy as jnp

from jax.sharding import Mesh, PartitionSpec as P
from jax.sharding import NamedSharding
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.tree_util import tree_map, tree_all
import timeit

class Transformer:

    mesh: Mesh
    algorithm: str
    blocksize: int
    tuner: Autotuner
    def __init__(self, mesh, tuner:Autotuner B,S,H,D):
