import os
from functools import partial

from TensorParallel import SPMD, createMultihostMatrix
from Autotuner import ComputeGraph, build_transformerBlock, Autotuner

import jax
import jax.numpy as jnp
from jax import custom_vjp

from jax.sharding import Mesh, PartitionSpec as P
from jax.sharding import NamedSharding
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.tree_util import tree_map, tree_all
import time
import sys

if __name__ == '__main__':
    jax.distributed.initialize()
    print("Global device count: {}".format(jax.device_count()))
    print("Local device count: {}".format(jax.local_device_count()))
    B=jax.device_count()
    S=2048
    H=96
    D=128
    alg = sys.argv[1]
    
    colcount = jax.local_device_count()
    rowcount = jax.device_count() // colcount
    #print(jax.devices())
    #print(jax.local_devices())
    devices = mesh_utils.create_device_mesh((rowcount,colcount))
    mesh = Mesh(devices, axis_names=('x','y'))
    x = createMultihostMatrix(mesh, NamedSharding(mesh, P('x','y')), [B*S,H*D])
    w = createMultihostMatrix(mesh, NamedSharding(mesh, P('x','y')), [H*D, 4*H*D])
    dy = createMultihostMatrix(mesh, NamedSharding(mesh, P('x','y')), [B*S,4*H*D])
    
    myalg = SPMD(mesh,alg)
    OS = myalg.OS()
    IS =myalg.IS()
    WS = myalg.WS()
    y = OS(x,w)
    dx = IS(dy,w)
    dw = WS(x,dy)

    
    jax.profiler.start_trace("/tmp/tensorboard")
    for _ in range(5):
        OS(x,w).block_until_ready()
    for _ in range(5):
        IS(dy,w).block_until_ready()
    for _ in range(5):
        WS(x,dy).block_until_ready()
    jax.profiler.stop_trace()
    starttime = time.time()
    for _ in range(10):
        OS(x,w).block_until_ready()
    endtime = time.time()
    print("Forward time: {:.3f}ms".format((endtime-starttime)/10))

    starttime = time.time()
    for _ in range(10):
        IS(dy,w).block_until_ready()
    endtime = time.time()
    print("Backward data time: {:.3f}ms".format((endtime-starttime)/10))

    starttime = time.time()
    for _ in range(10):
        WS(x,dy).block_until_ready()
    endtime = time.time()
    print("Backward weight time: {:.3f}ms".format((endtime-starttime)/10))