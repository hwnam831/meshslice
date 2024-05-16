import os
from functools import partial

import jax
import jax.numpy as jnp

from jax.sharding import Mesh, PartitionSpec as P
from jax.sharding import NamedSharding
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.tree_util import tree_map, tree_all
import time
from statistics import mean, stdev

def AG_(buf, axis):
    return jax.lax.all_gather(buf, axis, tiled=True, axis=0)

def RS_(buf, axis):
    return jax.lax.psum_scatter(buf, axis, scatter_dimension=1, tiled=True)

def SendRecv_(buf, axis):
    ndev = jax.lax.psum(1, axis)
    shift_up = partial(jax.lax.ppermute, axis_name=axis,
                            perm=[(i, (i + 1) % ndev) for i in range(ndev)])
    return shift_up(buf)

def repeatBench(func, args, repeat=10, warmup=10):
    benchtimes = []

    for _ in range(warmup):
        func(args).block_until_ready()

    for _ in range(repeat):
        starttime = time.time()
        func(args).block_until_ready()
        func(args).block_until_ready()
        func(args).block_until_ready()
        func(args).block_until_ready()
        func(args).block_until_ready()
        func(args).block_until_ready()
        func(args).block_until_ready()
        func(args).block_until_ready()
        func(args).block_until_ready()
        func(args).block_until_ready()
        endtime = time.time()
        benchtimes.append((endtime - starttime)/10)
    return benchtimes

if __name__=='__main__':
    jax.distributed.initialize()
    print("Global device count: {}".format(jax.device_count()))
    print("Global device count: {}".format(jax.local_device_count()))
    colcount = jax.local_device_count()
    rowcount = jax.device_count() // colcount
    devices = mesh_utils.create_device_mesh((rowcount,colcount))
    mesh = Mesh(devices, axis_names=('x','y'))
    print("Running benchmark on {} devices.".format((rowcount,colcount)))
    data_sizes = [4096*2**i for i in range(17)]
    myspec = P('x','y')
    AG_y = jax.jit(partial(shard_map, mesh=mesh, in_specs=myspec,
         out_specs=myspec)(partial(AG_, axis='y')))
    RS_y = jax.jit(partial(shard_map, mesh=mesh, in_specs=myspec,
         out_specs=myspec)(partial(RS_, axis='y')))
    SendRecv_y = jax.jit(partial(shard_map, mesh=mesh, in_specs=myspec,
         out_specs=myspec)(partial(SendRecv_, axis='y')))
    
    print("\nAllgather_y: ")
    for dsize in data_sizes:
        buffer = jnp.ones([rowcount*dsize//4096, colcount*4096], dtype=jnp.bfloat16)
        buf_split = jax.device_put(buffer, NamedSharding(mesh, myspec))
        benchtimes = repeatBench(AG_y, buf_split)
        print("Size: {}KB,\t Time: {:.4f}+-{:.4f}ms".format(2*dsize//1024, mean(benchtimes)*1000, stdev(benchtimes)*1000))

    print("\nReduceScatter_y: ")
    for dsize in data_sizes:
        buffer = jnp.ones([rowcount*dsize//4096, colcount*colcount*4096], dtype=jnp.bfloat16)
        buf_split = jax.device_put(buffer, NamedSharding(mesh, myspec))
        benchtimes = repeatBench(RS_y, buf_split)
        print("Size: {}KB,\t Time: {:.4f}+-{:.4f}ms".format(2*dsize//1024, mean(benchtimes)*1000, stdev(benchtimes)*1000))

    print("\nSendRecv_y: ")
    for dsize in data_sizes:
        buffer = jnp.ones([rowcount*dsize//4096, colcount*4096], dtype=jnp.bfloat16)
        buf_split = jax.device_put(buffer, NamedSharding(mesh, myspec))
        benchtimes = repeatBench(SendRecv_y, buf_split)
        print("Size: {}KB,\t Time: {:.4f}+-{:.4f}ms".format(2*dsize//1024, mean(benchtimes)*1000, stdev(benchtimes)*1000))




