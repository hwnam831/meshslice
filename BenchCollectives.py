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

def AG_(buf):
    return jax.lax.all_gather(buf, 'x', tiled=True, axis=0)

def RS_(buf):
    return jax.lax.psum_scatter(buf, 'x', scatter_dimension=0, tiled=True)

def SendRecv_(buf, ndev):
    shift_up = partial(jax.lax.ppermute, axis_name='x',
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
    ndev = jax.device_count()
    devices = mesh_utils.create_device_mesh((ndev,))
    mesh = Mesh(devices, axis_names=('x',))
    print("Running benchmark on {} devices.".format(ndev))
    data_sizes = [4096*2**i for i in range(18)]
    myspec = P('x',None)
    AG = jax.jit(partial(shard_map, mesh=mesh, in_specs=myspec,
         out_specs=myspec)(AG_))
    RS = jax.jit(partial(shard_map, mesh=mesh, in_specs=myspec,
         out_specs=myspec)(RS_))
    SendRecv = jax.jit(partial(shard_map, mesh=mesh, in_specs=myspec,
         out_specs=myspec)(partial(SendRecv_, ndev=ndev)))
    
    print("\nAllgather: ")
    for dsize in data_sizes:
        buffer = jnp.ones([ndev*dsize//4096, 4096], dtype=jnp.float16)
        buf_split = jax.device_put(buffer, NamedSharding(mesh, myspec))
        benchtimes = repeatBench(AG, buf_split)
        print("Size: {}KB,\t Time: {}+-{}ms".format(2*dsize//1024, mean(benchtimes)*1000, stdev(benchtimes)*1000))

    print("\nReduceScatter: ")
    for dsize in data_sizes:
        buffer = jnp.ones([ndev*ndev*dsize//4096, 4096], dtype=jnp.float16)
        buf_split = jax.device_put(buffer, NamedSharding(mesh, myspec))
        benchtimes = repeatBench(RS, buf_split)
        print("Size: {}KB,\t Time: {}+-{}ms".format(2*dsize//1024, mean(benchtimes)*1000, stdev(benchtimes)*1000))

    print("\nSendRecv: ")
    for dsize in data_sizes:
        buffer = jnp.ones([ndev*dsize//4096, 4096], dtype=jnp.float16)
        buf_split = jax.device_put(buffer, NamedSharding(mesh, myspec))
        benchtimes = repeatBench(SendRecv, buf_split)
        print("Size: {}KB,\t Time: {}+-{}ms".format(2*dsize//1024, mean(benchtimes)*1000, stdev(benchtimes)*1000))




