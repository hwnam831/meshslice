import os
import sys
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

def AG_(buf, axis, dim):
    return jax.lax.all_gather(buf, axis, tiled=True, axis=dim)

def RS_(buf, axis, dim):
    return jax.lax.psum_scatter(buf, axis, scatter_dimension=dim, tiled=True)

def SendRecv_(buf, axis):
    ndev = jax.lax.psum(1, axis)
    shift_up = partial(jax.lax.ppermute, axis_name=axis,
                            perm=[(i, (i + 1) % ndev) for i in range(ndev)])
    return shift_up(buf)

def repeatBench(func, args, repeat=10, warmup=10):
    benchtimes = []
    print(args.shape)
    for _ in range(warmup):
        func(args).block_until_ready()
    print(func(args).shape)
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

def createMultihostMatrix(mesh, sharding, global_shape): #only works for 2d
    local_shape = [global_shape[0]//mesh.devices.shape[0], global_shape[1]]
    print(local_shape)
    local_buffer = jax.random.normal(jax.random.PRNGKey(jax.process_index()),local_shape, dtype=jnp.bfloat16)
    local_sharded = jax.device_put(jnp.split(local_buffer,len(mesh.local_devices),axis=1), mesh.local_devices)
    return jax.make_array_from_single_device_arrays(global_shape,sharding, local_sharded)

if __name__=='__main__':
    jax.distributed.initialize()
    func_to_bench = sys.argv[1]
    print("Global device count: {}".format(jax.device_count()))
    print("Local device count: {}".format(jax.local_device_count()))
    colcount = jax.local_device_count()
    rowcount = jax.device_count() // colcount
    print(jax.devices())
    print(jax.local_devices())
    devices = mesh_utils.create_device_mesh((rowcount,colcount))
    mesh = Mesh(devices, axis_names=('x','y'))
    print("Running benchmark on {} devices.".format((rowcount,colcount)))
    data_sizes = [4096*2**i for i in range(16)]
    myspec = P('x','y')
    print("Along dim=0")
    AG_y = jax.jit(partial(shard_map, mesh=mesh, in_specs=myspec,
         out_specs=myspec)(partial(AG_, axis='y', dim=0)))
    RS_y = jax.jit(partial(shard_map, mesh=mesh, in_specs=myspec,
         out_specs=myspec)(partial(RS_, axis='y', dim=0)))
    SendRecv_y = jax.jit(partial(shard_map, mesh=mesh, in_specs=myspec,
         out_specs=myspec)(partial(SendRecv_, axis='y')))
    if func_to_bench == 'allgather':
        print("\nAllgather_y: dim=0")
        for dsize in data_sizes:
            #buffer = jnp.ones([rowcount*dsize//4096, colcount*4096], dtype=jnp.bfloat16)
            #buf_split = jax.device_put(buffer, NamedSharding(mesh, myspec))
            buf_split = createMultihostMatrix(mesh, NamedSharding(mesh, myspec),[rowcount*dsize//4096, colcount*4096])
            benchtimes = repeatBench(AG_y, buf_split)
            print("Size: {}KB,\t Time: {:.4f}+-{:.4f}ms".format(2*dsize//1024, mean(benchtimes)*1000, stdev(benchtimes)*1000))
        
        AG_y = jax.jit(partial(shard_map, mesh=mesh, in_specs=myspec,
            out_specs=myspec)(partial(AG_, axis='y', dim=1)))
        print("\nAllgather_y dim=1: ")
        for dsize in data_sizes:
            #buffer = jnp.ones([rowcount*dsize//4096, colcount*4096], dtype=jnp.bfloat16)
            #buf_split = jax.device_put(buffer, NamedSharding(mesh, myspec))
            buf_split = createMultihostMatrix(mesh, NamedSharding(mesh, myspec),[rowcount*dsize//4096, colcount*4096])
            benchtimes = repeatBench(AG_y, buf_split)
            print("Size: {}KB,\t Time: {:.4f}+-{:.4f}ms".format(2*dsize//1024, mean(benchtimes)*1000, stdev(benchtimes)*1000))
        
        AG_x = jax.jit(partial(shard_map, mesh=mesh, in_specs=myspec,
         out_specs=myspec)(partial(AG_, axis='x', dim=0)))
        print("\nAllgather_x: dim=0")
        for dsize in data_sizes:
            #buffer = jnp.ones([rowcount*dsize//4096, colcount*4096], dtype=jnp.bfloat16)
            #buf_split = jax.device_put(buffer, NamedSharding(mesh, myspec))
            buf_split = createMultihostMatrix(mesh, NamedSharding(mesh, myspec),[rowcount*dsize//4096, colcount*4096])
            benchtimes = repeatBench(AG_x, buf_split)
            print("Size: {}KB,\t Time: {:.4f}+-{:.4f}ms".format(2*dsize//1024, mean(benchtimes)*1000, stdev(benchtimes)*1000))
        AG_x = jax.jit(partial(shard_map, mesh=mesh, in_specs=myspec,
         out_specs=myspec)(partial(AG_, axis='x', dim=1)))
        print("\nAllgather_x: ")
        for dsize in data_sizes:
            #buffer = jnp.ones([rowcount*dsize//4096, colcount*4096], dtype=jnp.bfloat16)
            #buf_split = jax.device_put(buffer, NamedSharding(mesh, myspec))
            buf_split = createMultihostMatrix(mesh, NamedSharding(mesh, myspec),[rowcount*dsize//4096, colcount*4096])
            benchtimes = repeatBench(AG_x, buf_split)
            print("Size: {}KB,\t Time: {:.4f}+-{:.4f}ms".format(2*dsize//1024, mean(benchtimes)*1000, stdev(benchtimes)*1000))
        
    elif func_to_bench=='reducescatter':
        data_sizes = [4096*2**i for i in range(15)]
        print("\nReduceScatter_y: ")
        for dsize in data_sizes:
            #buffer = jnp.ones([rowcount*dsize//4096, colcount*colcount*4096], dtype=jnp.bfloat16)
            #buf_split = jax.device_put(buffer, NamedSharding(mesh, myspec))
            buf_split = createMultihostMatrix(mesh, NamedSharding(mesh, myspec),[colcount*rowcount*dsize//4096, colcount*4096])
            benchtimes = repeatBench(RS_y, buf_split)
            print("Size: {}KB,\t Time: {:.4f}+-{:.4f}ms".format(2*dsize//1024, mean(benchtimes)*1000, stdev(benchtimes)*1000))
        RS_y = jax.jit(partial(shard_map, mesh=mesh, in_specs=myspec,
            out_specs=myspec)(partial(RS_, axis='y', dim=1)))
        print("\nReduceScatter_y: dim=1")
        for dsize in data_sizes:
            #buffer = jnp.ones([rowcount*dsize//4096, colcount*colcount*4096], dtype=jnp.bfloat16)
            #buf_split = jax.device_put(buffer, NamedSharding(mesh, myspec))
            buf_split = createMultihostMatrix(mesh, NamedSharding(mesh, myspec),[rowcount*dsize//4096, colcount*colcount*4096])
            benchtimes = repeatBench(RS_y, buf_split)
            print("Size: {}KB,\t Time: {:.4f}+-{:.4f}ms".format(2*dsize//1024, mean(benchtimes)*1000, stdev(benchtimes)*1000))
        RS_x = jax.jit(partial(shard_map, mesh=mesh, in_specs=myspec,
            out_specs=myspec)(partial(RS_, axis='x', dim=0)))
        print("\nReduceScatter_x: ")
        for dsize in data_sizes:
            #buffer = jnp.ones([rowcount*dsize//4096, colcount*colcount*4096], dtype=jnp.bfloat16)
            #buf_split = jax.device_put(buffer, NamedSharding(mesh, myspec))
            buf_split = createMultihostMatrix(mesh, NamedSharding(mesh, myspec),[rowcount*rowcount*dsize//4096, colcount*4096])
            benchtimes = repeatBench(RS_x, buf_split)
            print("Size: {}KB,\t Time: {:.4f}+-{:.4f}ms".format(2*dsize//1024, mean(benchtimes)*1000, stdev(benchtimes)*1000))
        RS_x = jax.jit(partial(shard_map, mesh=mesh, in_specs=myspec,
            out_specs=myspec)(partial(RS_, axis='x', dim=1)))
        print("\nReduceScatter_x: dim=1")
        for dsize in data_sizes:
            #buffer = jnp.ones([rowcount*dsize//4096, colcount*colcount*4096], dtype=jnp.bfloat16)
            #buf_split = jax.device_put(buffer, NamedSharding(mesh, myspec))
            buf_split = createMultihostMatrix(mesh, NamedSharding(mesh, myspec),[rowcount*dsize//4096, rowcount*colcount*4096])
            benchtimes = repeatBench(RS_x, buf_split)
            print("Size: {}KB,\t Time: {:.4f}+-{:.4f}ms".format(2*dsize//1024, mean(benchtimes)*1000, stdev(benchtimes)*1000))
        

    else:
        print("\nSendRecv_y: ")
        
        for dsize in data_sizes:
            #buffer = jnp.ones([rowcount*dsize//4096, colcount*4096], dtype=jnp.bfloat16)
            #buf_split = jax.device_put(buffer, NamedSharding(mesh, myspec))
            buf_split = createMultihostMatrix(mesh, NamedSharding(mesh, myspec),[rowcount*dsize//4096, colcount*4096])
            benchtimes = repeatBench(SendRecv_y, buf_split)
            print("Size: {}KB,\t Time: {:.4f}+-{:.4f}ms".format(2*dsize//1024, mean(benchtimes)*1000, stdev(benchtimes)*1000))
        
        

        
        

        


        print("Along dim=0")
        
        
        SendRecv_x = jax.jit(partial(shard_map, mesh=mesh, in_specs=myspec,
            out_specs=myspec)(partial(SendRecv_, axis='x')))
        
        

        

        print("\nSendRecv_x: ")
        for dsize in data_sizes:
            #buffer = jnp.ones([rowcount*dsize//4096, colcount*4096], dtype=jnp.bfloat16)
            #buf_split = jax.device_put(buffer, NamedSharding(mesh, myspec))
            buf_split = createMultihostMatrix(mesh, NamedSharding(mesh, myspec),[rowcount*dsize//4096, colcount*4096])
            benchtimes = repeatBench(SendRecv_x, buf_split)
            print("Size: {}KB,\t Time: {:.4f}+-{:.4f}ms".format(2*dsize//1024, mean(benchtimes)*1000, stdev(benchtimes)*1000))

        print("Along dim=1")
        
        
        SendRecv_x = jax.jit(partial(shard_map, mesh=mesh, in_specs=myspec,
            out_specs=myspec)(partial(SendRecv_, axis='x')))
        
        

        