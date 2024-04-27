from TensorParallel import SPMD, allclose

from functools import partial

import jax
import jax.numpy as jnp

from jax.sharding import Mesh, PartitionSpec as P
from jax.sharding import NamedSharding
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.tree_util import tree_map, tree_all
import os
import sys
#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8' # Use 8 CPU devices
#jax.config.update('jax_platform_name', 'cpu')
if __name__ == '__main__':
    hostip = os.environ["MASTER_ADDR"]
    num_proc= int(os.environ["SLURM_NTASKS"])
    num_nodes= int(os.environ["SLURM_NNODES"])
    
    pid = int(os.environ["SLURM_PROCID"])
    mport = os.environ["MASTER_PORT"]
    print("ntasks: {}, nnodes: {}, pid:{}".format(num_proc, num_nodes, pid))
    jax.distributed.initialize(coordinator_address=hostip+':1234', num_processes=num_proc, process_id=pid)
    print("Global device count: {}".format(jax.device_count()))
    devices = mesh_utils.create_device_mesh((jax.device_count()//2, 2))
    mesh = Mesh(devices, axis_names=('x', 'y'))
    B,S,H,D = (4,128, 48,64)

    X = jnp.arange( B*S*H*D,dtype=jnp.float32).reshape(B*S, H*D)/(B*S*H*D)
    W = jnp.arange(H*D*4*H*D,dtype=jnp.float32).reshape(H*D, 4*H*D) / (4*H*D*H*D)
    Y = X@W
    myalg = SPMD(mesh,'systolic')
    
    Xo = jax.device_put(X, NamedSharding(mesh, P('x', 'y')))
    Wo = jax.device_put(W, NamedSharding(mesh, P('x', 'y')))
    
    OS = myalg.OS()
    IS = myalg.IS()
    WS = myalg.WS()

    Yo = OS(Xo,Wo)
    print(allclose(Yo,X@W))
    Xp = IS(Yo,Wo)
    print(allclose(Xp,Y@W.transpose()))
    Wp = WS(Xo,Yo)
    print(allclose(Wp,X.transpose()@Y))