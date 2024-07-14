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


class ShardedFFLayer:
    mesh: Mesh
    algorithm: str
    dataflow:str
    blocksize: int
    ksplit: int
    def __init__(self, mesh, algorithm, dataflow, ksplit, in_dim, out_dim, blocksize=8):
        self.mesh = mesh
        row,col = mesh.axis_names
        self.algorithm = algorithm
        self.dataflow = dataflow
        self.ksplit = ksplit
        self.blocksize = blocksize
        if dataflow == 'ls':
            self.weight = createMultihostMatrix(mesh, NamedSharding(mesh, P(row,col)), [out_dim, in_dim])
        else:
            self.weight = createMultihostMatrix(mesh, NamedSharding(mesh, P(row,col)), [in_dim, out_dim])
        SPMDbuilder = SPMD(mesh, algorithm, blocksize)
        OS = SPMDbuilder.OS(ksplit)
        LS = SPMDbuilder.LS(ksplit)
        RS = SPMDbuilder.RS(ksplit)
        if dataflow=='os':
            @custom_vjp
            def compute(x, w):
                #print('OS compute func')
                return OS(x, w)
            def compute_fwd(x, w):
                #print('compute_fwd')
                return compute(x, w), (x, w)
            def compute_bwd(res, dy):
                #print('OS compute_bwd')
                x,w = res
                dx = LS(dy, w)
                dw = RS(x, dy)
                return (dx, dw)
            compute.defvjp(compute_fwd, compute_bwd)
        elif dataflow=='ls':
            @custom_vjp
            def compute(x, w):
                return LS(x, w)
            def compute_fwd(x, w):
                #print('compute_fwd')
                return compute(x, w), (x, w)
            def compute_bwd(res, dy):
                x,w = res
                dx = OS(dy, w)
                dw = RS(dy, x)
                return (dx, dw)
            compute.defvjp(compute_fwd, compute_bwd)
        elif dataflow=='rs':
            @custom_vjp
            def compute(x, w):
                return RS(x, w)
            def compute_fwd(x, w):
                #print('compute_fwd')
                return compute(x, w), (x, w)
            def compute_bwd(res, dy):
                x,w = res
                dx = LS(w, dy)
                dw = OS(x,dy)
                return (dx, dw)
            compute.defvjp(compute_fwd, compute_bwd)

        self.forward = partial(compute, w=self.weight)

class ShardedLayerNorm:
    mesh: Mesh
    def __init__(self, mesh, in_dim):
        self.mesh = mesh
        row, col = mesh.axis_names
        local_gamma = jax.device_put(
            jnp.split(jnp.ones([in_dim], dtype=jnp.bfloat16), len(mesh.local_devices)),
            mesh.local_devices)
        self.gamma = jax.make_array_from_single_device_arrays(
            [in_dim], NamedSharding(mesh, P(col)), local_gamma)

        local_bias = jax.device_put(
            jnp.split(jnp.zeros([in_dim], dtype=jnp.bfloat16), len(mesh.local_devices)),
            mesh.local_devices)
        self.bias = jax.make_array_from_single_device_arrays(
            [in_dim], NamedSharding(mesh, P(col)), local_bias)
        '''
        #@jax.jit
        @partial(shard_map,mesh=mesh,in_specs=(P(col), P(col),P(row,col)),
                 out_specs=P(row,col))
        def _layernorm(g,b,x):
            #print(g.shape)
            #print(b.shape)
            #ndev = jax.lax.psum(1,mesh.axis_names[1])
            ndev = mesh.devices.shape[1]
            mn_l = jnp.mean(x,axis=-1)
            sqmn_l = jnp.mean(x*x, axis=-1)
            mn = jax.lax.psum(mn_l, mesh.axis_names[1])/ndev
            #print(mn.shape)
            sqmn = jax.lax.psum(sqmn_l, mesh.axis_names[1])/ndev
            #print(sqmn.shape)
            stdev = 1/jnp.sqrt(sqmn - mn*mn + 1e-6)
            out =  g*(x-mn[:,None])*stdev[:,None] + b
            #print("layernorm shard shape: {}".format(out.shape))
            return out
        self.forward = jax.jit(partial(_layernorm, self.gamma, self.bias))
        '''
        #only local layernorm
        @partial(shard_map,mesh=mesh,in_specs=(P(row,col)),
                 out_specs=P(row,col))
        def _layernorm(x):
            ndev = jax.lax.psum(1,mesh.axis_names[1])
            mn_l = jnp.mean(x,axis=-1)
            sqmn_l = jnp.mean(x*x, axis=-1)
            mn = jax.lax.psum(mn_l, mesh.axis_names[1])/ndev
            #print(mn.shape)
            sqmn = jax.lax.psum(sqmn_l, mesh.axis_names[1])/ndev
            #print(sqmn.shape)
            stdev = 1/jnp.sqrt(sqmn - mn*mn + 1e-6)
            out =  (x-mn[:,None])*stdev[:,None]
            return out
        

        self.forward = jax.jit(_layernorm)
        

class ShardedAttention:
    mesh: Mesh
    def __init__(self, mesh, n_heads, seqlen):
        self.mesh = mesh
        self.heads_per_shard = n_heads // mesh.devices.shape[1]
        self.seqlen = seqlen
        row,col = mesh.axis_names
        @partial(shard_map,mesh=mesh,in_specs=P(row,col),
                 out_specs=P(row,col))
        def _attn(Xij):
            #print(Xij.shape)
            Q,K,V = jnp.split(Xij.reshape(Xij.shape[0]//self.seqlen, 
                            self.seqlen,self.heads_per_shard,-1), 3, axis=-1)
            #print(V.shape)
            #q: num_queries=S, k: num_keys=num_vals=S
            attn_weights = jax.nn.softmax(jnp.einsum('bqhd,bkhd->bhqk',Q,K)/jnp.sqrt(Q.shape[-1]),axis=3)
            return jnp.einsum('bhqk,bkhd->bqhd',attn_weights,V).reshape(Xij.shape[0],-1)

        self.forward = jax.jit(_attn)
    
class TransformerBlock:
    def __init__(self, mesh, S,H,D, dataflows, alg='meshflow', ksplits=[4,4,4,4]):
        self.mesh = mesh
        self.norm1 = ShardedLayerNorm(mesh, H*D)
        self.in_proj = ShardedFFLayer(mesh, alg, dataflows[0], ksplits[0], H*D, 3*H*D)
        self.attn = ShardedAttention(mesh, H, S)
        self.out_proj = ShardedFFLayer(mesh, alg, dataflows[1], ksplits[1], H*D, H*D)
        self.norm2 = ShardedLayerNorm(mesh, H*D)
        self.ff1 = ShardedFFLayer(mesh, alg, dataflows[2], ksplits[2], H*D, 4*H*D)
        self.ff2 = ShardedFFLayer(mesh, alg, dataflows[3], ksplits[3], 4*H*D, H*D)
        self.norm3 = ShardedLayerNorm(mesh, H*D)
    def _forward(self, x):
        out = self.norm1.forward(x)
        #out = x
        out = self.in_proj.forward(out)
        out = self.attn.forward(out)
        out = self.out_proj.forward(out)
        res = self.norm2.forward(out + x)
        #res = out+x
        #print(x2.shape)
        out = self.ff1.forward(res)
        out = jax.nn.gelu(out)
        out = self.ff2.forward(out)
        out = self.norm3.forward(out+res)
        
        return out
        
    def forward(self, x):
        out, self.backward = jax.vjp(self._forward, x)
        return out

if __name__ == '__main__':
    B=32
    S=2048
    H=96
    D=128
    alg = sys.argv[1]
    jax.distributed.initialize()
    print("Global device count: {}".format(jax.device_count()))
    print("Local device count: {}".format(jax.local_device_count()))
    colcount = jax.local_device_count()
    rowcount = jax.device_count() // colcount
    #print(jax.devices())
    #print(jax.local_devices())
    devices = mesh_utils.create_device_mesh((rowcount,colcount))
    mesh = Mesh(devices, axis_names=('x','y'))
    x = createMultihostMatrix(mesh, NamedSharding(mesh, P('x','y')), [B*S,H*D])
    dy = createMultihostMatrix(mesh, NamedSharding(mesh, P('x','y')), [B*S,H*D])
    
    model = TransformerBlock(mesh, S, H, D, dataflows=['os','os','os','ls'], alg=alg)
    out = model.forward(x)
    grads = model.backward(dy)
    out.block_until_ready()
    grads[0].block_until_ready()
    print(out.shape)
    jax.profiler.start_trace("/tmp/tensorboard")
    for _ in range(5):
        out = model._forward(x)
        out.block_until_ready()
    for _ in range(5):
        grads = model.backward(dy)
        grads[0].block_until_ready()
    jax.profiler.stop_trace()
    starttime = time.time()
    for _ in range(10):
        out = model._forward(x)
        out.block_until_ready()
    endtime = time.time()
    print("Forward time: {:.3f}ms".format((endtime-starttime)/10))
    starttime = time.time()
    for _ in range(10):
        grads = model.backward(dy)
        grads[0].block_until_ready()
    endtime = time.time()
    print("Backward time: {:.3f}ms".format((endtime-starttime)/10))