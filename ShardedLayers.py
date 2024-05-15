import os
from functools import partial

from TensorParallel import SPMD
from Autotuner import ComputeGraph, build_transformerBlock, Autotuner

import jax
import jax.numpy as jnp
from jax import custom_vjp

from jax.sharding import Mesh, PartitionSpec as P
from jax.sharding import NamedSharding
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.tree_util import tree_map, tree_all
import timeit
import flax.linen as nn



class ShardedFFLayer:
    mesh: Mesh
    algorithm: str
    dataflow:str
    blocksize: int
    ksplit: int
    def __init__(self, mesh, algorithm, dataflow, ksplit, in_dim, out_dim, blocksize=8):
        self.mesh = mesh
        self.algorithm = algorithm
        self.dataflow = dataflow
        self.ksplit = ksplit
        self.blocksize = blocksize
        if dataflow == 'is':
            self.weight = jax.random.normal(jax.random.PRNGKey(1),[out_dim,in_dim], dtype=jnp.bfloat16)
        else:
            self.weight = jax.random.normal(jax.random.PRNGKey(2),[in_dim,out_dim], dtype=jnp.bfloat16)
        SPMDbuilder = SPMD(mesh, algorithm, blocksize)
        OS = SPMDbuilder.OS(ksplit)
        IS = SPMDbuilder.IS(ksplit)
        WS = SPMDbuilder.WS(ksplit)
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
                dx = IS(dy, w)
                dw = WS(x, dy)
                return (dx, dw)
            compute.defvjp(compute_fwd, compute_bwd)
        elif dataflow=='is':
            @custom_vjp
            def compute(x, w):
                #print('IS compute func')
                return IS(x, w)
            def compute_fwd(x, w):
                #print('compute_fwd')
                return compute(x, w), (x, w)
            def compute_bwd(res, dy):
                #print('IS compute_bwd')
                x,w = res
                dx = OS(dy, w)
                dw = WS(dy, x)
                return (dx, dw)
            compute.defvjp(compute_fwd, compute_bwd)
        elif dataflow=='ws':
            @custom_vjp
            def compute(x, w):
                #print('WS compute func')
                return WS(x, w)
            def compute_fwd(x, w):
                #print('compute_fwd')
                return compute(x, w), (x, w)
            def compute_bwd(res, dy):
                #print('WS compute_bwd')
                x,w = res
                dx = IS(w, dy)
                dw = OS(x,dy)
                return (dx, dw)
            compute.defvjp(compute_fwd, compute_bwd)

        self.forward = partial(compute, w=self.weight)

class ShardedLayerNorm:
    mesh: Mesh
    def __init__(self, mesh, in_dim):
        self.mesh = mesh
        
        self.gamma = jax.device_put(jnp.ones([in_dim], dtype=jnp.bfloat16),
                                    NamedSharding(mesh, P(mesh.axis_names[1])))
        self.bias = jax.device_put(
            jax.random.normal(jax.random.PRNGKey(2),[in_dim], dtype=jnp.bfloat16),
            NamedSharding(mesh, P(mesh.axis_names[1])))
        row, col = mesh.axis_names
        #@jax.jit
        @partial(shard_map,mesh=mesh,in_specs=(P(col), P(col),P(row,col)),
                 out_specs=P(row,col))
        def _layernorm(g,b,x):
            #print(g.shape)
            #print(b.shape)
            ndev = jax.lax.psum(1,mesh.axis_names[1])
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

        self.forward = partial(_layernorm, self.gamma, self.bias)

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

        self.forward = _attn
    
class TransformerBlock:
    def __init__(self, mesh, S,H,D, dataflows, alg='systolic', ksplits=[4,4,4,4]):
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
        out = self.in_proj.forward(out)
        out = self.attn.forward(out)
        out = self.out_proj.forward(out)
        res = self.norm2.forward(out + x)
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
    jax.config.update('jax_platform_name', 'cpu')
    os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=16' # Use 16 CPU devices
    devices = mesh_utils.create_device_mesh((4, 4))
    mesh = Mesh(devices, axis_names=('x', 'y'))
    x = jax.device_put(
        jax.random.normal(jax.random.PRNGKey(3),[8*128,384], dtype=jnp.bfloat16),
        NamedSharding(mesh, P('x','y')))
    model = TransformerBlock(mesh, 128, 12, 32, dataflows=['os','os','os','is'], alg='systolic')
    out = model.forward(x)


    dy = jax.random.normal(jax.random.PRNGKey(3),[8*128,384], dtype=jnp.bfloat16)
    model.backward(dy)