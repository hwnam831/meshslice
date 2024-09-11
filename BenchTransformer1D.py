import os
from functools import partial

from TensorParallel1D import SPMDWang, createShardedMatrix

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
import argparse



class ShardedFFLayer1D:
    mesh: Mesh
    algorithm: str
    dataflow:str
    def __init__(self, mesh, dataflow, in_dim, out_dim):
        self.mesh = mesh
        axis = mesh.axis_names[0]
        self.dataflow = dataflow
        SPMDbuilder = SPMDWang(mesh)
        if dataflow == 'is':
            self.weight = createShardedMatrix(mesh, axis, [out_dim, in_dim])
            self.forward = SPMDbuilder.IS
        elif dataflow == 'os':
            self.weight = createShardedMatrix(mesh, axis, [in_dim, out_dim])
            self.forward = SPMDbuilder.OS
        else: #FSDP
            self.weight = createShardedMatrix(mesh, axis, [in_dim, out_dim],shard_axis=0)
            self.forward = SPMDbuilder.DP
        

        

class ShardedLayerNorm1D:
    mesh: Mesh
    def __init__(self, mesh, in_dim, sharding='tp'):
        self.mesh = mesh
        axis = mesh.axis_names[0]
        if sharding=='tp':
            pspec = P(None,axis)
        else:
            pspec = P(axis,None)
        #only local layernorm
        @partial(shard_map,mesh=mesh,in_specs=(pspec),
                 out_specs=pspec)
        def tp_layernorm(x):
            ndev = jax.lax.psum(1,mesh.axis_names[0])
            mn_l = jnp.mean(x,axis=-1)
            sqmn_l = jnp.mean(x*x, axis=-1)
            mn = jax.lax.psum(mn_l, mesh.axis_names[0])/ndev
            #print(mn.shape)
            sqmn = jax.lax.psum(sqmn_l, mesh.axis_names[0])/ndev
            #print(sqmn.shape)
            stdev = 1/jnp.sqrt(sqmn - mn*mn + 1e-6)
            out =  (x-mn[:,None])*stdev[:,None]
            return out
        @partial(shard_map,mesh=mesh,in_specs=(pspec),
                 out_specs=pspec)
        def bp_layernorm(x):
            mn = jnp.mean(x,axis=-1)
            sqmn = jnp.mean(x*x, axis=-1)
            #print(mn.shape)
            #print(sqmn.shape)
            stdev = 1/jnp.sqrt(sqmn - mn*mn + 1e-6)
            out =  (x-mn[:,None])*stdev[:,None]
            return out
        
        if sharding == 'tp':
            self.forward = jax.jit(tp_layernorm)
        else:
            self.forward = jax.jit(bp_layernorm)
        

class ShardedAttention1D:
    mesh: Mesh
    def __init__(self, mesh, n_heads, seqlen, sharding='tp'):
        self.mesh = mesh
        self.seqlen = seqlen
        axis = mesh.axis_names[0]
        if sharding=='tp':
            self.heads_per_shard = n_heads // len(mesh.local_devices)
            @partial(shard_map,mesh=mesh,in_specs=P(None,axis),
                    out_specs=P(None,axis))
            def _attn(Xij):
                #print(Xij.shape)
                Q,K,V = jnp.split(Xij.reshape(Xij.shape[0]//self.seqlen, 
                                self.seqlen,self.heads_per_shard,-1), 3, axis=-1)
                #print(V.shape)
                #q: num_queries=S, k: num_keys=num_vals=S
                attn_weights = jax.nn.softmax(jnp.einsum('bqhd,bkhd->bhqk',Q,K)/jnp.sqrt(Q.shape[-1]),axis=3)
                return jnp.einsum('bhqk,bkhd->bqhd',attn_weights,V).reshape(Xij.shape[0],-1)

            self.forward = jax.jit(_attn)
        else:
            self.heads_per_shard = n_heads
            @partial(shard_map,mesh=mesh,in_specs=P(axis,None),
                    out_specs=P(axis,None))
            def _attn(Xij):
                #print(Xij.shape)
                Q,K,V = jnp.split(Xij.reshape(Xij.shape[0]//self.seqlen, 
                                self.seqlen,self.heads_per_shard,-1), 3, axis=-1)
                #print(V.shape)
                #q: num_queries=S, k: num_keys=num_vals=S
                attn_weights = jax.nn.softmax(jnp.einsum('bqhd,bkhd->bhqk',Q,K)/jnp.sqrt(Q.shape[-1]),axis=3)
                return jnp.einsum('bhqk,bkhd->bqhd',attn_weights,V).reshape(Xij.shape[0],-1)

            self.forward = jax.jit(_attn)

if __name__ == '__main__':
    jax.config.update('jax_platform_name', 'cpu')
    os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=4'
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=-1, help='Default bathsize number of chips')
    parser.add_argument('--seqlen', type=int, default=2048, help='Number of tokens per input sequence')
    parser.add_argument('--nheads', type=int, default=160, help='Number of attention head (gpt3:96, megatron:160)')
    parser.add_argument('--headdim', type=int, default=128, help='Hidden dimension per attention head')
    parser.add_argument('--alg', type=str, default='tp', choices=['tp', 'fsdp'])
    parser.add_argument('--noff', action='store_true')
    parser.add_argument('--ndev', type=int, default=4, help='Number of devices to extrapolate')
    args = parser.parse_args()

    NDEV=args.ndev
    B = NDEV if args.batchsize <= 0 else args.batchsize
    S = args.seqlen
    H = args.nheads
    D = args.headdim
    assert (args.alg=='fsdp') or (args.nheads%NDEV == 0)
    alg = args.alg
    
    axis_name='i'
    mesh = Mesh(jax.devices(), (axis_name,))
    devicecount = len(jax.devices())
    benchtag = "{}{}_{}_{}_{}_{}".format(alg, '_noff' if args.noff else '' ,NDEV, B, H, D)
    if args.noff:
        if alg == 'tp':
            B2 = B
            H2 = H * devicecount // NDEV
            input_0 = createShardedMatrix(mesh, axis_name, [B2*S,3*H2*D])
            norm1 = ShardedLayerNorm1D(mesh, H2*D)       
            attn = ShardedAttention1D(mesh, H2, S)
            #Three norms, two adds, one gelu
            def forward_fn(x):
                res = attn.forward(x)
                out = norm1.forward(res)
                res2 = norm1.forward(res+out)
                out = jax.nn.gelu(res2)
                out = norm1.forward(res2+out)
                return out
            dy, backward_fn = jax.vjp(forward_fn, input_0)
            backward_fn(dy)[-1].block_until_ready()
            forward_fn(input_0).block_until_ready()
            jax.profiler.start_trace("/tmp/tensorboard/"+benchtag)
            with jax.profiler.TraceAnnotation("forward"):
                forward_fn(input_0).block_until_ready()
            with jax.profiler.TraceAnnotation("backward"):
                backward_fn(dy)[-1].block_until_ready()
            jax.profiler.stop_trace()
        else:
            B2 = B * devicecount // NDEV
            H2 = H 
            input_0 = createShardedMatrix(mesh, axis_name, [B2*S,3*H2*D],shard_axis=0)
            norm1 = ShardedLayerNorm1D(mesh, H2*D, sharding='dp')       
            attn = ShardedAttention1D(mesh, H2, S, sharding='dp')
            #Three norms, two adds, one gelu
            def forward_fn(x):
                res = attn.forward(x)
                out = norm1.forward(res)
                res2 = norm1.forward(res+out)
                out = jax.nn.gelu(res2)
                out = norm1.forward(res2+out)
                return out
            dy, backward_fn = jax.vjp(forward_fn, input_0)
            backward_fn(dy)[-1].block_until_ready()
            forward_fn(input_0).block_until_ready()
            jax.profiler.start_trace("/tmp/tensorboard/"+benchtag)
            with jax.profiler.TraceAnnotation("forward"):
                forward_fn(input_0).block_until_ready()
            with jax.profiler.TraceAnnotation("backward"):
                backward_fn(dy)[-1].block_until_ready()
            jax.profiler.stop_trace()
    elif alg == 'tp':
        B2 = B
        H2 = H * devicecount // NDEV
        weights=[]
        out_proj = ShardedFFLayer1D(mesh, 'is',H2*D, H*D)
        weights.append(out_proj.weight)
        inp_op = createShardedMatrix(mesh, axis_name, [B2*S,H2*D])
        
        ff1 = ShardedFFLayer1D(mesh, 'os', H*D, 4*H2*D)
        weights.append(ff1.weight)
    
        ff2 = ShardedFFLayer1D(mesh, 'is', 4*H2*D, H*D)
        weights.append(ff2.weight)

        in_proj = ShardedFFLayer1D(mesh, 'os', H*D, 3*H2*D)
        weights.append(in_proj.weight)

        def forward_fn(x, *w):
            out = out_proj.forward(x, w[0])
            out = ff1.forward(out, w[1])
            out = ff2.forward(out, w[2])
            out = in_proj.forward(out, w[3])
            return out
        dy, backward_fn = jax.vjp(forward_fn, inp_op, *weights)
        backward_fn(dy)[-1].block_until_ready()
        forward_fn(inp_op, *weights).block_until_ready()
        jax.profiler.start_trace("/tmp/tensorboard/"+benchtag)
        with jax.profiler.TraceAnnotation("forward"):
            forward_fn(inp_op, *weights).block_until_ready()
        with jax.profiler.TraceAnnotation("backward"):
            backward_fn(dy)[-1].block_until_ready()
        jax.profiler.stop_trace()
    else:
        B2 = B * devicecount // NDEV
        H2 = H
        weights=[]
        out_proj = ShardedFFLayer1D(mesh, 'fsdp',H2*D, H*D)
        weights.append(out_proj.weight)
        inp_op = createShardedMatrix(mesh, axis_name, [B2*S,H2*D])
        
        ff1 = ShardedFFLayer1D(mesh, 'fsdp', H*D, 4*H2*D)
        weights.append(ff1.weight)
    
        ff2 = ShardedFFLayer1D(mesh, 'fsdp', 4*H2*D, H*D)
        weights.append(ff2.weight)

        in_proj = ShardedFFLayer1D(mesh, 'fsdp', H*D, 3*H2*D)
        weights.append(in_proj.weight)

        def forward_fn(x, *w):
            out = out_proj.forward(x, w[0])
            out = ff1.forward(out, w[1])
            out = ff2.forward(out, w[2])
            out = in_proj.forward(out, w[3])
            return out
        dy, backward_fn = jax.vjp(forward_fn, inp_op, *weights)
        backward_fn(dy)[-1].block_until_ready()
        forward_fn(inp_op, *weights).block_until_ready()
        
        jax.profiler.start_trace("/tmp/tensorboard/"+benchtag)
        with jax.profiler.TraceAnnotation("forward"):
            forward_fn(inp_op, *weights).block_until_ready()
        with jax.profiler.TraceAnnotation("backward"):
            backward_fn(dy)[-1].block_until_ready()
        jax.profiler.stop_trace()
