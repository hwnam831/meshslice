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
import argparse
from ShardedLayers import ShardedAttention, ShardedFFLayer, ShardedLayerNorm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ksplit', nargs=4, type=int, default=[8,8,8,8], help='Loop iteration count per FC layer')
    parser.add_argument('--batchsize', type=int, default=-1, help='Default bathsize number of chips')
    parser.add_argument('--seqlen', type=int, default=2048, help='Number of tokens per input sequence')
    parser.add_argument('--nheads', type=int, default=96, help='Number of attention head (gpt3:96, megatron:160)')
    parser.add_argument('--headdim', type=int, default=128, help='Hidden dimension per attention head')
    parser.add_argument('--nrows', type=int, default=4, help='Number of rows in device mesh. Must be multiple of 4')
    parser.add_argument('--ncols', type=int, default=4, help='Number of cols in device mesh. Must be multiple of 4')
    parser.add_argument('--alg', type=str, default='noff', choices=['noff','collective', 'cannon', 'wang', 'meshflow'])
    args = parser.parse_args()

    B = args.nrows*args.ncols if args.batchsize <= 0 else args.batchsize
    S = args.seqlen
    H = args.nheads
    D = args.headdim
    alg = args.alg
    NROW = args.nrows
    NCOL = args.ncols

    jax.distributed.initialize()
    print("Global device count: {}".format(jax.device_count()))
    print("Local device count: {}".format(jax.local_device_count()))
    
    colcount = jax.local_device_count()
    rowcount = jax.device_count() // colcount
    assert NROW % rowcount == 0
    assert NCOL % colcount == 0
    #print(jax.devices())
    #print(jax.local_devices())
    devices = mesh_utils.create_device_mesh((rowcount,colcount))
    mesh = Mesh(devices, axis_names=('x','y'))
    benchtag = "{}_{}x{}_{}_{}_{}".format(alg, NROW, NCOL, B, H, D)
    if alg == 'noff':
        B2 = B * rowcount // NROW
        H2 = H * colcount // NCOL
        input_0 = createMultihostMatrix(mesh, NamedSharding(mesh, P('x','y')), [B2*S,3*H2*D])
        norm1 = ShardedLayerNorm(mesh, H2*D)       
        attn = ShardedAttention(mesh, H2, S)
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
        B2 = B * rowcount // NROW
        H2 = H * colcount // NCOL
        out_proj = ShardedFFLayer(mesh, alg, 'ls', args.ksplit[1],
                                 H2*D, H*D)
        inp_op = createMultihostMatrix(mesh, NamedSharding(mesh, P('x','y')), [B2*S,H2*D])
        
        ff1 = ShardedFFLayer(mesh, alg, 'os', args.ksplit[2],
                                 H*D, 4*H2*D)
        inp_ff1 = createMultihostMatrix(mesh, NamedSharding(mesh, P('x','y')), [B2*S,H*D])

        ff2 = ShardedFFLayer(mesh, alg, 'ls', args.ksplit[3],
                             4*H2*D, H*D)
        inp_ff2 = createMultihostMatrix(mesh, NamedSharding(mesh, P('x','y')), [B2*S,4*H2*D])

        in_proj = ShardedFFLayer(mesh, alg, 'os', args.ksplit[0],
                             H*D, 3*H2*D)
        inp_ip = createMultihostMatrix(mesh, NamedSharding(mesh, P('x','y')), [B2*S,H*D])

        def forward_fn(x):
            out = out_proj.forward(x)
            out = ff1.forward(out)
            out = ff2.forward(out)
            out = in_proj.forward(out)
            return out
        dy, backward_fn = jax.vjp(forward_fn, inp_op)
        backward_fn(dy)[-1].block_until_ready()
        forward_fn(inp_op).block_until_ready()
        jax.profiler.start_trace("/tmp/tensorboard/"+benchtag)
        with jax.profiler.TraceAnnotation("forward"):
            forward_fn(inp_op).block_until_ready()
        with jax.profiler.TraceAnnotation("backward"):
            backward_fn(dy)[-1].block_until_ready()
        jax.profiler.stop_trace()
    
