import os
from functools import partial

import jax
import jax.numpy as jnp

from jax.sharding import Mesh, PartitionSpec as P
from jax.sharding import NamedSharding
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.tree_util import tree_map, tree_all
import sys

def createShardedMatrix(mesh, axis_name, global_shape, dtype=jnp.bfloat16, shard_axis=1): #only works for 2d
    local_buffer = jax.random.normal(jax.random.PRNGKey(jax.process_index()),global_shape, dtype=dtype)
    if shard_axis == 1:
        pspec = P(None, axis_name)
    else: #batch sharding
        pspec = P(axis_name, None)
    return jax.device_put(local_buffer, NamedSharding(mesh, pspec))

def allclose(a, b):
  return tree_all(tree_map(partial(jnp.allclose, atol=1e-2, rtol=1e-2), a, b))

def Wang_DPOS(Xi, Wi, axis):
    size = jax.lax.psum(1, axis)
    idx = jax.lax.axis_index(axis)
    shift_up = partial(jax.lax.ppermute, axis_name=axis,
                        perm=[(i, (i + 1) % size) for i in range(size)])
    shift_dn = partial(jax.lax.ppermute, axis_name=axis,
                        perm=[(i, (i - 1) % size) for i in range(size)])

    O_half = Xi.shape[1] // 2  # half-size blocks
    lhs_blocks = lambda i, hi: jax.lax.dynamic_slice_in_dim(Xi, (2*i+hi) * O_half, O_half, 1)

    W_lo, W_hi = jnp.split(Wi, 2, axis=0)
    out_block  = lhs_blocks(idx, 0) @ W_lo
    out_block += lhs_blocks(idx, 1) @ W_hi
    for i in range(1, size):
        W_lo = shift_up(W_lo)
        W_hi = shift_dn(W_hi)
        out_block += lhs_blocks((idx - i) % size, 0) @ W_lo
        out_block += lhs_blocks((idx + i) % size, 1) @ W_hi
    return out_block

def FSDP(Xi, Wi, axis):
    size = jax.lax.psum(1, axis)
    idx = jax.lax.axis_index(axis)
    W = jax.lax.all_gather(Wi, axis, tiled=True, axis=1)
    return Xi@W

def Wang_1DOS(Xi, Wi, axis):
    size = jax.lax.psum(1, axis)
    idx = jax.lax.axis_index(axis)
    shift_up = partial(jax.lax.ppermute, axis_name=axis,
                        perm=[(i, (i + 1) % size) for i in range(size)])
    shift_dn = partial(jax.lax.ppermute, axis_name=axis,
                        perm=[(i, (i - 1) % size) for i in range(size)])

    I_half = Xi.shape[1] // 2  # half-size blocks
    rhs_blocks = lambda i, hi: jax.lax.dynamic_slice_in_dim(Wi, (2*i+hi) * I_half, I_half, 0)

    X_lo, X_hi = jnp.split(Xi, 2, axis=1)
    out_block  = X_lo @ rhs_blocks(idx, 0)
    out_block += X_hi @ rhs_blocks(idx, 1)
    for i in range(1, size):
        X_lo = shift_up(X_lo)
        X_hi = shift_dn(X_hi)
        out_block += X_lo @ rhs_blocks((idx - i)%size, 0)
        out_block += X_hi @ rhs_blocks((idx + i)%size, 1)
    return out_block

#Run XW^T (bi,oi->bo)
def Wang_1DIS(Xi, Wi, axis):
  size = jax.lax.psum(1, axis)
  idx = jax.lax.axis_index(axis)
  shift_up = partial(jax.lax.ppermute, axis_name=axis,
                     perm=[(i, (i + 1) % size) for i in range(size)])
  shift_dn = partial(jax.lax.ppermute, axis_name=axis,
                     perm=[(i, (i - 1) % size) for i in range(size)])

  O_half = Wi.shape[0] // size // 2  # half-size blocks
  rhs_blocks = lambda i, hi: jax.lax.dynamic_slice_in_dim(Wi, (2*i+hi) * O_half, O_half, 0)

  out_lo = jnp.einsum('bi,oi->bo',
                      Xi, rhs_blocks((idx-1)%size, 0))
  out_hi = jnp.einsum('bi,oi->bo',
                      Xi, rhs_blocks((idx+1)%size, 1))
  for i in range(1, size):
    out_lo = shift_up(out_lo)
    out_hi = shift_dn(out_hi)
    out_lo += jnp.einsum('bi,oi->bo',
                      Xi, rhs_blocks((idx-i-1)%size, 0))
    out_hi += jnp.einsum('bi,oi->bo',
                      Xi, rhs_blocks((idx+i+1)%size, 1))
  return jnp.concatenate([out_lo, out_hi],axis=1)







class SPMDWang:
    def __init__(self, mesh, jit=True):
        self.mesh = mesh
        self.axis = mesh.axis_names[0]

        self.spec = P(None,self.axis)
        
        osfunc = partial(shard_map, mesh=self.mesh, in_specs=(self.spec,self.spec),
         out_specs=self.spec)(partial(
             Wang_1DOS, axis=self.axis))
        dpfunc = partial(shard_map, mesh=self.mesh, in_specs=(P(self.axis,None),self.spec),
         out_specs=P(self.axis,None))(partial(
             FSDP, axis=self.axis))
        isfunc = partial(shard_map, mesh=self.mesh, in_specs=(self.spec,self.spec),
         out_specs=self.spec)(partial(
             Wang_1DIS, axis=self.axis))
        if jit:
            self.OS = jax.jit(osfunc)
            self.DP = jax.jit(dpfunc)
            self.IS = jax.jit(isfunc)
        else:
            self.OS = osfunc
            self.DP = dpfunc
            self.IS = isfunc

if __name__=='__main__':
    
    B,S,H,D = (4,128, 48,64)

    jax.config.update('jax_platform_name', 'cpu')
    os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=4' # Use 16 CPU devices
    
    axis_name='i'
    mesh = Mesh(jax.devices(), (axis_name,))
    X = jnp.arange( B*S*H*D,dtype=jnp.float32).reshape(B*S, H*D)/(B*S*H*D)
    W = jnp.arange(H*D*4*H*D,dtype=jnp.float32).reshape(H*D, 4*H*D) / (4*H*D*H*D)
    Y = X@W
    dX = Y @ W.transpose()
    TP = SPMDWang(mesh)
    
    Xtp = jax.device_put(X, NamedSharding(mesh,P(None,axis_name)))
    Wtp = jax.device_put(W, NamedSharding(mesh,P(None,axis_name)))
    
    Ytp = TP.OS(Xtp, Wtp)
    print(allclose(Ytp,X @ W))

    dXtp = TP.IS(Ytp, Wtp)
    print(allclose(dXtp, dX))

    dy, backward_fn = jax.vjp(TP.OS, Xtp,Wtp)
    print(dy.shape)
    dx,dw = backward_fn(dy)
    print(allclose(TP.IS(dy, Wtp), dx))

    Xdp = jax.device_put(X, NamedSharding(mesh,P(axis_name,None)))
    Ydp = TP.DP(Xdp,Wtp)
    print(allclose(Ydp,X @ W))
    