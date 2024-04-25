import os
from functools import partial

import jax
import jax.numpy as jnp

from jax.sharding import Mesh, PartitionSpec as P
from jax.sharding import NamedSharding
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.tree_util import tree_map, tree_all
import timeit



def allclose(a, b):
  return tree_all(tree_map(partial(jnp.allclose, atol=1e-2, rtol=1e-2), a, b))

def Collective_OS(Xij, Wij, row, col, K, B):
    Xi = jax.lax.all_gather(Xij, col, tiled=True, axis=1)
    Wj = jax.lax.all_gather(Wij, row, tiled=True, axis=0)
    return Xi @ Wj

def Collective_IS(Xij, Wji, row, col, K, B):
    Wj = jax.lax.all_gather(Wji, row, tiled=True, axis=0)
    Yp = jnp.einsum('mn,kn->mk',Xij,Wj)
    return jax.lax.psum_scatter(Yp, col, scatter_dimension=1, tiled=True)

def Collective_WS(Xji, Wij, row, col, K, B):
    Xi = jax.lax.all_gather(Xji, col, tiled=True, axis=1)
    Yp = jnp.einsum('ib,io->bo',Xi,Wij)
    return jax.lax.psum_scatter(Yp, row, scatter_dimension=0, tiled=True)

def Wang_OS(Xij, Wij, row, col, K, B):
    Wj = jax.lax.all_gather(Wij, row, tiled=True, axis=0)
    size = jax.lax.psum(1, col)
    idx = jax.lax.axis_index(col)
    shift_up = partial(jax.lax.ppermute, axis_name=col,
                        perm=[(i, (i + 1) % size) for i in range(size)])
    shift_dn = partial(jax.lax.ppermute, axis_name=col,
                        perm=[(i, (i - 1) % size) for i in range(size)])

    B = Wj.shape[0] // size // 2  # half-size blocks
    w_blocks = lambda i, hi: jax.lax.dynamic_slice_in_dim(Wj, (2*i+hi) * B, B, 0)

    x_lo, x_hi = jnp.split(Xij, 2, axis=1)

    out_block  =  x_lo @ w_blocks(idx, 0)
    out_block +=  x_hi @ w_blocks(idx, 1)

    def body_fn(ii,carry, myidx, mysize):
        low,high,oblock = carry
        low = shift_up(low)
        high = shift_dn(high)
        oblock +=  low @ w_blocks((myidx - ii) % mysize, 0)
        oblock +=  high @ w_blocks((myidx + ii) % mysize, 1)
        return (low,high,oblock)
    mybody = partial(body_fn, myidx=idx, mysize=size)
    return jax.lax.fori_loop(1,size,mybody,init_val=(x_lo,x_hi,out_block))[2]


def Wang_IS(Xij, Wji, row, col, K, B):
    Wj = jax.lax.all_gather(Wji, row, tiled=True, axis=0)
    size = jax.lax.psum(1, col)
    idx = jax.lax.axis_index(col)
    shift_up = partial(jax.lax.ppermute, axis_name=col,
                        perm=[(i, (i + 1) % size) for i in range(size)])
    shift_dn = partial(jax.lax.ppermute, axis_name=col,
                        perm=[(i, (i - 1) % size) for i in range(size)])

    B = Wj.shape[0] // size // 2  # half-size blocks
    w_blocks = lambda i, hi: jax.lax.dynamic_slice_in_dim(Wj, (2*i+hi) * B, B, 0)

    y_lo  = jnp.einsum('bi,oi->bo',Xij,w_blocks((idx-1)%size, 0))
    y_hi  = jnp.einsum('bi,oi->bo',Xij, w_blocks((idx+1)%size, 1))
    def body_fn(ii,carry, myidx, mysize):
        low,high = carry
        low = shift_up(low)
        high = shift_dn(high)
        low += jnp.einsum('bi,oi->bo',Xij, w_blocks((myidx - ii - 1) % mysize, 0))
        
        high += jnp.einsum('bi,oi->bo',Xij, w_blocks((myidx + ii + 1) % mysize, 1))
        return (low,high)
    mybody = partial(body_fn, myidx=idx, mysize=size)
    
    y_lo,y_hi =  jax.lax.fori_loop(1,size,mybody,init_val=(y_lo,y_hi))
    return jnp.concatenate([y_lo,y_hi],axis=1)

def Wang_WS(Xji, Wij, row, col, K, B):
    R = jax.lax.psum(1,row)
    C = jax.lax.psum(1,col)
    #input traffic is bigger, so osplit
    if C * Xji.shape[0] > R*Wij.shape[1]:
        Xi = jax.lax.all_gather(Xji, col, tiled=True, axis=1)
        size = R
        idx = jax.lax.axis_index(row)
        shift_up = partial(jax.lax.ppermute, axis_name=row,
                            perm=[(i, (i + 1) % size) for i in range(size)])
        shift_dn = partial(jax.lax.ppermute, axis_name=row,
                            perm=[(i, (i - 1) % size) for i in range(size)])

        B = Xi.shape[1] // size // 2  # half-size blocks
        x_blocks = lambda i, hi: jax.lax.dynamic_slice_in_dim(Xi, (2*i+hi) * B, B, 1)

        y_lo  =  jnp.einsum('ib,io->bo',x_blocks((idx-1)%size, 0), Wij)
        y_hi  =  jnp.einsum('ib,io->bo',x_blocks((idx+1)%size, 1), Wij)
        def body_fn(ii,carry, myidx, mysize):
            low,high = carry
            low = shift_up(low)
            high = shift_dn(high)
            low +=  jnp.einsum('ib,io->bo',x_blocks((myidx - ii - 1) % mysize, 0), Wij)
            high +=  jnp.einsum('ib,io->bo',x_blocks((myidx + ii + 1) % mysize, 1), Wij)
            return (low,high)
        mybody = partial(body_fn, myidx=idx, mysize=size)
        
        
        y_lo,y_hi =  jax.lax.fori_loop(1,size,mybody,init_val=(y_lo,y_hi))
        return jnp.concatenate([y_lo,y_hi],axis=0)
    #output traffic is bigger, so isplit
    else:
        size = C
        idx = jax.lax.axis_index(col)
        shift_up = partial(jax.lax.ppermute, axis_name=col,
                            perm=[(i, (i + 1) % size) for i in range(size)])
        shift_dn = partial(jax.lax.ppermute, axis_name=col,
                            perm=[(i, (i - 1) % size) for i in range(size)])
        Yj = jnp.zeros((Xji.shape[1]*C, Wij.shape[1]), Xji.dtype)
        B = Yj.shape[0] // size //2 # half-size blocks
        
        Yk = jnp.einsum('ib,io->bo',Xji, Wij)
        Yj = jax.lax.dynamic_update_slice_in_dim(Yj, Yk, 2*((idx)%size)*B, axis=0)
        x_lo, x_hi = jnp.split(Xji, 2, 1)
        def body_fn(ii,carry, myidx, mysize):
            lo,hi, Yj = carry
            lo = shift_up(lo)
            hi = shift_dn(hi)
            Y_lo = jnp.einsum('ib,io->bo',lo, Wij)
            Yj = jax.lax.dynamic_update_slice_in_dim(Yj, Y_lo, 2*((idx-ii)%size)*B, axis=0)
            Y_hi = jnp.einsum('ib,io->bo',hi, Wij)
            Yj = jax.lax.dynamic_update_slice_in_dim(Yj, Y_hi, 2*((idx+ii)%size)*B+B, axis=0)
            return (lo,hi,Yj)
        mybody = partial(body_fn, myidx=idx, mysize=size)
        Yj = jax.lax.fori_loop(1,size,mybody,init_val=(x_lo,x_hi,Yj))[2]
        return jax.lax.psum_scatter(Yj, row, scatter_dimension=0, tiled=True)

def Systolic_OS(Xij, Wij, row, col, K, B): #smallest block size B
    
    X_split = Xij.reshape(Xij.shape[0], Xij.shape[1]//(K*B),K,B)
    W_split = Wij.reshape(Wij.shape[0]//(K*B),K,B, Wij.shape[1])
    x_blocks = lambda i: jax.lax.dynamic_slice_in_dim(X_split, i, 1, 2)
    w_blocks = lambda i: jax.lax.dynamic_slice_in_dim(W_split, i, 1, 1)
    Xk = jax.lax.all_gather(x_blocks(0), col, tiled=True, axis=1)
    Wk = jax.lax.all_gather(w_blocks(0), row, tiled=True, axis=0)
    
    Yij = Xk.reshape(Xij.shape[0],-1) @ Wk.reshape(-1,Wij.shape[1])
    for k in range(1,K):
        Xk = jax.lax.all_gather(x_blocks(k), col, tiled=True, axis=1)
        Wk = jax.lax.all_gather(w_blocks(k), row, tiled=True, axis=0)
        Yij += Xk.reshape(Xij.shape[0],-1) @ Wk.reshape(-1,Wij.shape[1])
    return Yij

def Systolic_IS(Xij, Wji, row, col, K, B): #smallest block size B
    O = Wji.shape[0]*jax.lax.psum(1, row)
    C = jax.lax.psum(1,col)
    W_split = Wji.reshape(Wji.shape[0]//K//B,K,B, Wji.shape[1])
    w_blocks = lambda i: jax.lax.dynamic_slice_in_dim(W_split, i, 1, 1)
    Yij = jnp.zeros((Xij.shape[0],O//(K*B*C), K,B),Xij.dtype)
    Wk = jax.lax.all_gather(w_blocks(0), row, tiled=True, axis=0)
    Yk = jnp.einsum('ni,okbi->nokb',Xij,Wk) #(B/R, O/K/B, 1, B)
    Yk2 = jax.lax.psum_scatter(Yk, col, scatter_dimension=1, tiled=True)#(B/R, O/K/B/C, 1, B)
  
    Yij = jax.lax.dynamic_update_index_in_dim(Yij,Yk2,0,axis=2)
    def body_fn(kk,carry):
        Wk = jax.lax.all_gather(w_blocks(kk), row, tiled=True, axis=0)
        Yk = jnp.einsum('ni,okbi->nokb',Xij,Wk) #(B/R, O/K/B, 1, B)
        Yk2 = jax.lax.psum_scatter(Yk, col, scatter_dimension=1, tiled=True)
        #Yijs.append(Yk2)
        return jax.lax.dynamic_update_index_in_dim(carry,Yk2,kk,axis=2)
        
    Yij = jax.lax.fori_loop(1,K,body_fn,init_val=Yij)
    return Yij.reshape(Xij.shape[0],-1)

def Systolic_WS(Xji, Wij, row, col, K, B): #smallest block size B
    N = Xji.shape[1]*jax.lax.psum(1, col)
    R = jax.lax.psum(1,row)
    X_split = Xji.reshape(Xji.shape[0], Xji.shape[1]//K//B,K,B)
    x_blocks = lambda i: jax.lax.dynamic_slice_in_dim(X_split, i, 1, 2)
    Yij = jnp.zeros((N//(K*B*R), K,B, Wij.shape[1]),Xji.dtype)
    
    Xk = jax.lax.all_gather(x_blocks(0), col, tiled=True, axis=1)
    Yk = jnp.einsum('inkb,io->nkbo',Xk,Wij) 
    Yk2 = jax.lax.psum_scatter(Yk, row, scatter_dimension=0, tiled=True)
    
    Yij = jax.lax.dynamic_update_index_in_dim(Yij,Yk2,0,axis=1)
    def body_fn(kk,carry):
        Xk = jax.lax.all_gather(x_blocks(kk), col, tiled=True, axis=1)
        Yk = jnp.einsum('inkb,io->nkbo',Xk,Wij) 
        Yk2 = jax.lax.psum_scatter(Yk, row, scatter_dimension=0, tiled=True)
        return jax.lax.dynamic_update_index_in_dim(carry,Yk2,kk,axis=1)
        
    Yij = jax.lax.fori_loop(1,K,body_fn,init_val=Yij)
    return Yij.reshape(-1,Wij.shape[1])


class SPMD:
    def __init__(self, mesh, algorithm='collective', blocksize=8):
        self.mesh = mesh
        self.rowaxis = mesh.axis_names[0]
        self.colaxis = mesh.axis_names[1]
        self.spec = P(self.rowaxis,self.colaxis)
        self.blocksize=blocksize
        if algorithm=='collective':
            self.funcs = {'os':Collective_OS, 'is':Collective_IS, 'ws':Collective_WS}
        elif algorithm=='wang':
            self.funcs = {'os':Wang_OS, 'is':Wang_IS, 'ws':Wang_WS}
        else:
            self.funcs = {'os':Systolic_OS, 'is':Systolic_IS, 'ws':Systolic_WS}
    def OS(self, K=8):
        return jax.jit(partial(shard_map, mesh=self.mesh, in_specs=(self.spec,self.spec),
         out_specs=self.spec)(partial(
             self.funcs['os'], row=self.rowaxis, col=self.colaxis, K=K,B=self.blocksize)))
    def IS(self, K=8):
        return jax.jit(partial(shard_map, mesh=self.mesh, in_specs=(self.spec,self.spec),
         out_specs=self.spec)(partial(
             self.funcs['is'], row=self.rowaxis, col=self.colaxis, K=K,B=self.blocksize)))
    def WS(self, K=8):
        return jax.jit(partial(shard_map, mesh=self.mesh, in_specs=(self.spec,self.spec),
         out_specs=self.spec)(partial(
             self.funcs['ws'], row=self.rowaxis, col=self.colaxis, K=K,B=self.blocksize)))


if __name__=='__main__':
    os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8' # Use 8 CPU devices
    devices = mesh_utils.create_device_mesh((4, 2))
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
