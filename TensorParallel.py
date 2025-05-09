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



def allclose(a, b):
  return tree_all(tree_map(partial(jnp.allclose, atol=1e-2, rtol=1e-2), a, b))

def Collective_OS(Xij, Wij, row, col, K, B):
    Xi = jax.lax.all_gather(Xij, col, tiled=True, axis=1)
    Wj = jax.lax.all_gather(Wij, row, tiled=True, axis=0)
    return Xi @ Wj

def Collective_LS(Xij, Wji, row, col, K, B):
    Wj = jax.lax.all_gather(Wji, row, tiled=True, axis=0)
    Yp = jnp.einsum('mn,kn->mk',Xij,Wj)
    return jax.lax.psum_scatter(Yp, col, scatter_dimension=1, tiled=True)

def Collective_RS(Xji, Wij, row, col, K, B):
    Xi = jax.lax.all_gather(Xji, col, tiled=True, axis=1)
    Yp = jnp.einsum('ib,io->bo',Xi,Wij)
    return jax.lax.psum_scatter(Yp, row, scatter_dimension=0, tiled=True)

def Wang_OS(Xij, Wij, row, col, K, B):
    Wj = jax.lax.all_gather(Wij, row, tiled=True, axis=0)
    size = jax.lax.psum(1, col)
    idx = jax.lax.ax_index(col)
    shift_up = partial(jax.lax.ppermute, axis_name=col,
                        perm=[(i, (i + 1) % size) for i in range(size)])
    shift_dn = partial(jax.lax.ppermute, axis_name=col,
                        perm=[(i, (i - 1) % size) for i in range(size)])

    B = Wj.shape[0] // size // 2  # half-size blocks
    w_blocks = lambda i, hi: jax.lax.dynamic_slice_in_dim(Wj, (2*i+hi) * B, B, 0)

    x_lo, x_hi = jnp.split(Xij, 2, axis=1)

    out_block  =  x_lo @ w_blocks(idx, 0)
    out_block +=  x_hi @ w_blocks(idx, 1)
    x_lo = shift_up(x_lo)
    x_hi = shift_dn(x_hi)
    def body_fn(ii,carry, myidx, mysize):
        low,high,oblock = carry
        low2 = shift_up(low)
        high2 = shift_dn(high)
        oblock +=  low @ w_blocks((myidx - ii) % mysize, 0)
        oblock +=  high @ w_blocks((myidx + ii) % mysize, 1)
        return (low2,high2,oblock)
    mybody = partial(body_fn, myidx=idx, mysize=size)
    return jax.lax.fori_loop(1,size,mybody,init_val=(x_lo,x_hi,out_block))[2]


def Wang_LS(Xij, Wji, row, col, K, B):
    Wj = jax.lax.all_gather(Wji, row, tiled=True, axis=0)
    size = jax.lax.psum(1, col)
    idx = jax.lax.ax_index(col)
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
        lo = jnp.einsum('bi,oi->bo',Xij, w_blocks((myidx - ii-1) % mysize, 0))
        
        hi = jnp.einsum('bi,oi->bo',Xij, w_blocks((myidx + ii +1) % mysize, 1))
        return (low+lo, high+hi)
    mybody = partial(body_fn, myidx=idx, mysize=size)
    
    y_lo,y_hi =  jax.lax.fori_loop(1,size,mybody,init_val=(y_lo,y_hi))
    return jnp.concatenate([y_lo,y_hi],axis=1)

def Wang_RS(Xji, Wij, row, col, K, B):
    R = jax.lax.psum(1,row)
    C = jax.lax.psum(1,col)
    #input traffic  bigger, so osplit
    if False:
        Xi = jax.lax.all_gather(Xji, col, tiled=True, axis=1)
        size = R
        idx = jax.lax.ax_index(row)
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
            lo =  jnp.einsum('ib,io->bo',x_blocks((myidx - ii - 1) % mysize, 0), Wij)
            hi =  jnp.einsum('ib,io->bo',x_blocks((myidx + ii + 1) % mysize, 1), Wij)
            return (low+lo,high+hi)
        mybody = partial(body_fn, myidx=idx, mysize=size)
        
        
        y_lo,y_hi =  jax.lax.fori_loop(1,size,mybody,init_val=(y_lo,y_hi))
        return jnp.concatenate([y_lo,y_hi],axis=0)
    #output traffic  bigger, so plit
    else:
        size = C
        idx = jax.lax.ax_index(col)
        shift_up = partial(jax.lax.ppermute, axis_name=col,
                            perm=[(i, (i + 1) % size) for i in range(size)])
        shift_dn = partial(jax.lax.ppermute, axis_name=col,
                            perm=[(i, (i - 1) % size) for i in range(size)])
        Yj = jnp.zeros((Xji.shape[1]*C, Wij.shape[1]), Xji.dtype)
        B = Yj.shape[0] // size //2 # half-size blocks
        
        Yk = jnp.einsum('ib,io->bo',Xji, Wij)
        Yj = jax.lax.dynamic_update_slice_in_dim(Yj, Yk, 2*((idx)%size)*B, axis=0)
        x_lo, x_hi = jnp.split(Xji, 2, 1)
        x_lo = shift_up(x_lo)
        x_hi = shift_dn(x_hi)
        def body_fn(ii,carry, myidx, mysize):
            lo,hi, Yj = carry
            Y_lo = jnp.einsum('ib,io->bo',lo, Wij)
            Yj = jax.lax.dynamic_update_slice_in_dim(Yj, Y_lo, 2*((idx-ii)%size)*B, axis=0)
            Y_hi = jnp.einsum('ib,io->bo',hi, Wij)
            Yj = jax.lax.dynamic_update_slice_in_dim(Yj, Y_hi, 2*((idx+ii)%size)*B+B, axis=0)
            lo2 = shift_up(lo)
            hi2 = shift_dn(hi)
            return (lo2,hi2,Yj)
        mybody = partial(body_fn, myidx=idx, mysize=size)
        Yj = jax.lax.fori_loop(1,size,mybody,init_val=(x_lo,x_hi,Yj))[2]
        return jax.lax.psum_scatter(Yj, row, scatter_dimension=0, tiled=True)

    

def MeshFlow_OS(Xij, Wij, row, col, K, B): #smallest block size B
    
    X_split = Xij.reshape(Xij.shape[0], Xij.shape[1]//(K*B),K,B)
    W_split = Wij.reshape(Wij.shape[0]//(K*B),K,B, Wij.shape[1])
    x_blocks = lambda i: jax.lax.dynamic_slice_in_dim(X_split, i, 1, 2)
    w_blocks = lambda i: jax.lax.dynamic_slice_in_dim(W_split, i, 1, 1)
    Xk0 = jax.lax.all_gather(x_blocks(0), col, tiled=True, axis=1)
    Wk0 = jax.lax.all_gather(w_blocks(0), row, tiled=True, axis=0)
    
    Yij = jnp.einsum('nikb,ikbo->no',Xk0,Wk0)
    Xk = jax.lax.all_gather(x_blocks(1), col, tiled=True, axis=1)
    Wk = jax.lax.all_gather(w_blocks(1), row, tiled=True, axis=0)
    def body_fn(kk,carry):
        YY, Xk,Wk = carry
        YY += jnp.einsum('nikb,ikbo->no',Xk,Wk)
        Xk2 = jax.lax.all_gather(x_blocks(kk+1), col, tiled=True, axis=1)
        Wk2 = jax.lax.all_gather(w_blocks(kk+1), row, tiled=True, axis=0)
        return YY,Xk2,Wk2
    Yij, Xk, Wk = jax.lax.fori_loop(1,K-1,body_fn,init_val=(Yij,Xk,Wk))
    return Yij + jnp.einsum('nikb,ikbo->no',Xk,Wk)

def MeshFlow_LS(Xij, Wji, row, col, K, B): #smallest block size B
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

def MeshFlow_RS(Xji, Wij, row, col, K, B): #smallest block size B
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



def Cannon_OS(Xij, Wij, row, col, K, B):
    N = jax.lax.psum(1, col)
    
    colidx = jax.lax.ax_index(axis_name=col)
    rowidx = jax.lax.ax_index(axis_name=row)

    shift_r = partial(jax.lax.ppermute, axis_name=col,
                        perm=[(i, (i + 1) % N) for i in range(N)])
    shift_l = partial(jax.lax.ppermute, axis_name=col,
                        perm=[(i, (i - 1) % N) for i in range(N)])
    shift_up = partial(jax.lax.ppermute, axis_name=row,
                        perm=[(i, (i + 1) % N) for i in range(N)])
    shift_dn = partial(jax.lax.ppermute, axis_name=row,
                        perm=[(i, (i - 1) % N) for i in range(N)])

    Xi = jax.lax.all_gather(Xij, col, tiled=True, axis=1)
    Wj = jax.lax.all_gather(Wij, row, tiled=True, axis=0)
    
    B = Xi.shape[1] // N // 2
    #print(B)
    k = (colidx+rowidx)%N
    x_lo = jax.lax.dynamic_slice_in_dim(Xi, (2*k) * B, B, 1)
    x_hi = jax.lax.dynamic_slice_in_dim(Xi, (2*k+1) * B, B, 1)
    w_lo = jax.lax.dynamic_slice_in_dim(Wj, (2*k) * B, B, 0)
    w_hi = jax.lax.dynamic_slice_in_dim(Wj, (2*k+1) * B, B, 0)
    #print(x_lo.shape)
    #print(w_hi.shape)
    out_block  =  x_lo @ w_lo
    out_block +=  x_hi @ w_hi
    w_lo = shift_dn(w_lo)
    w_hi = shift_up(w_hi)
    x_lo = shift_l(x_lo)
    x_hi = shift_r(x_hi)
    def body_fn(ii, carry):
        xl,xh,wl,wh,oblock = carry
        wl2 = shift_dn(wl)
        wh2 = shift_up(wh)
        xl2 = shift_l(xl)
        xh2 = shift_r(xh)
        oblock += xl @ wl
        oblock += xh @ wh
        return (xl2,xh2,wl2,wh2,oblock)
    
    return jax.lax.fori_loop(1,N,body_fn,init_val=(x_lo,x_hi,w_lo,w_hi,out_block))[4]

def Cannon_LS(Xij, Wji, row, col, K, B):
    N = jax.lax.psum(1, col)
    
    colidx = jax.lax.ax_index(axis_name=col)
    rowidx = jax.lax.ax_index(axis_name=row)
    k = (colidx + rowidx)%N
    B = Wji.shape[0] // 2
    shift_r = partial(jax.lax.ppermute, axis_name=col,
                        perm=[(i, (i + 1) % N) for i in range(N)])
    shift_l = partial(jax.lax.ppermute, axis_name=col,
                        perm=[(i, (i - 1) % N) for i in range(N)])
    shift_up = partial(jax.lax.ppermute, axis_name=row,
                        perm=[(i, (i + 1) % N) for i in range(N)])
    shift_dn = partial(jax.lax.ppermute, axis_name=row,
                        perm=[(i, (i - 1) % N) for i in range(N)])

    Wj = jax.lax.all_gather(Wji, row, tiled=True, axis=0)
    w_lo = jax.lax.dynamic_slice_in_dim(Wj, (2*k) * B, B, 0)
    w_hi = jax.lax.dynamic_slice_in_dim(Wj, (2*k+1) * B, B, 0)
    
    y_lo  = jnp.einsum('bi,oi->bo',Xij,w_lo)
    y_hi  = jnp.einsum('bi,oi->bo',Xij,w_hi)
    
    def body_fn(ii, carry):
        wl,wh,yl,yh = carry
        
        wl = shift_dn(wl)
        wh = shift_up(wh)
        yl = shift_l(yl)
        yh = shift_r(yh)
        lo = jnp.einsum('bi,oi->bo',Xij,wl)
        hi = jnp.einsum('bi,oi->bo',Xij,wh)
        
        return (wl,wh,yl+lo,yh+hi)
    
    _,_,y_lo,y_hi = jax.lax.fori_loop(1,N,body_fn,init_val=(w_lo,w_hi,y_lo,y_hi))
    y_lo = shift_l(y_lo)
    y_hi = shift_r(y_hi)
    Yik = jnp.concatenate([y_lo,y_hi],axis=1)
    Yi = jax.lax.all_gather(Yik, col, tiled=True, axis=1)
    return jax.lax.dynamic_slice_in_dim(Yi, 2*((colidx-rowidx)%N) * B, 2*B, axis=1)

def Cannon_RS(Xji, Wij, row, col, K, B):
    N = jax.lax.psum(1, col)
    
    colidx = jax.lax.ax_index(axis_name=col)
    rowidx = jax.lax.ax_index(axis_name=row)
    k = (colidx + rowidx)%N
    B = Xji.shape[1] // 2
    shift_r = partial(jax.lax.ppermute, axis_name=col,
                        perm=[(i, (i + 1) % N) for i in range(N)])
    shift_l = partial(jax.lax.ppermute, axis_name=col,
                        perm=[(i, (i - 1) % N) for i in range(N)])
    shift_up = partial(jax.lax.ppermute, axis_name=row,
                        perm=[(i, (i + 1) % N) for i in range(N)])
    shift_dn = partial(jax.lax.ppermute, axis_name=row,
                        perm=[(i, (i - 1) % N) for i in range(N)])

    Xi = jax.lax.all_gather(Xji, col, tiled=True, axis=1)
    x_lo = jax.lax.dynamic_slice_in_dim(Xi, (2*k) * B, B, 1)
    x_hi = jax.lax.dynamic_slice_in_dim(Xi, (2*k+1) * B, B, 1)
    
    y_lo  = jnp.einsum('ib,io->bo',x_lo,Wij)
    y_hi  = jnp.einsum('ib,io->bo',x_hi,Wij)
    
    def body_fn(ii, carry):
        xl,xh,yl,yh = carry
        
        xl = shift_l(xl)
        xh = shift_r(xh)
        yl = shift_dn(yl)
        yh = shift_up(yh)
        lo = jnp.einsum('ib,io->bo',xl,Wij)
        hi = jnp.einsum('ib,io->bo',xh,Wij)
        return (xl,xh,yl+lo,yh+hi)
    
    _,_,y_lo,y_hi = jax.lax.fori_loop(1,N,body_fn,init_val=(x_lo,x_hi,y_lo,y_hi))
    y_lo = shift_dn(y_lo)
    y_hi = shift_up(y_hi)
    Ykj = jnp.concatenate([y_lo,y_hi],axis=0)
    Yj = jax.lax.all_gather(Ykj, row, tiled=True, axis=0)
    return jax.lax.dynamic_slice_in_dim(Yj, 2*((rowidx-colidx)%N) * B, 2*B, axis=0)

class SPMD:
    def __init__(self, mesh, algorithm='collective', blocksize=8, jit=True):
        self.mesh = mesh
        self.rowaxis = mesh.axis_names[0]
        self.colaxis = mesh.axis_names[1]
        self.spec = P(self.rowaxis,self.colaxis)
        self.blocksize=blocksize
        self.jit = jit
        if algorithm=='collective':
            self.funcs = {'os':Collective_OS, 'ls':Collective_LS, 'rs':Collective_RS}
        elif algorithm=='wang':
            self.funcs = {'os':Wang_OS, 'ls':Wang_LS, 'rs':Wang_RS}
        elif algorithm=='cannon':
            assert mesh.devices.shape[0] == mesh.devices.shape[1]
            self.funcs = {'os':Cannon_OS, 'ls':Cannon_LS, 'rs':Cannon_RS}
        else:
            self.funcs = {'os':MeshFlow_OS, 'ls':MeshFlow_LS, 'rs':MeshFlow_RS}
    def OS(self, K=8):
        myfunc = partial(shard_map, mesh=self.mesh, in_specs=(self.spec,self.spec),
         out_specs=self.spec)(partial(
             self.funcs['os'], row=self.rowaxis, col=self.colaxis, K=K,B=self.blocksize))
        return jax.jit(myfunc) if self.jit else myfunc
    def LS(self, K=8):
        myfunc = partial(shard_map, mesh=self.mesh, in_specs=(self.spec,self.spec),
         out_specs=self.spec)(partial(
             self.funcs['ls'], row=self.rowaxis, col=self.colaxis, K=K,B=self.blocksize))
        return jax.jit(myfunc) if self.jit else myfunc
    def RS(self, K=8):
        myfunc = partial(shard_map, mesh=self.mesh, in_specs=(self.spec,self.spec),
         out_specs=self.spec)(partial(
             self.funcs['rs'], row=self.rowaxis, col=self.colaxis, K=K,B=self.blocksize))
        return jax.jit(myfunc) if self.jit else myfunc

def Cannon_RS(Xji, Wij, row, col, K, B):
    N = jax.lax.psum(1, col)
    
    colidx = jax.lax.ax_index(axis_name=col)
    rowidx = jax.lax.ax_index(axis_name=row)
    k = (colidx + rowidx)%N
    B = Xji.shape[1] // 2
    shift_r = partial(jax.lax.ppermute, axis_name=col,
                        perm=[(i, (i + 1) % N) for i in range(N)])
    shift_l = partial(jax.lax.ppermute, axis_name=col,
                        perm=[(i, (i - 1) % N) for i in range(N)])
    shift_up = partial(jax.lax.ppermute, axis_name=row,
                        perm=[(i, (i + 1) % N) for i in range(N)])
    shift_dn = partial(jax.lax.ppermute, axis_name=row,
                        perm=[(i, (i - 1) % N) for i in range(N)])

    Xi = jax.lax.all_gather(Xji, col, tiled=True, axis=1)
    x_lo = jax.lax.dynamic_slice_in_dim(Xi, (2*k) * B, B, 1)
    x_hi = jax.lax.dynamic_slice_in_dim(Xi, (2*k+1) * B, B, 1)
    
    y_lo  = jnp.einsum('ib,io->bo',x_lo,Wij)
    y_hi  = jnp.einsum('ib,io->bo',x_hi,Wij)
    y_lo = shift_dn(y_lo)
    y_hi = shift_up(y_hi)
    def body_fn(ii, carry):
        xl,xh,yl,yh = carry
        
        xl = shift_l(xl)
        xh = shift_r(xh)
        yl += jnp.einsum('ib,io->bo',xl,Wij)
        yh += jnp.einsum('ib,io->bo',xh,Wij)
        yl = shift_dn(yl)
        yh = shift_up(yh)
        return (xl,xh,yl,yh)
    
    _,_,y_lo,y_hi = jax.lax.fori_loop(1,N,body_fn,init_val=(x_lo,x_hi,y_lo,y_hi))
    Ykj = jnp.concatenate([y_lo,y_hi],axis=0)
    Yj = jax.lax.all_gather(Ykj, row, tiled=True, axis=0)
    return jax.lax.dynamic_slice_in_dim(Yj, 2*((rowidx-colidx)%N) * B, 2*B, axis=0)

def createMultihostMatrix(mesh, sharding, global_shape, dtype=jnp.bfloat16): #only works for 2d
    local_shape = [global_shape[0]//mesh.devices.shape[0], global_shape[1]]
    local_buffer = jax.random.normal(jax.random.PRNGKey(jax.process_index()),local_shape, dtype=dtype)
    local_sharded = jax.device_put(jnp.split(local_buffer,len(mesh.local_devices),axis=1), mesh.local_devices)
    return jax.make_array_from_single_device_arrays(global_shape,sharding, local_sharded)

if __name__=='__main__':
    
    B,S,H,D = (4,128, 48,64)

    jax.config.update('jax_platform_name', 'cpu')
    os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=16' # Use 16 CPU devices
    devices = mesh_utils.create_device_mesh((4, 4))
    mesh = Mesh(devices, axis_names=('x', 'y'))
    X = jnp.arange( B*S*H*D,dtype=jnp.float32).reshape(B*S, H*D)/(B*S*H*D)
    W = jnp.arange(H*D*4*H*D,dtype=jnp.float32).reshape(H*D, 4*H*D) / (4*H*D*H*D)
    Y = X@W
    myalg = SPMD(mesh,'wang')
    
    Xo = jax.device_put(X, NamedSharding(mesh, P('x', 'y')))
    Wo = jax.device_put(W, NamedSharding(mesh, P('x', 'y')))
    
    
    myalg = SPMD(mesh,sys.argv[1])
    gspmd = SPMD(mesh,'collective')
    my_os = myalg.OS()
    gspmd_os = gspmd.OS()
    Yo = my_os(Xo,Wo)
    print(allclose(Yo,gspmd_os(Xo,Wo)))

    my_=myalg.LS()
    gspmd_ = gspmd.LS()
    Xp = my_(Yo,Wo)
    print(allclose(Xp,gspmd_(Yo,Wo)))

    RS = myalg.RS()
    gspmd_rs = gspmd.RS()
    Wp2 = RS(Xo,Yo)
    print(allclose(Wp2,gspmd_rs(Xo,Yo)))