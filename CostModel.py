import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import time


class DeviceMesh:
    def __init__(self, device_shape: tuple, flops_per_chip: int, bws_per_direction, link_latencies, base_overheads):
        self.reshape(device_shape)
        
        self.flops = flops_per_chip
        self.bws = bws_per_direction
        self.link_latencies = link_latencies
        self.base_overheads = base_overheads
    def reshape(self, newshape):
        self.shape = newshape
        self.meshdim = None
        self.submesh = None
        if type(newshape[0]) is tuple:
            self.meshdim = 0
            self.submesh = newshape[0]
            self.shape =(newshape[0][0]*newshape[0][1], newshape[1])
        elif type(newshape[1]) is tuple:
            self.meshdim = 1
            self.submesh = newshape[1]
            self.shape = (newshape[0], newshape[1][0]*newshape[1][1])
        else:
            self.shape=newshape

def bytesize(shape):
    return shape[0]*shape[1]*2
def estimateBroadcast(mesh, data_shape: tuple, dim: int, precision):
    data_size = data_shape[0]*data_shape[1]*precision//8
    time_per_step = data_size / bw 
    link_latency = mesh.link_latencies['allgather']
    base_overhead = mesh.base_overheads['allgather']
    bw = mesh.bws['allgather'][dim]
    if dim is mesh.meshdim:
        r,c = mesh.submesh
        halfsize = data_size // 2
        total_latency = link_latency * (r+c-1)
        
        return base_overhead + halfsize / bw + total_latency
    return base_overhead + time_per_step + link_latency*(mesh.shape[dim]-1)

def estimateAllgather(mesh, data_shape: tuple, dim: int, precision):
    data_size = data_shape[0]*data_shape[1]*precision//8
    steps = mesh.shape[dim]-1
    
    link_latency = mesh.link_latencies['allgather']
    base_overhead = mesh.base_overheads['allgather']
    bw = mesh.bws['allgather'][dim]
    time_per_step = data_size / bw # bidirectional
    if dim is mesh.meshdim:
        r,c = mesh.submesh
        halfsize = data_size // 2
        total_latency = link_latency * (r+c-2)
        row_first = (halfsize/bw) * (r-1) + (halfsize*r/bw)*(c-1)
        col_first = (halfsize/bw) * (c-1) + (halfsize*c/bw)*(r-1)
        
        return base_overhead + max(row_first, col_first) + total_latency
    return base_overhead + steps*time_per_step + link_latency*steps
    
def estimateSkew(mesh, data_shape: tuple, precision):
    assert mesh.shape[0] == mesh.shape[1]
    link_latency = mesh.link_latencies['sendrecv']
    base_overhead = mesh.base_overheads['sendrecv']
    bw = mesh.bws['sendrecv'][1]
    steps = mesh.shape[0]//2
    time_per_step = data_shape[0]*data_shape[1] * (precision//8) / bw
    return base_overhead + steps * time_per_step + link_latency*steps

def estimateMatmul(mesh, M, N, K, input_precision=jnp.bfloat16, layout='nn', repeat=5):
    #flop_count = M*N*K*2
    #return flop_count/mesh.flops
    
    A = jnp.ones([M,K], dtype=input_precision)
    B = jnp.ones([K,N],dtype=input_precision)
    einsum_rule = 'mn,nk->mk'
    if layout == 'nt':
        B = B.transpose()
        einsum_rule = 'mn,kn->mk'
    elif layout == 'tn':
        A = A.transpose()
        einsum_rule = 'nm,nk->mk'
    
    
    MM = jax.jit(partial(jnp.einsum, einsum_rule))
    C = MM(A,B).block_until_ready()

    starttime = time.time()
    for _ in range(repeat):
        MM(A,B).block_until_ready()
    endtime = time.time()
    
    
    return (endtime-starttime)/repeat
    
#def estimateMatmul(mesh, M, N, K, input_precision=jnp.bfloat16, output_precision=jnp.float32, repeat=10):
#    flop_count = M*N*K*2
#    return flop_count/mesh.flops
    
    

def estimateReduce(mesh, data_shape: tuple, dim: int, precision):
    data_size = data_shape[0]*data_shape[1]*precision//8
    
    link_latency = mesh.link_latencies['reducescatter']
    base_overhead = mesh.base_overheads['reducescatter']
    bw = mesh.bws['reducescatter'][dim]
    time_per_step = data_size / bw 
    if dim is mesh.meshdim:
        r,c = mesh.submesh
        halfsize = data_size // 2
        total_latency = link_latency * (r+c-1)
        return base_overhead + halfsize / bw + total_latency
    return base_overhead + time_per_step + link_latency*(mesh.shape[dim]-1)

def estimateReduceScatter(mesh, data_shape: tuple, dim: int, precision):
    data_size = data_shape[0]*data_shape[1]*precision//8
    link_latency = mesh.link_latencies['reducescatter']
    base_overhead = mesh.base_overheads['reducescatter']
    bw = mesh.bws['reducescatter'][dim]
    steps = mesh.shape[dim]-1
    time_per_step = data_size / bw # bidirectional
    if dim is mesh.meshdim:
        r,c = mesh.submesh
        halfsize = data_size // 2
        total_latency = link_latency * (r+c-2)
        row_first = (halfsize/bw) * (r-1) + (halfsize*r/bw)*(c-1)
        col_first = (halfsize/bw) * (c-1) + (halfsize*c/bw)*(r-1)
        return base_overhead + max(row_first, col_first) + total_latency
    return base_overhead + steps*time_per_step + link_latency*steps

def SUMMA_OS(mesh, M, N, K, steps=8, precisions=(16,16,16)):
    steps = np.lcm(mesh.shape[0], mesh.shape[1])
    ishape = (M//mesh.shape[0], K//steps)
    wshape = (K//steps, N//mesh.shape[1])
    broadcast_i = estimateBroadcast(mesh, ishape, 1, precision=precisions[0])
    broadcast_w = estimateBroadcast(mesh, wshape, 0, precision=precisions[1])
    compute = estimateMatmul(mesh, ishape[0], wshape[1], ishape[1])

    return (max(broadcast_i, broadcast_w), max(broadcast_i, broadcast_w, compute) * (steps-1) + compute)

def SUMMA_LS(mesh, M, N, K, steps=8, precisions=(16,16,16)):
    steps = np.lcm(mesh.shape[0], mesh.shape[1])
    ishape = (M//mesh.shape[0], K//mesh.shape[1])
    oshape = (M//mesh.shape[0], N//steps)
    wshape = (N//steps, K//mesh.shape[1])
    broadcast_w = estimateBroadcast(mesh, wshape, 0, precision=precisions[1])
    reduce_o = estimateReduce(mesh, oshape, 1, precision=precisions[2])

    compute = estimateMatmul(mesh, ishape[0], oshape[1], ishape[1], layout='nt')

    return (broadcast_w + reduce_o, max(reduce_o, broadcast_w, compute) * (steps-1) + compute)

def SUMMA_RS(mesh, M, N, K, steps=8, precisions=(16,16,16)):
    steps = np.lcm(mesh.shape[0], mesh.shape[1])
    ishape = (K//mesh.shape[0], M//steps)
    oshape = (M//steps, N//mesh.shape[1])
    wshape = (K//mesh.shape[0], N//mesh.shape[1])
    broadcast_i = estimateBroadcast(mesh, ishape, 1, precision=precisions[0])
    reduce_o = estimateReduce(mesh, oshape,0, precision=precisions[2])

    compute = estimateMatmul(mesh, oshape[0], oshape[1], wshape[0], layout='tn')

    return (broadcast_i + reduce_o, max(reduce_o, broadcast_i, compute) * (steps-1) + compute)

def Cannon_OS(mesh, M, N, K, steps=8, precisions=(16,16,16)):
    link_latency = mesh.link_latencies['sendrecv']
    base_overhead = mesh.base_overheads['sendrecv']
    bw = mesh.bws['sendrecv'][1]
    ishape = (M//mesh.shape[0], K//(mesh.shape[1]))
    wshape = (K//(mesh.shape[0]), N//mesh.shape[1])
    skew_i = estimateSkew(mesh, ishape)
    skew_w = estimateSkew(mesh, wshape)
    skew = max(skew_i, skew_w)
    compute = estimateMatmul(mesh, ishape[0], wshape[1], ishape[1])
    roll_i = bytesize(ishape)/bw + link_latency + base_overhead
    roll_w = bytesize(wshape)/bw + link_latency + base_overhead
    roll = max(roll_i, roll_w)
    steps = mesh.shape[0]
    return (skew, steps*max(roll, compute))

def Cannon_LS(mesh, M, N, K, steps=8, precisions=(16,16,16)):
    ishape = (M//mesh.shape[0], K//(mesh.shape[1]))
    wshape = (N//(mesh.shape[0]), K//mesh.shape[1])
    oshape = (M//(mesh.shape[0]), N//mesh.shape[1])
    link_latency = mesh.link_latencies['sendrecv']
    base_overhead = mesh.base_overheads['sendrecv']
    bw = mesh.bws['sendrecv'][1]
    skew_o = estimateSkew(mesh, oshape)
    skew_w = estimateSkew(mesh, wshape)
    skew = max(skew_o, skew_w)
    compute = estimateMatmul(mesh, oshape[0], oshape[1], ishape[1], layout='nt')
    roll_o = bytesize(oshape)/bw + link_latency + base_overhead
    roll_w = bytesize(wshape)/bw + link_latency + base_overhead
    roll = max(roll_o, roll_w)
    steps = mesh.shape[0]
    return (skew, steps*max(roll, compute))

def Cannon_RS(mesh, M, N, K, steps=8, precisions=(16,16,16)):
    ishape = (M//mesh.shape[0], K//(mesh.shape[1]))
    wshape = (N//(mesh.shape[0]), K//mesh.shape[1])
    oshape = (M//(mesh.shape[0]), N//mesh.shape[1])
    link_latency = mesh.link_latencies['sendrecv']
    base_overhead = mesh.base_overheads['sendrecv']
    bw = mesh.bws['sendrecv'][1]
    skew_o = estimateSkew(mesh, oshape)
    skew_i = estimateSkew(mesh, ishape)
    skew = max(skew_o, skew_i)
    compute = estimateMatmul(mesh, oshape[0], oshape[1], ishape[1], layout='nt')
    roll_o = bytesize(oshape)/bw + link_latency + base_overhead
    roll_i = bytesize(ishape)/bw + link_latency + base_overhead
    roll = max(roll_o, roll_i)
    steps = mesh.shape[0]
    return (skew, steps*max(roll, compute))

def GSPMD_OS(mesh, M, N, K, steps=8, precisions=(16,16,16)):
    ishape = (M//mesh.shape[0], K//(mesh.shape[1]))
    wshape = (K//(mesh.shape[0]), N//mesh.shape[1])
    allgather_i = estimateAllgather(mesh, ishape, 1, precision=precisions[0])
    allgather_w = estimateAllgather(mesh, wshape, 0, precision=precisions[1])
    compute = estimateMatmul(mesh, ishape[0], wshape[1], K)
    
    return (allgather_i, max(compute,allgather_w))

def GSPMD_LS(mesh, M, N, K, steps=8, precisions=(16,16,16)):
    ishape = (M//mesh.shape[0], K//(mesh.shape[1]))
    wshape = (N//(mesh.shape[0]), K//mesh.shape[1])
    oshape = (M//(mesh.shape[0]), N//mesh.shape[1])
    
    allgather_w = estimateAllgather(mesh, wshape, 0, precision=precisions[1])
    compute = estimateMatmul(mesh, ishape[0], N, ishape[1], layout='nt')
    reducescatter_o = estimateReduceScatter(mesh, oshape, 1, precision=precisions[2])
    return (reducescatter_o, max(allgather_w,compute))

def GSPMD_RS(mesh, M, N, K, steps=8, precisions=(16,16,16)):
    ishape = (K//mesh.shape[0], M//(mesh.shape[1]))
    wshape = (K//(mesh.shape[0]), N//mesh.shape[1])
    oshape = (M//(mesh.shape[0]), N//mesh.shape[1])
    
    allgather_i = estimateAllgather(mesh, ishape, 1, precision=precisions[0])
    compute = estimateMatmul(mesh, M, wshape[1], ishape[0], layout='tn')
    reducescatter_o = estimateReduceScatter(mesh, oshape, 0, precision=precisions[2])
    return (reducescatter_o, max(allgather_i,compute))

def MeshFlow_OS(mesh, M, N, K, steps=32, precisions=(16,16,16)): #assume synchronized allgather

    ishape = (M//mesh.shape[0], K//mesh.shape[1]//steps)
    wshape = (K//mesh.shape[0]//steps, N//mesh.shape[1])
    
    allgather_i = estimateAllgather(mesh, ishape, 1, precision=precisions[0])
    allgather_w = estimateAllgather(mesh, wshape, 0, precision=precisions[1])
    compute = estimateMatmul(mesh, ishape[0], wshape[1], K//steps)
    
    return (max(allgather_i,allgather_w), max(allgather_i, allgather_w, compute) * (steps-1) + compute)

def MeshFlow_LS(mesh, M, N, K, steps=32, precisions=(16,16,16)): #assume synchronized allgather
    
    ishape = (M//mesh.shape[0], K//mesh.shape[1])
    wshape = (N//mesh.shape[0]//steps, K//mesh.shape[1])
    oshape = (M//mesh.shape[0], N//mesh.shape[1]//steps)
    
    allgather_w = estimateAllgather(mesh, wshape, 0, precision=precisions[1])
    reducescatter_o = estimateReduceScatter(mesh, oshape, 1, precision=precisions[2])
    compute = estimateMatmul(mesh, ishape[0], N//steps, ishape[1], layout='nt')
    
    #startup = max(mesh.shape[0],mesh.shape[1])-1
    #return (roll_o+ mesh.link_latency*mesh.shape[1], max(roll_o, roll_w, compute) * (steps-1) + compute)
    return (allgather_w+reducescatter_o, compute + max(allgather_w, reducescatter_o, compute) * (steps-1))
def MeshFlow_RS(mesh, M, N, K, steps=32, precisions=(16,16,16)): #assume synchronized allgather
    #steps = np.lcm(mesh.shape[0], mesh.shape[1])*min(np.gcd(mesh.shape[0], mesh.shape[1]), multiplier)
    #steps = np.lcm(mesh.shape[0], mesh.shape[1])
    ishape = (K//mesh.shape[0], M//mesh.shape[1]//steps)
    wshape = (K//mesh.shape[0], N//mesh.shape[1])
    oshape = (M//mesh.shape[0]//steps, N//mesh.shape[1])
    
    allgather_i = estimateAllgather(mesh, ishape, 1, precision=precisions[0])
    reducescatter_o = estimateReduceScatter(mesh, oshape, 0, precision=precisions[2])
    compute = estimateMatmul(mesh, M//steps, oshape[1], ishape[0], layout='tn')
    
    #startup = max(mesh.shape[0],mesh.shape[1])-1
    return (allgather_i+ reducescatter_o, compute + max(allgather_i, reducescatter_o, compute) * (steps-1))

def LogicalRing_ISplit(mesh, M, N, K, steps=8):
    P = mesh.shape[0]*mesh.shape[1]
    ishape = (M, K//P)
    wshape = (K, N//P)
    link_latency = mesh.link_latencies['sendrecv']
    base_overhead = mesh.base_overheads['sendrecv']
    bw = mesh.bws['sendrecv'][1]
    roll_i = bytesize(ishape)/(bw*2) + link_latency + base_overhead
    compute = estimateMatmul(mesh, M, N//P, K//P)
    return (0, max(roll_i, compute) * P)
def LogicalRing_WSplit(mesh, M, N, K, steps=8):
    P = mesh.shape[0]*mesh.shape[1]
    ishape = (M//P, K)
    wshape = (K//P, N)
    link_latency = mesh.link_latencies['sendrecv']
    base_overhead = mesh.base_overheads['sendrecv']
    bw = mesh.bws['sendrecv'][1]
    roll_w = bytesize(wshape)/(bw*2) + link_latency + base_overhead
    compute = estimateMatmul(mesh, M//P, N, K//P)
    return (0, max(roll_w, compute) * P)

def LogicalRing_OSplit(mesh, M, N, K, steps=8):
    P = mesh.shape[0]*mesh.shape[1]
    ishape = (M, K//P)
    wshape = (K//P, N)
    oshape = (M, N//P)
    link_latency = mesh.link_latencies['sendrecv']
    base_overhead = mesh.base_overheads['sendrecv']
    bw = mesh.bws['sendrecv'][1]
    roll_o = bytesize(oshape)/(bw*2) + link_latency + base_overhead
    compute = estimateMatmul(mesh, M, N//P, K//P)
    return (0, max(roll_o, compute) * P)

def tup2ms(t):
    return (str(t[0]*1000)+'ms', str(t[1]*1000)+'ms', str((t[0]+t[1])*1000)+'ms')