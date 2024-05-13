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
    time_per_step = data_size / (bw*2) 
    link_latency = mesh.link_latencies['allgather']
    base_overhead = mesh.base_overheads['allgather']
    bw = mesh.bws['allgather']
    if dim is mesh.meshdim:
        r,c = mesh.submesh
        halfsize = data_size // 2
        total_latency = link_latency * (r+c-1)
        
        return base_overhead + halfsize / (bw*2) + total_latency
    return base_overhead + time_per_step + link_latency*(mesh.shape[dim]-1)

def estimateAllgather(mesh, data_shape: tuple, dim: int, precision):
    data_size = data_shape[0]*data_shape[1]*precision//8
    steps = mesh.shape[dim]-1
    
    link_latency = mesh.link_latencies['allgather']
    base_overhead = mesh.base_overheads['allgather']
    bw = mesh.bws['allgather']
    time_per_step = data_size / (bw*2) # bidirectional
    if dim is mesh.meshdim:
        r,c = mesh.submesh
        halfsize = data_size // 2
        total_latency = link_latency * (r+c-2)
        row_first = (halfsize/(bw*2)) * (r-1) + (halfsize*r/(bw*2))*(c-1)
        col_first = (halfsize/(bw*2)) * (c-1) + (halfsize*c/(bw*2))*(r-1)
        
        return base_overhead + max(row_first, col_first) + total_latency
    return base_overhead + steps*time_per_step + link_latency*steps
    
def estimateSkew(mesh, data_shape: tuple, precision):
    assert mesh.shape[0] == mesh.shape[1]
    link_latency = mesh.link_latencies['sendrecv']
    base_overhead = mesh.base_overheads['sendrecv']
    bw = mesh.bws['sendrecv']
    steps = mesh.shape[0]//2
    time_per_step = data_shape[0]*data_shape[1] * (precision//8) / (bw*2)
    return base_overhead + steps * time_per_step + link_latency*steps

def estimateMatmul(mesh, M, N, K, input_precision=jnp.bfloat16, layout='nn', repeat=10):
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
    bw = mesh.bws['reducescatter']
    time_per_step = data_size / (bw*2) 
    if dim is mesh.meshdim:
        r,c = mesh.submesh
        halfsize = data_size // 2
        total_latency = link_latency * (r+c-1)
        return base_overhead + halfsize / (bw*2) + total_latency
    return base_overhead + time_per_step + link_latency*(mesh.shape[dim]-1)

def estimateReduceScatter(mesh, data_shape: tuple, dim: int, precision):
    data_size = data_shape[0]*data_shape[1]*precision//8
    link_latency = mesh.link_latencies['reducescatter']
    base_overhead = mesh.base_overheads['reducescatter']
    bw = mesh.bws['reducescatter']
    steps = mesh.shape[dim]-1
    time_per_step = data_size / (bw*2) # bidirectional
    if dim is mesh.meshdim:
        r,c = mesh.submesh
        halfsize = data_size // 2
        total_latency = link_latency * (r+c-2)
        row_first = (halfsize/(bw*2)) * (r-1) + (halfsize*r/(bw*2))*(c-1)
        col_first = (halfsize/(bw*2)) * (c-1) + (halfsize*c/(bw*2))*(r-1)
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

def SUMMA_IS(mesh, M, N, K, steps=8, precisions=(16,16,16)):
    steps = np.lcm(mesh.shape[0], mesh.shape[1])
    ishape = (M//mesh.shape[0], K//mesh.shape[1])
    oshape = (M//mesh.shape[0], N//steps)
    wshape = (N//steps, K//mesh.shape[1])
    broadcast_w = estimateBroadcast(mesh, wshape, 0, precision=precisions[1])
    reduce_o = estimateReduce(mesh, oshape, 1, precision=precisions[2])

    compute = estimateMatmul(mesh, ishape[0], oshape[1], ishape[1], layout='nt')

    return (broadcast_w + reduce_o, max(reduce_o, broadcast_w, compute) * (steps-1) + compute)

def SUMMA_WS(mesh, M, N, K, steps=8, precisions=(16,16,16)):
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
    bw = mesh.bws['sendrecv']
    ishape = (M//mesh.shape[0], K//(mesh.shape[1]))
    wshape = (K//(mesh.shape[0]), N//mesh.shape[1])
    skew_i = estimateSkew(mesh, ishape)
    skew_w = estimateSkew(mesh, wshape)
    skew = max(skew_i, skew_w)
    compute = estimateMatmul(mesh, ishape[0], wshape[1], ishape[1])
    roll_i = bytesize(ishape)/(bw*2) + link_latency + base_overhead
    roll_w = bytesize(wshape)/(bw*2) + link_latency + base_overhead
    roll = max(roll_i, roll_w)
    steps = mesh.shape[0]
    return (skew, steps*max(roll, compute))

def Cannon_IS(mesh, M, N, K, steps=8, precisions=(16,16,16)):
    ishape = (M//mesh.shape[0], K//(mesh.shape[1]))
    wshape = (N//(mesh.shape[0]), K//mesh.shape[1])
    oshape = (M//(mesh.shape[0]), N//mesh.shape[1])
    link_latency = mesh.link_latencies['sendrecv']
    base_overhead = mesh.base_overheads['sendrecv']
    bw = mesh.bws['sendrecv']
    skew_o = estimateSkew(mesh, oshape)
    skew_w = estimateSkew(mesh, wshape)
    skew = max(skew_o, skew_w)
    compute = estimateMatmul(mesh, oshape[0], oshape[1], ishape[1], layout='nt')
    roll_o = bytesize(oshape)/(bw*2) + link_latency + base_overhead
    roll_w = bytesize(wshape)/(bw*2) + link_latency + base_overhead
    roll = max(roll_o, roll_w)
    steps = mesh.shape[0]
    return (skew, steps*max(roll, compute))

def Cannon_WS(mesh, M, N, K, steps=8, precisions=(16,16,16)):
    ishape = (M//mesh.shape[0], K//(mesh.shape[1]))
    wshape = (N//(mesh.shape[0]), K//mesh.shape[1])
    oshape = (M//(mesh.shape[0]), N//mesh.shape[1])
    link_latency = mesh.link_latencies['sendrecv']
    base_overhead = mesh.base_overheads['sendrecv']
    bw = mesh.bws['sendrecv']
    skew_o = estimateSkew(mesh, oshape)
    skew_i = estimateSkew(mesh, ishape)
    skew = max(skew_o, skew_i)
    compute = estimateMatmul(mesh, oshape[0], oshape[1], ishape[1], layout='nt')
    roll_o = bytesize(oshape)/(bw*2) + link_latency + base_overhead
    roll_i = bytesize(ishape)/(bw*2) + link_latency + base_overhead
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

def GSPMD_IS(mesh, M, N, K, steps=8, precisions=(16,16,16)):
    ishape = (M//mesh.shape[0], K//(mesh.shape[1]))
    wshape = (N//(mesh.shape[0]), K//mesh.shape[1])
    oshape = (M//(mesh.shape[0]), N//mesh.shape[1])
    
    allgather_w = estimateAllgather(mesh, wshape, 0, precision=precisions[1])
    compute = estimateMatmul(mesh, ishape[0], N, ishape[1], layout='nt')
    reducescatter_o = estimateReduceScatter(mesh, oshape, 1, precision=precisions[2])
    return (reducescatter_o, max(allgather_w,compute))

def GSPMD_WS(mesh, M, N, K, steps=8, precisions=(16,16,16)):
    ishape = (K//mesh.shape[0], M//(mesh.shape[1]))
    wshape = (K//(mesh.shape[0]), N//mesh.shape[1])
    oshape = (M//(mesh.shape[0]), N//mesh.shape[1])
    
    allgather_i = estimateAllgather(mesh, ishape, 1, precision=precisions[0])
    compute = estimateMatmul(mesh, M, wshape[1], ishape[0], layout='tn')
    reducescatter_o = estimateReduceScatter(mesh, oshape, 0, precision=precisions[2])
    return (reducescatter_o, max(allgather_i,compute))

def Systolic_OS(mesh, M, N, K, steps=32, precisions=(16,16,16)): #assume synchronized allgather

    ishape = (M//mesh.shape[0], K//mesh.shape[1]//steps)
    wshape = (K//mesh.shape[0]//steps, N//mesh.shape[1])
    
    allgather_i = estimateAllgather(mesh, ishape, 1, precision=precisions[0])
    allgather_w = estimateAllgather(mesh, wshape, 0, precision=precisions[1])
    compute = estimateMatmul(mesh, ishape[0], wshape[1], K//steps)
    
    return (max(allgather_i,allgather_w), max(allgather_i, allgather_w, compute) * (steps-1) + compute)

def Systolic_IS(mesh, M, N, K, steps=32, precisions=(16,16,16)): #assume synchronized allgather
    
    ishape = (M//mesh.shape[0], K//mesh.shape[1])
    wshape = (N//mesh.shape[0]//steps, K//mesh.shape[1])
    oshape = (M//mesh.shape[0], N//mesh.shape[1]//steps)
    
    allgather_w = estimateAllgather(mesh, wshape, 0, precision=precisions[1])
    reducescatter_o = estimateReduceScatter(mesh, oshape, 1, precision=precisions[2])
    compute = estimateMatmul(mesh, ishape[0], N//steps, ishape[1], layout='nt')
    
    #startup = max(mesh.shape[0],mesh.shape[1])-1
    #return (roll_o+ mesh.link_latency*mesh.shape[1], max(roll_o, roll_w, compute) * (steps-1) + compute)
    return (allgather_w+reducescatter_o, compute + max(allgather_w, reducescatter_o, compute) * (steps-1))
def Systolic_WS(mesh, M, N, K, steps=32, precisions=(16,16,16)): #assume synchronized allgather
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
    bw = mesh.bws['sendrecv']
    roll_i = bytesize(ishape)/(bw*2) + link_latency + base_overhead
    compute = estimateMatmul(mesh, M, N//P, K//P)
    return (0, max(roll_i, compute) * P)
def LogicalRing_WSplit(mesh, M, N, K, steps=8):
    P = mesh.shape[0]*mesh.shape[1]
    ishape = (M//P, K)
    wshape = (K//P, N)
    link_latency = mesh.link_latencies['sendrecv']
    base_overhead = mesh.base_overheads['sendrecv']
    bw = mesh.bws['sendrecv']
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
    bw = mesh.bws['sendrecv']
    roll_o = bytesize(oshape)/(bw*2) + link_latency + base_overhead
    compute = estimateMatmul(mesh, M, N//P, K//P)
    return (0, max(roll_o, compute) * P)

def tup2ms(t):
    return (str(t[0]*1000)+'ms', str(t[1]*1000)+'ms', str((t[0]+t[1])*1000)+'ms')

def MeshTune(mesh, func, M, N, K, verbose=False, precisions=(16,16,16), steps=16):
    meshsize = mesh.shape[0]*mesh.shape[1]
    meshshape = (2, meshsize//2)
    mesh.reshape(meshshape)
    comm, overlap = func(mesh, M, N, K, precisions=precisions, steps=steps)
    besttime = comm+overlap
    bestshape = meshshape
    while meshshape[1] >= 2:
        comm, overlap = func(mesh, M, N, K, precisions=precisions, steps=steps)
        curtime = comm+overlap
        if curtime < besttime:
            bestshape = meshshape
            besttime = curtime
        if verbose:
            print("Best mesh for {} is {}x{}, result: {}".format(
                func.__name__,meshshape[0],meshshape[1],tup2ms(func(mesh, M, N, K, precisions=precisions, steps=steps))))
        meshshape = (meshshape[0]*2, meshshape[1]//2)
        mesh.reshape(meshshape)
        
    mesh.reshape(bestshape)
    print("Best mesh for {} is {}x{}, result: {}".format(
        func.__name__,bestshape[0],bestshape[1],tup2ms(func(mesh, M, N, K, precisions=precisions, steps=steps))))
    return bestshape

def MeshTune3D(dims, func, M, N, K, flops, bw, link_latency=1e-6, precisions=(16,16,16), verbose=False):
    shapes = [(dims[0],(dims[1],dims[2])), (dims[1],(dims[0],dims[2])),(dims[2],(dims[1],dims[0])),
              ((dims[1],dims[2]), dims[0]), ((dims[0],dims[2]), dims[1]), ((dims[1],dims[0]), dims[2])]
    mesh = DeviceMesh(shapes[0], flops, bw, link_latency)
    
    comm, overlap = func(mesh, M, N, K, precisions=precisions)
    besttime = comm+overlap
    bestshape = shapes[0]
    for s in shapes:
        mesh.reshape(s)
        comm, overlap = func(mesh, M, N, K, precisions=precisions)
        curtime = comm+overlap
        if curtime < besttime:
            bestshape = s
            besttime = curtime
        if verbose:
            print("For {}, mesh shape {}, result: {}, {}".format(
                func.__name__,s,comm, overlap))
    mesh.reshape(bestshape)
    print("Best mesh for {} is {}, result: {}".format(
        func.__name__,bestshape,tup2ms(func(mesh, M, N, K, precisions=precisions))))
    return bestshape

def AutotuneLayer3D(dims, B, I, O, algorithm='SUMMA', flops=240*1024**4, bw=40*1024**3,link_latency=6e-6, verbose=True, steps=32):
    
    if algorithm == 'SUMMA':
        funcs = {'OS':SUMMA_OS, 'IS':SUMMA_IS, 'WS':SUMMA_WS}
    elif algorithm == 'GSPMD':
        funcs = {'OS':GSPMD_OS, 'IS':GSPMD_IS, 'WS':GSPMD_WS}
    else:
        funcs = {'OS':Systolic_OS, 'IS':Systolic_IS, 'WS':Systolic_WS}
    
    shapes = [(dims[0],(dims[1],dims[2])), (dims[1],(dims[0],dims[2])),(dims[2],(dims[1],dims[0])),
              ((dims[1],dims[2]), dims[0]), ((dims[0],dims[2]), dims[1]), ((dims[1],dims[0]), dims[2])]
    mesh = DeviceMesh(shapes[0], flops, bw, link_latency)
    comm_fw, overlap_fw = funcs['OS'](mesh, B, O, I, steps=steps)
    comm_bd, overlap_bd = funcs['IS'](mesh, B, I, O, steps=steps)
    comm_bw, overlap_bw = funcs['WS'](mesh, I, O, B, steps=steps, precisions=(16,16,32))
    bestcomm = (comm_fw + comm_bd + comm_bw)
    bestoverlap = (overlap_fw + overlap_bd + overlap_bw)
    
    bestshape = shapes[0]
    for s in shapes:
        mesh.reshape(s)
        comm_fw, overlap_fw = funcs['OS'](mesh, B, O, I, steps=steps)
        comm_bd, overlap_bd = funcs['IS'](mesh, B, I, O, steps=steps)
        comm_bw, overlap_bw = funcs['WS'](mesh, I, O, B, steps=steps, precisions=(16,16,32))
        comm = (comm_fw + comm_bd + comm_bw)
        overlap = (overlap_fw + overlap_bd + overlap_bw)
        if comm+overlap < bestcomm + bestoverlap:
            bestshape = s
            bestcomm = comm
            bestoverlap = overlap
        #print("Best mesh for {} is {}x{}, result: {}".format(
        #    func.__name__,meshshape[0],meshshape[1],tup2ms(func(mesh, M, N, K))))
        
        
    mesh.reshape(bestshape)
    if verbose:
        print("Best mesh for, {} ,is, {},{}, result:, {:.4f}, {:.4f}, ms".format(
        algorithm,bestshape[0],bestshape[1],bestcomm*1000, bestoverlap*1000))
    return bestshape

def AutotuneLayer3D2(dims, B, I, O, algorithm='SUMMA', flops=240*1024**4, bw=40*1024**3,link_latency=6e-6, verbose=True, steps=32):
    
    if algorithm == 'SUMMA':
        funcs = {'OS':SUMMA_OS, 'IS':SUMMA_IS, 'WS':SUMMA_WS}
    elif algorithm == 'GSPMD':
        funcs = {'OS':GSPMD_OS, 'IS':GSPMD_IS, 'WS':GSPMD_WS}
    else:
        funcs = {'OS':Systolic_OS, 'IS':Systolic_IS, 'WS':Systolic_WS}
    
    shapes = [(dims[0],dims[1]*dims[2]), (dims[1],dims[0]*dims[2]),(dims[2],dims[1]*dims[0]),
              ((dims[1]*dims[2]), dims[0]), ((dims[0]*dims[2]), dims[1]), ((dims[1]*dims[0]), dims[2])]
    mesh = DeviceMesh(shapes[0], flops, bw, link_latency)
    comm_fw, overlap_fw = funcs['OS'](mesh, B, O, I, steps=steps)
    comm_bd, overlap_bd = funcs['IS'](mesh, B, I, O, steps=steps)
    comm_bw, overlap_bw = funcs['WS'](mesh, I, O, B, steps=steps, precisions=(16,16,32))
    bestcomm = (comm_fw + comm_bd + comm_bw)
    bestoverlap = (overlap_fw + overlap_bd + overlap_bw)
    
    bestshape = shapes[0]
    for s in shapes:
        mesh.reshape(s)
        comm_fw, overlap_fw = funcs['OS'](mesh, B, O, I, steps=steps)
        comm_bd, overlap_bd = funcs['IS'](mesh, B, I, O, steps=steps)
        comm_bw, overlap_bw = funcs['WS'](mesh, I, O, B, steps=steps, precisions=(16,16,32))
        comm = (comm_fw + comm_bd + comm_bw)
        overlap = (overlap_fw + overlap_bd + overlap_bw)
        if comm+overlap < bestcomm + bestoverlap:
            bestshape = s
            bestcomm = comm
            bestoverlap = overlap
        #print("Best mesh for {} is {}x{}, result: {}".format(
        #    func.__name__,meshshape[0],meshshape[1],tup2ms(func(mesh, M, N, K))))
        
        
    mesh.reshape(bestshape)
    if verbose:
        print("Best mesh for, {} ,is, {},{}, result:, {:.4f}, {:.4f}, ms".format(
        algorithm,bestshape[0],bestshape[1],bestcomm*1000, bestoverlap*1000))
    return bestshape

def estimate1bitAdam(dim, link_latency, bw, data_shape: tuple, precision):
    reducescatter_size = data_shape[0]*data_shape[1]*precision//8
    allgather_size = data_shape[0]*data_shape[1]//8
    reducescatter = (reducescatter_size / (bw*2)) * (dim-1)
    allgather = allgather_size * (dim-1)/(bw*2)
    
    return reducescatter + allgather + link_latency*(2*dim - 1)


if __name__=='__main__':
    N=32
    flops=300*1024**4
    bw=44*1024**3
    #myMesh = DeviceMesh((N,N), flops, bw, link_latency=3e-6) 

    #myMesh = DeviceMesh(((16,4),8), flops, bw, link_latency=6e-6)
    myMesh = DeviceMesh((64,8), flops, bw, link_latency=6e-6) 
    
    print(SUMMA_OS(myMesh,4*8*16*2048, 4*96*128, 96*128))
    print(GSPMD_OS(myMesh,4*8*16*2048, 4*96*128, 96*128))
    print(Systolic_OS(myMesh,4*8*16*2048, 4*96*128, 96*128, steps=8))
    
    #MeshTune3D([4,8,16], Systolic_OS, 2*128*2048, 160*128, 4*160*128,flops, bw, verbose=True)
    #MeshTune3D([4,8,16], SUMMA_OS, 2*128*2048, 160*128, 4*160*128,flops, bw, verbose=True)
    #MeshTune3D([8,32,16], Systolic_OS, 2*1024*2048, 160*128, 4*160*128,flops, bw, verbose=True)
    #MeshTune3D([8,32,16], SUMMA_OS, 2*1024*2048, 160*128, 4*160*128,flops, bw, verbose=True)
    #MeshTune3D([4,8,8], GSPMD_OS, 2*64*2048, 160*128, 4*160*128,flops, bw, verbose=True)
    #MeshTune(myMesh, Systolic_OS, 2*256*2048, 160*128, 4*160*128,verbose=True, steps=16)
    #MeshTune(myMesh, Systolic_IS, 2*256*2048, 4*160*128, 160*128,verbose=True, steps=16)
    #MeshTune(myMesh, Systolic_WS, 160*128, 4*160*128, 2*256*2048, verbose=True, precisions=(16,16,32), steps=16)

    AutotuneLayer3D((8,16,8), 4*128*2048, 160*128, 4*160*128, flops=flops, algorithm='SUMMA')
    AutotuneLayer3D((8,16,8), 4*128*2048, 160*128, 4*160*128, flops=flops, algorithm='GSPMD')
    AutotuneLayer3D((8,16,8), 4*128*2048, 160*128, 4*160*128, flops=flops, algorithm='Ours', steps=8)

    AutotuneLayer3D((8,16,8), 4*128*2048, 160*128, 3*160*128, flops=flops, algorithm='SUMMA')
    AutotuneLayer3D((8,16,8), 4*128*2048, 160*128, 3*160*128, flops=flops, algorithm='GSPMD')
    AutotuneLayer3D((8,16,8), 4*128*2048, 160*128, 3*160*128, flops=flops, algorithm='Ours', steps=8)

    AutotuneLayer3D((8,16,8), 4*128*2048, 160*128, 160*128, flops=flops, algorithm='SUMMA')
    AutotuneLayer3D((8,16,8), 4*128*2048, 160*128, 160*128, flops=flops, algorithm='GSPMD')
    AutotuneLayer3D((8,16,8), 4*128*2048, 160*128, 160*128, flops=flops, algorithm='Ours', steps=8)

    #AutotuneLayer3D((8,32,4), 2*256*2048, 160*128, 4*160*128, algorithm='SUMMA')
    #AutotuneLayer3D((8,32,4), 2*256*2048, 160*128, 4*160*128, algorithm='GSPMD')
    #AutotuneLayer3D((8,32,4), 2*256*2048, 160*128, 4*160*128, algorithm='Ours', steps=16)

    #AutotuneLayer3D((4,32,16), 2*512*2048, 160*128, 4*160*128, algorithm='SUMMA')
    #AutotuneLayer3D((4,32,16), 2*512*2048, 160*128, 4*160*128, algorithm='GSPMD')
    #AutotuneLayer3D((4,32,16), 2*512*2048, 160*128, 4*160*128, algorithm='Ours', steps=16)

    AutotuneLayer3D((8,16,16), 1024*2048, 160*128, 4*160*128, flops=flops, algorithm='SUMMA')
    AutotuneLayer3D((8,16,16), 1024*2048, 160*128, 4*160*128, flops=flops, algorithm='GSPMD')
    AutotuneLayer3D((8,16,16), 1024*2048, 160*128, 4*160*128, flops=flops, algorithm='Ours', steps=8)

    #AutotuneLayer3D((8,32,32), 2*2048*2048, 160*128, 4*160*128, algorithm='SUMMA')
    #AutotuneLayer3D((8,32,32), 2*2048*2048, 160*128, 4*160*128, algorithm='GSPMD')
    #AutotuneLayer3D((8,32,32), 2*2048*2048, 160*128, 4*160*128, algorithm='Ours', steps=16)

    #AutotuneLayer3D((8,32,64), 2*4096*2048, 160*128, 4*160*128, algorithm='SUMMA')
    #AutotuneLayer3D((8,32,64), 2*4096*2048, 160*128, 4*160*128, algorithm='GSPMD')
    #AutotuneLayer3D((8,32,64), 2*4096*2048, 160*128, 4*160*128, algorithm='Ours', steps=16)

    #AutotuneLayer3D((8,64,64), 2*8192*2048, 160*128, 4*160*128, algorithm='SUMMA')
    #AutotuneLayer3D((8,64,64), 2*8192*2048, 160*128, 4*160*128, algorithm='GSPMD')
    #AutotuneLayer3D((8,64,64), 2*8192*2048, 160*128, 4*160*128, algorithm='Ours', steps=16)

    #AutotuneLayer3D((8,128,64), 2*4096*4*2048, 160*128, 4*160*128, algorithm='SUMMA')
    #AutotuneLayer3D((8,128,64), 2*4096*4*2048, 160*128, 4*160*128, algorithm='GSPMD')
    #AutotuneLayer3D((8,128,64), 2*4096*4*2048, 160*128, 4*160*128, algorithm='Ours', steps=8)
    
    #Systolic_OS(DeviceMesh(((32,64),8), flops, bw, link_latency=3e-6), 2*4096*2048, 160*128, 4*160*128)
    #SUMMA_OS(DeviceMesh(((32,64),8), flops, bw, link_latency=3e-6), 2*4096*2048, 160*128, 4*160*128)
    #weight_shape = (160*128//16//32, 4*160*128)
    #print(estimate1bitAdam(256, 15e-6, bw=100*1024*1024*1024//8, data_shape=weight_shape, precision=8)*1000)

    '''
    weight_shape = (192*128//32//32, 4*192*128)
    AutotuneLayer3D((32,32,4), 8*32*16*2048, 192*128, 4*192*128, 
                    flops=989*10**12, bw=75*10**9, link_latency=5e-6, algorithm='GSPMD', steps=16)
    AutotuneLayer3D((4,32,32), 8*32*16*2048, 192*128, 4*192*128, 
                    flops=989*10**12, bw=75*10**9, link_latency=5e-6, algorithm='Ours', steps=16)
    print(estimate1bitAdam(256, 15e-6, bw=100*1024*1024*1024//8, data_shape=weight_shape, precision=8)*1000)

    AutotuneLayer3D((4,16,8), 4*8*16*2048, 192*128, 192*128, 
                    flops=989*10**12, bw=75*10**9, link_latency=5e-6, algorithm='GSPMD', steps=16)
    AutotuneLayer3D((4,32,32), 8*32*16*2048, 192*128, 192*128, 
                    flops=989*10**12, bw=75*10**9, link_latency=5e-6, algorithm='Ours', steps=16)
    #myMesh2 = DeviceMesh((512,8), 989*10**12, 75*10**9, link_latency=5e-6) 
    #MeshTune(myMesh2, GSPMD_OS, 16*8*32*2048, 192*128, 4*192*128, verbose=True)

    weight_shape = (192*128//32//64, 4*192*128)
    print(estimate1bitAdam(512, 15e-6, bw=100*1024*1024*1024//8, data_shape=weight_shape, precision=8)*1000)

    AutotuneLayer3D((4,16,8), 4*8*16*2048, 192*128, 192*128, 
                    flops=272*10**12, bw=50*10**9, link_latency=5e-6, algorithm='GSPMD', steps=16)
    AutotuneLayer3D((4,32,64), 4*32*64*2048, 192*128, 192*128, 
                    flops=272*10**12, bw=50*10**9, link_latency=5e-6, algorithm='Ours', steps=16)
    
    '''
    #AutotuneLayer3D((4,16,8), 4*8*16*2048, 192*128, 192*128, 
    #                flops=300*10**12, bw=44*10**9, link_latency=3e-6, algorithm='GSPMD', steps=8)
    