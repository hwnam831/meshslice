import numpy as np
import torch
import torch.nn as nn

import time
import math

MAXFLOPS=3e14
MAXBW = 8.8e10

broadcastBWtimeus = {
    12: 0.76,
    13: 1.69,
    14: 3.12,
    15: 3.37,
    16: 3.22,
    17: 4.67,
    18: 15.01,
    19: 18.52,
    20: 24.22,
    21: 36.23,
    22: 60.34,
    23: 110.05,
    24: 207.45,
    25: 400.35,
    26: 778.35,
    27: 1534.55,
    28: 3046.65,
    29: 6069.45,
    30: 12114.35,
}

reduceBWTimeus = {
    12:0.97,
    13:2.02,
    14:3.29,
    15:3.42,
    16:3.49,
    17:17.31,
    18:13.39,
    19:16.72,
    20:22.8,
    21:34.94,
    22:59.28,
    23:110.05,
    24:213.05,
    25:409.75,
    26:790.85,
    27:1549.65,
    28:3069.75,
    29:6099.25,
}

allgatherBWtimeus = {
    12:2.32,
    13:3.43,
    14:3.37,
    15:1.57,
    16:3.16,
    17:5.96,
    18:22.36,
    19:26.63,
    20:35.24,
    21:52.49,
    22:87.85,
    23:160.25,
    24:304.55,
    25:587.85,
    26:1117.65,
    27:2145.85,
    28:4201.25,
    29:8327.25,
}

reducescatterBWtimeus = {
    12:2.39,
    13:3.63,
    14:3.76,
    15:4.12,
    16:5.98,
    17:9.05,
    18:20.55,
    19:24.97,
    20:34.57,
    21:51.94,
    22:86.96,
    23:157.25,
    24:298.45,
    25:582.05,
    26:1139.85,
    27:2216.35,
    28:4303.65,
    29:8464.75,
}

def estimateBWTime(bytes, algorithm):
    if algorithm == 'allgather':
        timetable = allgatherBWtimeus
    elif algorithm == 'reducescatter':
        timetable = reducescatterBWtimeus
    elif algorithm == 'reduce':
        timetable = reduceBWTimeus
    else:
        timetable = broadcastBWtimeus
    bitcount = math.floor(math.log2(bytes))
    if bitcount > 29:
        basetime = timetable[28]
        doubletime = timetable[29]
        basesize = 1<<28
        return (basetime + (bytes-basesize)*(doubletime - basetime) / basesize)*1e-6
    elif bitcount < 12:
        return timetable[12]*1e-6
    basetime = timetable[bitcount]
    doubletime = timetable[bitcount+1]
    basesize = 1<<bitcount
    ratio = (bytes - basesize) / basesize
    return (basetime + (doubletime - basetime) * ratio) * 1e-6

class Torus3D:
    def __init__(self, device_shape: tuple, bw, link_latency=3e-6, base_overhead=9e-6):
        self.shape = device_shape
        self.bw = bw
        if type(link_latency) is tuple:
            assert len(link_latency) == len(device_shape)
            self.link_latency = link_latency
        else:
            self.link_latency = (link_latency, link_latency, link_latency)
        self.base_overhead = base_overhead
        self.permutation = ((0,1),2)

    def meshsize(self,dim):
        mydim = self.permutation[dim]
        if type(mydim) is tuple:
            return self.shape[mydim[0]]*self.shape[mydim[1]]
        else:
            return self.shape[mydim]
    def estimateCollective(self, datasize, dim, algorithm='broadcast'):
        mydim = self.permutation[dim]
        if type(mydim) is tuple:
            coldim = mydim[0]
            rowdim = mydim[1]
            colsize = self.shape[coldim]
            rowsize = self.shape[rowdim]
            halfsize = datasize // 2
            if algorithm in ['allgather', 'reducescatter', 'allreduce']:
                phase1col = estimateBWTime(halfsize*(colsize-1), algorithm) + \
                            self.link_latency[coldim] * (colsize-1)
                phase1row = estimateBWTime(halfsize*(rowsize-1), algorithm) + \
                            self.link_latency[rowdim] * (rowsize-1)
                phase2col = estimateBWTime(halfsize*rowsize*(colsize-1), algorithm) + \
                            self.link_latency[coldim] * (colsize-1)
                phase2row = estimateBWTime(halfsize*colsize*(rowsize-1), algorithm) + \
                            self.link_latency[rowdim] * (rowsize-1)
                return self.base_overhead*2 + max(phase1col+phase2row, phase1row+phase2col)
            else: #bcast or reduce
                return self.base_overhead + estimateBWTime(halfsize, algorithm) + \
                            self.link_latency[coldim] * (colsize-1) + self.link_latency[rowdim] * (rowsize-1)
                
        else:
            mysize = self.shape[mydim]
            if algorithm in ['allgather', 'reducescatter', 'allreduce']:
                traffic = datasize*(mysize-1)
            else:
                traffic = datasize
            return self.base_overhead + estimateBWTime(traffic, algorithm) + self.link_latency[mydim] * (mysize-1)

def estimateMatmul(M, N, K, input_precision=torch.float16, warmup=10, repeat=20):
    #flop_count = M*N*K*2
    #return flop_count/mesh.flops
    
    A = torch.zeros(M,K).to(device='cuda',dtype=input_precision)
    B = torch.zeros(K,N).to(device='cuda',dtype=input_precision)
    #C = torch.zeros(M,N).to(device='cuda',dtype=output_precision)
    C = torch.mm(A,B)
    for _ in range(warmup):
        torch.mm(A,B, out=C)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    for _ in range(repeat):
        torch.mm(A,B, out=C)
        #torch.cuda.synchronize()

    end_event.record()
    torch.cuda.synchronize()  # Wait for the events to be recorded!
    elapsed_time_ms = start_event.elapsed_time(end_event)
    
    return (elapsed_time_ms)/repeat/1000

def estimateAttention(S, B, H, D, repeat=10):
    
    Q = torch.randn(B, H, S, D).to(device='cuda',dtype=torch.float16)
    K = torch.randn(B, H, S, D).to(device='cuda',dtype=torch.float16)
    V = torch.randn(B, H, S, D).to(device='cuda',dtype=torch.float16)
    nn.functional.scaled_dot_product_attention(Q,K,V)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    for _ in range(repeat):
        out = nn.functional.scaled_dot_product_attention(Q,K,V)
        #torch.cuda.synchronize()

    end_event.record()
    torch.cuda.synchronize()  # Wait for the events to be recorded!
    elapsed_time_ms = start_event.elapsed_time(end_event)
    compute = (elapsed_time_ms)/repeat/1000
    return {
        'prologue':0.0,
        'gputime': compute,
        'overlap': compute,
        'epilogue': 0.0
    }

def estimateAddNorm(S, B, D, repeat=10):
    #flop_count = M*N*K*2
    #return flop_count/mesh.flops
    
    X1 = torch.randn(B, S, D).to(device='cuda',dtype=torch.float16)
    X2 = torch.randn(B, S, D).to(device='cuda',dtype=torch.float16)
    nn.functional.layer_norm(X1+X2, (D,))
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    for _ in range(repeat):
        nn.functional.layer_norm(X1+X2, (D,))
        #torch.cuda.synchronize()

    end_event.record()
    torch.cuda.synchronize()  # Wait for the events to be recorded!
    elapsed_time_ms = start_event.elapsed_time(end_event)
    
    compute = (elapsed_time_ms)/repeat/1000
    return {
        'prologue':0.0,
        'gputime': compute,
        'overlap': compute,
        'epilogue': 0.0
    }

def bytecount(shape, precision=16):
    return shape[0]*shape[1]*precision//8

def SUMMA_OS(mesh:Torus3D, M, N, K, steps=8, precisions=(16,16,16)):
    steps = np.lcm(mesh.meshsize(0), mesh.meshsize(1))
    ishape = (M//mesh.meshsize(0), K//steps)
    wshape = (K//steps, N//mesh.meshsize(1))
    broadcast_i = mesh.estimateCollective(bytecount(ishape, precisions[0]), 1, 'broadcast')
    broadcast_w = mesh.estimateCollective(bytecount(wshape, precisions[1]), 0, 'broadcast')
    compute = estimateMatmul(ishape[0], wshape[1], ishape[1])

    return {
        'prologue':max(broadcast_i, broadcast_w),
        'gputime': compute*steps,
        'overlap': max(broadcast_i, broadcast_w, compute) * (steps-1) + compute,
        'epilogue': 0.0
    }


def SUMMA_IS(mesh:Torus3D, M, N, K, steps=8, precisions=(16,16,16)):
    steps = np.lcm(mesh.meshsize(0), mesh.meshsize(1))
    ishape = (M//mesh.meshsize(0), K//mesh.meshsize(1))
    oshape = (M//mesh.meshsize(0), N//steps)
    wshape = (N//steps, K//mesh.meshsize(1))

    broadcast_w = mesh.estimateCollective(bytecount(wshape, precisions[1]), 0, 'broadcast')
    reduce_o = mesh.estimateCollective(bytecount(oshape, precisions[2]), 1, 'reduce')
    compute = estimateMatmul(ishape[0], oshape[1], ishape[1])

    return {
        'prologue':broadcast_w,
        'gputime': compute*steps,
        'overlap': max(reduce_o, broadcast_w, compute) * (steps-1) + compute,
        'epilogue': reduce_o
    }


def SUMMA_WS(mesh:Torus3D, M, N, K, steps=8, precisions=(16,16,16)):
    steps = np.lcm(mesh.meshsize(0), mesh.meshsize(1))
    ishape = (K//mesh.meshsize(0), M//steps)
    oshape = (M//steps, N//mesh.meshsize(1))
    wshape = (K//mesh.meshsize(0), N//mesh.meshsize(1))

    broadcast_i = mesh.estimateCollective(bytecount(ishape, precisions[0]), 1, 'broadcast')
    reduce_o = mesh.estimateCollective(bytecount(oshape, precisions[2]), 0, 'reduce')
    compute = estimateMatmul(oshape[0], oshape[1], wshape[0])

    return {
        'prologue':broadcast_i,
        'gputime': compute*steps,
        'overlap': max(reduce_o, broadcast_i, compute) * (steps-1) + compute,
        'epilogue': reduce_o
    }




def GSPMD_OS(mesh: Torus3D, M, N, K, steps=8, precisions=(16,16,16)):
    ishape = (M//mesh.meshsize(0), K//(mesh.meshsize(1)))
    wshape = (K//(mesh.meshsize(0)), N//mesh.meshsize(1))
    allgather_i = mesh.estimateCollective(bytecount(ishape, precisions[0]), 1, 'allgather')
    allgather_w = mesh.estimateCollective(bytecount(wshape, precisions[1]), 0, 'allgather')
    compute = estimateMatmul(ishape[0], wshape[1], K)
    
    return {
        'prologue':max(allgather_i, allgather_w),
        'gputime': compute,
        'overlap': compute,
        'epilogue': 0.0
    }

def GSPMD_IS(mesh:Torus3D, M, N, K, steps=8, precisions=(16,16,16)):
    ishape = (M//mesh.meshsize(0), K//(mesh.meshsize(1)))
    wshape = (N//(mesh.meshsize(0)), K//mesh.meshsize(1))
    oshape = (M//(mesh.meshsize(0)), N//mesh.meshsize(1))
    
    allgather_w = mesh.estimateCollective(bytecount(wshape, precisions[1]), 0, 'allgather')

    compute = estimateMatmul(ishape[0], N, ishape[1])
    reducescatter_o = mesh.estimateCollective(bytecount(oshape, precisions[2]), 1, 'reducescatter')
    
    return {
        'prologue':allgather_w,
        'gputime': compute,
        'overlap': compute,
        'epilogue': reducescatter_o
    }

def GSPMD_WS(mesh:Torus3D, M, N, K, steps=8, precisions=(16,16,16)):
    ishape = (K//mesh.meshsize(0), M//(mesh.meshsize(1)))
    wshape = (K//(mesh.meshsize(0)), N//mesh.meshsize(1))
    oshape = (M//(mesh.meshsize(0)), N//mesh.meshsize(1))
    
    allgather_i = mesh.estimateCollective(bytecount(ishape, precisions[0]), 1, 'allgather')

    compute = estimateMatmul(M, wshape[1], ishape[0])
    reducescatter_o = mesh.estimateCollective(bytecount(oshape, precisions[2]), 0, 'reducescatter')

    return {
        'prologue':allgather_i,
        'gputime': compute,
        'overlap': compute,
        'epilogue': reducescatter_o
    }
def roofline_collective(mesh: Torus3D, bytesize, dim, algorithm='allgather'):
    mydim = mesh.permutation[dim]
    if type(mydim) is tuple:
        coldim = mydim[0]
        rowdim = mydim[1]
        colsize = mesh.shape[coldim]
        rowsize = mesh.shape[rowdim]
        halfsize = bytesize // 2
        if algorithm in ['allgather', 'reducescatter', 'allreduce']:
            return mesh.base_overhead + mesh.link_latency[rowdim]*(colsize + rowsize - 1) + halfsize*colsize*rowsize / MAXBW
        else: #bcast or reduce
            return mesh.base_overhead + mesh.link_latency[rowdim]*(colsize + rowsize - 1) + halfsize / MAXBW
            
    else:
        mysize = mesh.shape[mydim]
        if algorithm in ['allgather', 'reducescatter', 'allreduce']:
            traffic = bytesize*(mysize-1)
        else:
            traffic = bytesize
        return mesh.base_overhead + mesh.link_latency[mydim] * (mysize-1) + bytesize / MAXBW
    
def roofline_matmul(M,N,K):
    flop_count = M*N*K*2
    return flop_count/MAXFLOPS
def Roofline_OS(mesh: Torus3D, M, N, K, steps=8, precisions=(16,16,16)):
    ishape = (M//mesh.meshsize(0), K//(mesh.meshsize(1)))
    wshape = (K//(mesh.meshsize(0)), N//mesh.meshsize(1))
    allgather_i = roofline_collective(mesh, bytecount(ishape, precisions[0]), 1, 'allgather')
    allgather_w = roofline_collective(mesh, bytecount(wshape, precisions[1]), 0, 'allgather')
    compute = roofline_matmul(ishape[0], wshape[1], K)
    
    return {
        'prologue':max(allgather_i, allgather_w),
        'gputime': compute,
        'overlap': compute,
        'epilogue': 0.0
    }

def Roofline_IS(mesh:Torus3D, M, N, K, steps=8, precisions=(16,16,16)):
    ishape = (M//mesh.meshsize(0), K//(mesh.meshsize(1)))
    wshape = (N//(mesh.meshsize(0)), K//mesh.meshsize(1))
    oshape = (M//(mesh.meshsize(0)), N//mesh.meshsize(1))
    
    allgather_w = roofline_collective(mesh, bytecount(wshape, precisions[1]), 0, 'allgather')

    compute = roofline_matmul(ishape[0], N, ishape[1])
    reducescatter_o = roofline_collective(mesh, bytecount(oshape, precisions[2]), 1, 'reducescatter')
    
    return {
        'prologue':allgather_w,
        'gputime': compute,
        'overlap': compute,
        'epilogue': reducescatter_o
    }

def Roofline_WS(mesh:Torus3D, M, N, K, steps=8, precisions=(16,16,16)):
    ishape = (K//mesh.meshsize(0), M//(mesh.meshsize(1)))
    wshape = (K//(mesh.meshsize(0)), N//mesh.meshsize(1))
    oshape = (M//(mesh.meshsize(0)), N//mesh.meshsize(1))
    
    allgather_i = roofline_collective(mesh, bytecount(ishape, precisions[0]), 1, 'allgather')

    compute = roofline_matmul(M, wshape[1], ishape[0])
    reducescatter_o = roofline_collective(mesh, bytecount(oshape, precisions[2]), 0, 'reducescatter')

    return {
        'prologue':allgather_i,
        'gputime': compute,
        'overlap': compute,
        'epilogue': reducescatter_o
    }

def Systolic_OS(mesh:Torus3D, M, N, K, steps=8, precisions=(16,16,16)): #assume synchronized allgather

    ishape = (M//mesh.meshsize(0), K//mesh.meshsize(1)//steps)
    wshape = (K//mesh.meshsize(0)//steps, N//mesh.meshsize(1))
    allgather_i = mesh.estimateCollective(bytecount(ishape, precisions[0]), 1, 'allgather')
    allgather_w = mesh.estimateCollective(bytecount(wshape, precisions[1]), 0, 'allgather')
    compute = estimateMatmul(ishape[0], wshape[1], K//steps)
    
    return {
        'prologue':max(allgather_i, allgather_w),
        'gputime': compute*steps,
        'overlap': max(allgather_i, allgather_w, compute) * (steps-1) + compute,
        'epilogue': 0.0
    }
    

def Systolic_IS(mesh:Torus3D, M, N, K, steps=8, precisions=(16,16,16)): #assume synchronized allgather
    
    ishape = (M//mesh.meshsize(0), K//mesh.meshsize(1))
    wshape = (N//mesh.meshsize(0)//steps, K//mesh.meshsize(1))
    oshape = (M//mesh.meshsize(0), N//mesh.meshsize(1)//steps)

    allgather_w = mesh.estimateCollective(bytecount(wshape, precisions[1]), 0, 'allgather')

    reducescatter_o = mesh.estimateCollective(bytecount(oshape, precisions[2]), 1, 'reducescatter')
    compute = estimateMatmul(ishape[0], N//steps, ishape[1])
    
    return {
        'prologue':allgather_w,
        'gputime': compute*steps,
        'overlap': compute + max(allgather_w, reducescatter_o, compute) * (steps-1),
        'epilogue': reducescatter_o
    }
    
    
def Systolic_WS(mesh:Torus3D, M, N, K, steps=32, precisions=(16,16,16)): #assume synchronized allgather

    ishape = (K//mesh.meshsize(0), M//mesh.meshsize(1)//steps)
    wshape = (K//mesh.meshsize(0), N//mesh.meshsize(1))
    oshape = (M//mesh.meshsize(0)//steps, N//mesh.meshsize(1))
    allgather_i = mesh.estimateCollective(bytecount(ishape, precisions[0]), 1, 'allgather')

    compute = estimateMatmul(M//steps, oshape[1], ishape[0])
    reducescatter_o = mesh.estimateCollective(bytecount(oshape, precisions[2]), 0, 'reducescatter')

    return {
        'prologue':allgather_i,
        'gputime': compute*steps,
        'overlap': compute + max(allgather_i, reducescatter_o, compute) * (steps-1),
        'epilogue': reducescatter_o
    }
    
def addResult(total, result, overlap=None):
    for key in result:
        total[key] = total[key] + result[key]
    return total
def emulate2DForward(mesh: Torus3D, bsize, seqlen, nheads, headdim, algorithm='GSPMD', steps=8):

    if algorithm == 'SUMMA':
        funcs = {'OS':SUMMA_OS, 'IS':SUMMA_IS, 'WS':SUMMA_WS}
    elif algorithm == 'GSPMD':
        funcs = {'OS':GSPMD_OS, 'IS':GSPMD_IS, 'WS':GSPMD_WS}
    elif algorithm == 'Roofline':
        funcs = {'OS':Roofline_OS, 'IS':Roofline_IS, 'WS':Roofline_WS}
    else:
        funcs = {'OS':Systolic_OS, 'IS':Systolic_IS, 'WS':Systolic_WS}
    #H -> 3H
    colsize = mesh.meshsize(0)
    rowsize = mesh.meshsize(1)
    inproj = funcs['OS'](mesh, bsize*seqlen, 3*nheads*headdim, nheads*headdim,steps=steps)
    att = estimateAttention(seqlen, bsize//colsize, nheads//rowsize, headdim)
    outproj = funcs['OS'](mesh, bsize*seqlen, nheads*headdim, nheads*headdim,steps=steps)
    addnorm1 = estimateAddNorm(seqlen, bsize//colsize, nheads*headdim//rowsize)
    ff1 = funcs['OS'](mesh, bsize*seqlen, 4*nheads*headdim, nheads*headdim,steps=steps)
    ff2 = funcs['IS'](mesh, bsize*seqlen, nheads*headdim, 4*nheads*headdim,steps=steps)
    addnorm2 = estimateAddNorm(seqlen, bsize//colsize, nheads*headdim//rowsize)
    total = {'prologue':0, 'gputime': 0, 'overlap': 0, 'epilogue': 0}
    addResult(total, inproj)
    addResult(total, att)
    addResult(total, outproj)
    addResult(total, addnorm1)
    addResult(total, ff1)
    addResult(total, ff2)
    addResult(total, addnorm2)
    return total

def emulate2DBackward(mesh: Torus3D, bsize, seqlen, nheads, headdim, algorithm='GSPMD', steps=8):

    if algorithm == 'SUMMA':
        funcs = {'OS':SUMMA_OS, 'IS':SUMMA_IS, 'WS':SUMMA_WS}
    elif algorithm == 'GSPMD':
        funcs = {'OS':GSPMD_OS, 'IS':GSPMD_IS, 'WS':GSPMD_WS}
    elif algorithm == 'Roofline':
        funcs = {'OS':Roofline_OS, 'IS':Roofline_IS, 'WS':Roofline_WS}
    else:
        funcs = {'OS':Systolic_OS, 'IS':Systolic_IS, 'WS':Systolic_WS}
    #H -> 3H
    colsize = mesh.meshsize(0)
    rowsize = mesh.meshsize(1)
    addnorm2 = estimateAddNorm(seqlen, bsize//colsize, nheads*headdim//rowsize)
    ff2_bd = funcs['OS'](mesh, bsize*seqlen, 4*nheads*headdim, nheads*headdim,steps=steps)
    ff2_bw = funcs['WS'](mesh, nheads*headdim, 4*nheads*headdim, bsize*seqlen, steps=steps)


    ff1_bd = funcs['IS'](mesh, bsize*seqlen, nheads*headdim, 4*nheads*headdim, steps=steps)
    ff1_bw = funcs['WS'](mesh, nheads*headdim, 4*nheads*headdim, bsize*seqlen, steps=steps)

    addnorm1 = estimateAddNorm(seqlen, bsize//colsize, nheads*headdim//rowsize)

    outproj_bd = funcs['OS'](mesh, bsize*seqlen, nheads*headdim, nheads*headdim,steps=steps)
    outproj_bw = funcs['WS'](mesh, nheads*headdim, nheads*headdim, bsize*seqlen, steps=steps)

    att_b1 = estimateAttention(seqlen, bsize//colsize, nheads//rowsize, headdim)
    att_b2 = estimateAttention(seqlen, bsize//colsize, nheads//rowsize, headdim)

    inproj_bd = funcs['IS'](mesh, bsize*seqlen, nheads*headdim, 3*nheads*headdim, steps=steps)
    inproj_bw = funcs['WS'](mesh, nheads*headdim, 3*nheads*headdim, bsize*seqlen, steps=steps)
    
    
    
    

    
    total = {'prologue':0, 'gputime': 0, 'overlap': 0, 'epilogue': 0}
    addResult(total, inproj_bd)
    addResult(total, inproj_bw)
    addResult(total, att_b1)
    addResult(total, att_b2)
    addResult(total, outproj_bd)
    addResult(total, outproj_bw)
    addResult(total, addnorm1)
    addResult(total, ff1_bd)
    addResult(total, ff1_bw)
    addResult(total, ff2_bd)
    addResult(total, ff2_bw)
    addResult(total, addnorm2)
    return total

def emulateCluster(cluster:Torus3D, algorithm, seqlen, nheads, headdim, steps=8):
    bsize = cluster.meshsize(0)*cluster.meshsize(1)//2

    #fwd = emulate2DForward(cluster, bsize, seqlen, nheads, headdim, algorithm, steps=steps)

    #total = emulate2DBackward(cluster, bsize, seqlen, nheads, headdim, algorithm, steps=steps)

    fwd = emulate2DForward(cluster, bsize, seqlen, nheads, headdim, algorithm, steps=steps)

    total = emulate2DBackward(cluster, bsize, seqlen, nheads, headdim, algorithm, steps=steps)
    addResult(total, fwd)
    addResult(total, fwd)
    gputime = total['gputime']
    commtime = total['prologue'] + total['epilogue'] + total['overlap'] - gputime
    print("#nodes:,{}, ffdim:{}, algorithm {}, gputime(ms):,{}, commtime(ms):, {} ".format(cluster.meshsize(0)*cluster.meshsize(1), nheads*headdim ,algorithm, gputime*1000, commtime*1000))



if __name__ == '__main__':
    cluster = Torus3D((8,16,32), 44*1e9, (3e-6, 3e-6, 3e-6), 9e-6)
    cluster.permutation = ((2,1),0)
    dsize = 1024*96*128 * 2
    wsize = 96*128*96*128*4*2//(16*32*8)
    estimateBWTime(dsize, 'allgather')
    #print(cluster.estimateCollective(dsize, 1, 'allgather')*1000)
    #print(cluster.estimateCollective(wsize, 0, 'allgather')*1000)
    from torch.profiler import profile, record_function, ProfilerActivity
    #with profile(activities=[ProfilerActivity.CUDA],record_shapes=True, with_flops=True) as prof:
    #    myres = emulate2DForward(cluster, 1024, 2048, 96, 128, 'Ours')
    #    print(myres)
    #print(prof.key_averages().table(sort_by="cuda_time_total",row_limit=10))
    #myres = emulate2DForward(cluster, 1024, 2048, 96, 128, 'GSPMD', steps=8)
    #myres = emulate2DForward(cluster, 1024, 2048, 96, 128, 'Ours')
    #print(myres)
   
    #myres = emulate2DForward(cluster, 1024, 2048, 96, 128, 'GSPMD', steps=8)
    #print(myres)

    #myres = emulate2DBackward(cluster, 1024, 2048, 96, 128, 'GSPMD', steps=8)
    #myres = emulate2DBackward(cluster, 1024, 2048, 96, 128, 'Ours')
    #print(myres)
   
    #myres = emulate2DBackward(cluster, 1024, 2048, 96, 128, 'GSPMD', steps=8)
    #print(myres)
    cluster.permutation = ((2,0),1)
    emulateCluster(cluster, 'SUMMA', 2048, 96, 128)
    cluster.permutation = ((2,1),0)
    emulateCluster(cluster, 'Roofline', 2048, 96, 128)
    emulateCluster(cluster, 'GSPMD', 2048, 96, 128)
    emulateCluster(cluster, 'Ours', 2048, 96, 128, steps=8)
    

    cluster = Torus3D((8,16,16), 44*1e9, (3e-6, 3e-6, 3e-6), 9e-6)

    cluster.permutation = ((2,0),1)
    emulateCluster(cluster, 'SUMMA', 2048, 96, 128)
    cluster.permutation = ((2,1),0)
    emulateCluster(cluster, 'Roofline', 2048, 96, 128)
    emulateCluster(cluster, 'GSPMD', 2048, 96, 128)
    emulateCluster(cluster, 'Ours', 2048, 96, 128, steps=8)
    

    cluster = Torus3D((8,16,8), 44*1e9, (3e-6, 3e-6, 3e-6), 9e-6)

    cluster.permutation = ((2,0),1)
    emulateCluster(cluster, 'SUMMA', 2048, 96, 128)
    cluster.permutation = ((2,1),0)
    emulateCluster(cluster, 'Roofline', 2048, 96, 128)
    emulateCluster(cluster, 'GSPMD', 2048, 96, 128)
    emulateCluster(cluster, 'Ours', 2048, 96, 128, steps=8)
    

    cluster = Torus3D((8,16,4), 44*1e9, (3e-6, 3e-6, 3e-6), 9e-6)

    cluster.permutation = ((2,0),1)
    emulateCluster(cluster, 'SUMMA', 2048, 96, 128)
    cluster.permutation = ((2,1),0)
    emulateCluster(cluster, 'Roofline', 2048, 96, 128)
    emulateCluster(cluster, 'GSPMD', 2048, 96, 128)
    emulateCluster(cluster, 'Ours', 2048, 96, 128, steps=8)
    

    cluster = Torus3D((8,8,4), 44*1e9, (3e-6, 3e-6, 3e-6), 9e-6)
    cluster.permutation = ((2,1),0)

    emulateCluster(cluster, 'SUMMA', 2048, 96, 128)
    emulateCluster(cluster, 'Roofline', 2048, 96, 128)
    emulateCluster(cluster, 'GSPMD', 2048, 96, 128)
    emulateCluster(cluster, 'Ours', 2048, 96, 128, steps=8)
   



    cluster = Torus3D((8,16,32), 44*1e9, (3e-6, 3e-6, 3e-6), 9e-6)
    cluster.permutation = ((2,0),1)
    emulateCluster(cluster, 'SUMMA', 2048, 160, 128)

    cluster.permutation = ((2,1),0)

    emulateCluster(cluster, 'Roofline', 2048, 96, 128)
    emulateCluster(cluster, 'GSPMD', 2048, 160, 128)
    emulateCluster(cluster, 'Ours', 2048, 160, 128, steps=8)
    

    cluster = Torus3D((8,16,16), 44*1e9, (3e-6, 3e-6, 3e-6), 9e-6)
    cluster.permutation = ((2,0),1)
    emulateCluster(cluster, 'SUMMA', 2048, 160, 128)
    cluster.permutation = ((2,1),0)
    emulateCluster(cluster, 'Roofline', 2048, 96, 128)
    emulateCluster(cluster, 'GSPMD', 2048, 160, 128)
    emulateCluster(cluster, 'Ours', 2048, 160, 128, steps=8)
    

    cluster = Torus3D((8,16,8), 44*1e9, (3e-6, 3e-6, 3e-6), 9e-6)
    cluster.permutation = ((2,0),1)
    emulateCluster(cluster, 'SUMMA', 2048, 160, 128)
    cluster.permutation = ((2,1),0)
    emulateCluster(cluster, 'Roofline', 2048, 96, 128)
    emulateCluster(cluster, 'GSPMD', 2048, 160, 128)
    emulateCluster(cluster, 'Ours', 2048, 160, 128, steps=8)
    

    cluster = Torus3D((8,16,4), 44*1e9, (3e-6, 3e-6, 3e-6), 9e-6)
    cluster.permutation = ((2,0),1)
    emulateCluster(cluster, 'SUMMA', 2048, 160, 128)
    cluster.permutation = ((2,1),0)
    emulateCluster(cluster, 'Roofline', 2048, 96, 128)
    emulateCluster(cluster, 'GSPMD', 2048, 160, 128)
    emulateCluster(cluster, 'Ours', 2048, 160, 128, steps=8)
    

    cluster = Torus3D((8,8,4), 44*1e9, (3e-6, 3e-6, 3e-6), 9e-6)
    cluster.permutation = ((2,1),0)
    emulateCluster(cluster, 'SUMMA', 2048, 160, 128)
    emulateCluster(cluster, 'Roofline', 2048, 96, 128)
    emulateCluster(cluster, 'GSPMD', 2048, 160, 128)
    emulateCluster(cluster, 'Ours', 2048, 160, 128, steps=8)
    

