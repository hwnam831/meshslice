import numpy as np
import torch
import time
import math

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

    def estimateCollective(self, datasize, dim, algorithm='broadcast'):
        mydim = self.permutation[dim]
        if type(mydim) is tuple:
            coldim = mydim[0]
            rowdim = mydim[1]
            colsize = self.shape[coldim]
            rowsize = self.shape[rowdim]
            halfsize = datasize // 2
            if algorithm in ['allgather', 'reducescatter', 'allreduce']:
                phase1col = estimateBWTime(halfsize, algorithm)*(colsize-1) + \
                            self.link_latency[coldim] * (colsize-1)
                phase1row = estimateBWTime(halfsize, algorithm)*(rowsize-1) + \
                            self.link_latency[rowdim] * (rowsize-1)
                phase2col = estimateBWTime(halfsize*rowsize, algorithm)*(colsize-1) + \
                            self.link_latency[coldim] * (colsize-1)
                phase2row = estimateBWTime(halfsize*colsize, algorithm)*(rowsize-1) + \
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
    
'''
def emulateForward(mesh, bsize, dim):
    emulateFF(dim, dim*3)
    emulateAttention()
    emulateAddNorm()
    emulateFF1()
    emulateFF2()
    emulateAddNorm()
'''


if __name__ == '__main__':
    cluster = Torus3D((8,16,32), 44*1e9, (3e-6, 6e-6, 9e-6), 9e-6)
    cluster.permutation = ((1,2),0)
    dsize = 1024*96*128 * 2
    wsize = 96*128*96*128*4*2//(16*32*8)
    estimateBWTime(dsize, 'allgather')
    print(cluster.estimateCollective(dsize, 1, 'allgather')*1000)
    print(cluster.estimateCollective(wsize, 0, 'allgather')*1000)


