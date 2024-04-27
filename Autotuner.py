import os
from functools import partial

import jax
import jax.numpy as jnp
import time
import CostModel

latencies = {'allgather':2e-5, 'reducescatter':2e-5, 'sendrecv':2e-5}
bws = {'allgather':33.224e9, 'reducescatter':31.73e9, 'sendrecv':36.3e9}
base_overheads = {'allgather':5e-5, 'reducescatter':5e-5, 'sendrecv':5e-5}

class FFLayerModel:
    def __init__(self, B,I,O):
        self.bsize = B
        self.in_dim = I
        self.out_dim = O
        self.transpose = False
        if O>=I and B>=I:
            self.dataflow = 'os'
        elif I>=O and B>=O:
            self.dataflow = 'is'
        else: #weight is the biggest
            self.dataflow = 'ws'
    def totalTrafficTime(self, mesh:CostModel.DeviceMesh):
        
        if self.dataflow == 'os':
            input_traffic = 2*self.bsize*self.in_dim*mesh.shape[1]
            weight_traffic = 2*self.in_dim*self.out_dim*mesh.shape[0]
            fwtime = max(input_traffic/mesh.bws['allgather'], weight_traffic/mesh.bws['allgather'])
            bdtime = max(input_traffic/mesh.bws['reducescatter'], weight_traffic/mesh.bws['allgather'])
            bwtime = max(input_traffic/mesh.bws['allgather'], weight_traffic/mesh.bws['reducescatter'])
            return (fwtime + bdtime + bwtime)/(mesh.shape[0]*mesh.shape[1])
        elif self.dataflow == 'is':
            output_traffic = 2*self.bsize*self.out_dim*mesh.shape[1]
            weight_traffic = 2*self.in_dim*self.out_dim*mesh.shape[0]
            fwtime = max(weight_traffic/mesh.bws['allgather'], output_traffic/mesh.bws['reducescatter'])
            bdtime = max(weight_traffic/mesh.bws['allgather'], output_traffic/mesh.bws['allgather'])
            bwtime = max(weight_traffic/mesh.bws['reducescatter'], output_traffic/mesh.bws['allgather'])
            return (fwtime + bdtime + bwtime)/(mesh.shape[0]*mesh.shape[1])
        else: #ws
            input_traffic = 2*self.bsize*self.in_dim*mesh.shape[1]
            output_traffic = 2*self.bsize*self.out_dim*mesh.shape[0]
            fwtime = max(input_traffic/mesh.bws['allgather'], output_traffic/mesh.bws['reducescatter'])
            bdtime = max(input_traffic/mesh.bws['reducescatter'], output_traffic/mesh.bws['allgather'])
            bwtime = max(input_traffic/mesh.bws['allgather'], output_traffic/mesh.bws['allgather'])
            return (fwtime + bdtime + bwtime)/(mesh.shape[0]*mesh.shape[1])
    def totalLatency(self, mesh:CostModel.DeviceMesh, K):
        
        if self.dataflow == 'os':
            dist_i = mesh.shape[1]
            dist_w = mesh.shape[0]
            fwtime = K*max(dist_i*mesh.link_latencies['allgather'] + mesh.base_overheads['allgather'], 
                           dist_w*mesh.link_latencies['allgather'] + mesh.base_overheads['allgather'])
            bdtime = K*max(dist_i*mesh.link_latencies['reducescatter'] + mesh.base_overheads['reducescatter'],
                           dist_w*mesh.link_latencies['allgather'] + mesh.base_overheads['allgather'])
            bwtime = K*max(dist_i*mesh.link_latencies['allgather'] + mesh.base_overheads['allgather'],
                           dist_w*mesh.link_latencies['reducescatter'] + mesh.base_overheads['reducescatter'])
            return (fwtime + bdtime + bwtime)
        elif self.dataflow == 'is':
            dist_o = mesh.shape[1]
            dist_w = mesh.shape[0]
            fwtime = K*max(dist_w*mesh.link_latencies['allgather'] + mesh.base_overheads['allgather'], 
                           dist_o*mesh.link_latencies['reducescatter'] + mesh.base_overheads['reducescatter'])
            bdtime = K*max(dist_o*mesh.link_latencies['allgather'] + mesh.base_overheads['allgather'],
                           dist_w*mesh.link_latencies['allgather'] + mesh.base_overheads['allgather'])
            bwtime = K*max(dist_o*mesh.link_latencies['allgather'] + mesh.base_overheads['allgather'],
                           dist_w*mesh.link_latencies['reducescatter'] + mesh.base_overheads['reducescatter'])
            return (fwtime + bdtime + bwtime)
        else: #ws
            dist_i = mesh.shape[1]
            dist_o = mesh.shape[0]
            fwtime = K*max(dist_i*mesh.link_latencies['allgather'] + mesh.base_overheads['allgather'], 
                           dist_o*mesh.link_latencies['reducescatter'] + mesh.base_overheads['reducescatter'])
            bdtime = K*max(dist_o*mesh.link_latencies['allgather'] + mesh.base_overheads['allgather'],
                           dist_i*mesh.link_latencies['reducescatter'] + mesh.base_overheads['reducescatter'])
            bwtime = K*max(dist_o*mesh.link_latencies['allgather'] + mesh.base_overheads['allgather'],
                           dist_i*mesh.link_latencies['allgather'] + mesh.base_overheads['allgather'])
            return (fwtime + bdtime + bwtime)
    def pipelineOverhead(self, mesh:CostModel.DeviceMesh, K):
        
        if self.dataflow == 'os':
            input_traffic = 2*self.bsize*self.in_dim*mesh.shape[1]
            weight_traffic = 2*self.in_dim*self.out_dim*mesh.shape[0]
            fwtime = max(input_traffic/mesh.bws['allgather'], weight_traffic/mesh.bws['allgather'])
            bdtime = (input_traffic/mesh.bws['reducescatter']+ weight_traffic/mesh.bws['allgather'])
            bwtime = (input_traffic/mesh.bws['allgather']+ weight_traffic/mesh.bws['reducescatter'])
            return (fwtime + bdtime + bwtime)/(mesh.shape[0]*mesh.shape[1])/K
        elif self.dataflow == 'is':
            output_traffic = 2*self.bsize*self.out_dim*mesh.shape[1]
            weight_traffic = 2*self.in_dim*self.out_dim*mesh.shape[0]
            fwtime = (weight_traffic/mesh.bws['allgather'] + output_traffic/mesh.bws['reducescatter'])
            bdtime = max(weight_traffic/mesh.bws['allgather'], output_traffic/mesh.bws['allgather'])
            bwtime = (weight_traffic/mesh.bws['reducescatter'] + output_traffic/mesh.bws['allgather'])
            return (fwtime + bdtime + bwtime)/(mesh.shape[0]*mesh.shape[1])/K
        else: #ws
            input_traffic = 2*self.bsize*self.in_dim*mesh.shape[1]
            output_traffic = 2*self.bsize*self.out_dim*mesh.shape[0]
            fwtime = (input_traffic/mesh.bws['allgather'] + output_traffic/mesh.bws['reducescatter'])
            bdtime = (input_traffic/mesh.bws['reducescatter'] + output_traffic/mesh.bws['allgather'])
            bwtime = max(input_traffic/mesh.bws['allgather'], output_traffic/mesh.bws['allgather'])
            return (fwtime + bdtime + bwtime)/(mesh.shape[0]*mesh.shape[1])/K
    def emulate(self, mesh:CostModel.DeviceMesh, K):
        
        if self.dataflow == 'os':
            fwtime = CostModel.Systolic_OS(mesh, self.bsize, self.out_dim, self.in_dim, K)
            bdtime = CostModel.Systolic_IS(mesh, self.bsize, self.in_dim, self.out_dim, K)
            bwtime = CostModel.Systolic_WS(mesh, )
        elif self.dataflow == 'is':
            output_traffic = 2*self.bsize*self.out_dim*mesh.shape[1]
            weight_traffic = 2*self.in_dim*self.out_dim*mesh.shape[0]
            fwtime = (weight_traffic/mesh.bws['allgather'] + output_traffic/mesh.bws['reducescatter'])
            bdtime = max(weight_traffic/mesh.bws['allgather'], output_traffic/mesh.bws['allgather'])
            bwtime = (weight_traffic/mesh.bws['reducescatter'] + output_traffic/mesh.bws['allgather'])
            return (fwtime + bdtime + bwtime)/(mesh.shape[0]*mesh.shape[1])/K
        else: #ws
            input_traffic = 2*self.bsize*self.in_dim*mesh.shape[1]
            output_traffic = 2*self.bsize*self.out_dim*mesh.shape[0]
            fwtime = (input_traffic/mesh.bws['allgather'] + output_traffic/mesh.bws['reducescatter'])
            bdtime = (input_traffic/mesh.bws['reducescatter'] + output_traffic/mesh.bws['allgather'])
            bwtime = max(input_traffic/mesh.bws['allgather'], output_traffic/mesh.bws['allgather'])
            return (fwtime + bdtime + bwtime)/(mesh.shape[0]*mesh.shape[1])/K
        return fwtime + bdtime + bwtime