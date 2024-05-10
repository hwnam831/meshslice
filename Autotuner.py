import os
from functools import partial

import jax
import jax.numpy as jnp
import time
import CostModel
import copy
import math
from CostModel import DeviceMesh

#data from GPU
#latencies = {'allgather':2e-5, 'reducescatter':2e-5, 'sendrecv':2e-5}
#bws = {'allgather':33.224e9, 'reducescatter':31.73e9, 'sendrecv':36.3e9}
#base_overheads = {'allgather':5e-5, 'reducescatter':5e-5, 'sendrecv':5e-5}

latencies = {'allgather':1e-5, 'reducescatter':1e-5, 'sendrecv':1e-5}
bws = {'allgather':43.224e9, 'reducescatter':41.73e9, 'sendrecv':46.3e9}
base_overheads = {'allgather':2e-5, 'reducescatter':2e-5, 'sendrecv':2e-5}

class FFLayerModel:
    def __init__(self, B,I,O, dataflow=None, transpose=False):
        self.bsize = B
        self.in_dim = I
        self.out_dim = O
        self.transpose = False
        if dataflow in ['os','ws','is']:
            self.dataflow=dataflow
        elif O>=I and B>=I:
            self.dataflow = 'os'
        elif I>=O and B>=O:
            self.dataflow = 'is'
        else: #weight is the biggest
            self.dataflow = 'ws'
        self.transpose = transpose
    def totalTrafficTime(self, mesh:DeviceMesh):
        
        if self.dataflow == 'os':
            input_traffic = 2*self.bsize*self.in_dim*(mesh.shape[1]-1)
            weight_traffic = 2*self.in_dim*self.out_dim*(mesh.shape[0]-1)
            if self.transpose:
                input_traffic = 2*self.bsize*self.in_dim*(mesh.shape[0]-1)
                weight_traffic = 2*self.in_dim*self.out_dim*(mesh.shape[1]-1)
            fwtime = max(input_traffic/mesh.bws['allgather'], weight_traffic/mesh.bws['allgather'])
            bdtime = max(input_traffic/mesh.bws['reducescatter'], weight_traffic/mesh.bws['allgather'])
            bwtime = max(input_traffic/mesh.bws['allgather'], weight_traffic/mesh.bws['reducescatter'])
            return (fwtime + bdtime + bwtime)/(mesh.shape[0]*mesh.shape[1])/2 #bidirectional
        elif self.dataflow == 'is':
            output_traffic = 2*self.bsize*self.out_dim*(mesh.shape[1]-1)
            weight_traffic = 2*self.in_dim*self.out_dim*(mesh.shape[0]-1)
            if self.transpose:
                output_traffic = 2*self.bsize*self.out_dim*(mesh.shape[0]-1)
                weight_traffic = 2*self.in_dim*self.out_dim*(mesh.shape[1]-1)
            fwtime = max(weight_traffic/mesh.bws['allgather'], output_traffic/mesh.bws['reducescatter'])
            bdtime = max(weight_traffic/mesh.bws['allgather'], output_traffic/mesh.bws['allgather'])
            bwtime = max(weight_traffic/mesh.bws['reducescatter'], output_traffic/mesh.bws['allgather'])
            return (fwtime + bdtime + bwtime)/(mesh.shape[0]*mesh.shape[1])/2 #bidirectional
        else: #ws
            input_traffic = 2*self.bsize*self.in_dim*(mesh.shape[1]-1)
            output_traffic = 2*self.bsize*self.out_dim*(mesh.shape[0]-1)
            if self.transpose:
                input_traffic = 2*self.bsize*self.in_dim*(mesh.shape[0]-1)
                output_traffic = 2*self.bsize*self.out_dim*(mesh.shape[1]-1)
            fwtime = max(input_traffic/mesh.bws['allgather'], output_traffic/mesh.bws['reducescatter'])
            bdtime = max(input_traffic/mesh.bws['reducescatter'], output_traffic/mesh.bws['allgather'])
            bwtime = max(input_traffic/mesh.bws['allgather'], output_traffic/mesh.bws['allgather'])
            return (fwtime + bdtime + bwtime)/(mesh.shape[0]*mesh.shape[1])/2 #bidirectional
    def totalLatency(self, mesh:DeviceMesh, K):
        
        if self.dataflow == 'os':
            dist_i = mesh.shape[1]-1
            dist_w = (mesh.shape[0]-1)
            if self.transpose:
                dist_i = (mesh.shape[0]-1)
                dist_w = mesh.shape[1]-1
            fwtime = K*max(dist_i*mesh.link_latencies['allgather'] + mesh.base_overheads['allgather'], 
                           dist_w*mesh.link_latencies['allgather'] + mesh.base_overheads['allgather'])
            bdtime = K*max(dist_i*mesh.link_latencies['reducescatter'] + mesh.base_overheads['reducescatter'],
                           dist_w*mesh.link_latencies['allgather'] + mesh.base_overheads['allgather'])
            bwtime = K*max(dist_i*mesh.link_latencies['allgather'] + mesh.base_overheads['allgather'],
                           dist_w*mesh.link_latencies['reducescatter'] + mesh.base_overheads['reducescatter'])
            return (fwtime + bdtime + bwtime)
        elif self.dataflow == 'is':
            dist_o = mesh.shape[1]-1
            dist_w = mesh.shape[0]-1
            if self.transpose:
                dist_o = mesh.shape[0]-1
                dist_w = mesh.shape[1]-1
            fwtime = K*max(dist_w*mesh.link_latencies['allgather'] + mesh.base_overheads['allgather'], 
                           dist_o*mesh.link_latencies['reducescatter'] + mesh.base_overheads['reducescatter'])
            bdtime = K*max(dist_o*mesh.link_latencies['allgather'] + mesh.base_overheads['allgather'],
                           dist_w*mesh.link_latencies['allgather'] + mesh.base_overheads['allgather'])
            bwtime = K*max(dist_o*mesh.link_latencies['allgather'] + mesh.base_overheads['allgather'],
                           dist_w*mesh.link_latencies['reducescatter'] + mesh.base_overheads['reducescatter'])
            return (fwtime + bdtime + bwtime)
        else: #ws
            dist_i = mesh.shape[1]-1
            dist_o = mesh.shape[0]-1
            if self.transpose:
                dist_i = mesh.shape[0]-1
                dist_o = mesh.shape[1]-1
            fwtime = K*max(dist_i*mesh.link_latencies['allgather'] + mesh.base_overheads['allgather'], 
                           dist_o*mesh.link_latencies['reducescatter'] + mesh.base_overheads['reducescatter'])
            bdtime = K*max(dist_o*mesh.link_latencies['allgather'] + mesh.base_overheads['allgather'],
                           dist_i*mesh.link_latencies['reducescatter'] + mesh.base_overheads['reducescatter'])
            bwtime = K*max(dist_o*mesh.link_latencies['allgather'] + mesh.base_overheads['allgather'],
                           dist_i*mesh.link_latencies['allgather'] + mesh.base_overheads['allgather'])
            return (fwtime + bdtime + bwtime)
    def pipelineOverhead(self, mesh:DeviceMesh, K):
        
        if self.dataflow == 'os':
            input_traffic = 2*self.bsize*self.in_dim*(mesh.shape[1]-1)
            weight_traffic = 2*self.in_dim*self.out_dim*(mesh.shape[0]-1)
            if self.transpose:
                input_traffic = 2*self.bsize*self.in_dim*(mesh.shape[0]-1)
                weight_traffic = 2*self.in_dim*self.out_dim*(mesh.shape[1]-1)
            fwtime = max(input_traffic/mesh.bws['allgather'], weight_traffic/mesh.bws['allgather'])
            bdtime = (input_traffic/mesh.bws['reducescatter']+ weight_traffic/mesh.bws['allgather'])
            bwtime = (input_traffic/mesh.bws['allgather']+ weight_traffic/mesh.bws['reducescatter'])
            return (fwtime + bdtime + bwtime)/(mesh.shape[0]*mesh.shape[1])/K/2 #bidirectional
        elif self.dataflow == 'is':
            output_traffic = 2*self.bsize*self.out_dim*(mesh.shape[1]-1)
            weight_traffic = 2*self.in_dim*self.out_dim*(mesh.shape[0]-1)
            if self.transpose:
                output_traffic = 2*self.bsize*self.out_dim*(mesh.shape[0]-1)
                weight_traffic = 2*self.in_dim*self.out_dim*(mesh.shape[1]-1)
            fwtime = (weight_traffic/mesh.bws['allgather'] + output_traffic/mesh.bws['reducescatter'])
            bdtime = max(weight_traffic/mesh.bws['allgather'], output_traffic/mesh.bws['allgather'])
            bwtime = (weight_traffic/mesh.bws['reducescatter'] + output_traffic/mesh.bws['allgather'])
            return (fwtime + bdtime + bwtime)/(mesh.shape[0]*mesh.shape[1])/K/2 #bidirectional
        else: #ws
            input_traffic = 2*self.bsize*self.in_dim*(mesh.shape[1]-1)
            output_traffic = 2*self.bsize*self.out_dim*(mesh.shape[0]-1)
            if self.transpose:
                input_traffic = 2*self.bsize*self.in_dim*(mesh.shape[0]-1)
                output_traffic = 2*self.bsize*self.out_dim*(mesh.shape[1]-1)
            fwtime = (input_traffic/mesh.bws['allgather'] + output_traffic/mesh.bws['reducescatter'])
            bdtime = (input_traffic/mesh.bws['reducescatter'] + output_traffic/mesh.bws['allgather'])
            bwtime = max(input_traffic/mesh.bws['allgather'], output_traffic/mesh.bws['allgather'])
            return (fwtime + bdtime + bwtime)/(mesh.shape[0]*mesh.shape[1])/K/2 #bidirectional
    def emulate(self, mesh:DeviceMesh, K):
        if not self.transpose:
            if self.dataflow == 'os':
                fwtime = CostModel.Systolic_OS(mesh, self.bsize, self.out_dim, self.in_dim, K)
                bdtime = CostModel.Systolic_IS(mesh, self.bsize, self.in_dim, self.out_dim, K)
                bwtime = CostModel.Systolic_WS(mesh, self.in_dim, self.out_dim, self.bsize, K)
            elif self.dataflow == 'is':
                fwtime = CostModel.Systolic_IS(mesh, self.bsize, self.out_dim, self.in_dim, K)
                bdtime = CostModel.Systolic_OS(mesh, self.bsize, self.in_dim, self.out_dim, K)
                bwtime = CostModel.Systolic_WS(mesh, self.out_dim, self.in_dim, self.bsize, K)
            else: #ws
                fwtime = CostModel.Systolic_WS(mesh, self.bsize, self.out_dim, self.in_dim, K)
                bdtime = CostModel.Systolic_IS(mesh, self.in_dim, self.bsize, self.out_dim, K)
                bwtime = CostModel.Systolic_OS(mesh, self.in_dim, self.out_dim, self.bsize, K)
        else: #transposed versions
            if self.dataflow == 'os':
                fwtime = CostModel.Systolic_OS(mesh, self.out_dim, self.bsize, self.in_dim, K)
                bdtime = CostModel.Systolic_WS(mesh, self.in_dim, self.bsize, self.out_dim, K)
                bwtime = CostModel.Systolic_IS(mesh, self.out_dim, self.in_dim, self.bsize, K)
            elif self.dataflow == 'is':
                fwtime = CostModel.Systolic_WS(mesh, self.out_dim, self.bsize, self.in_dim, K)
                bdtime = CostModel.Systolic_OS(mesh, self.in_dim, self.bsize, self.out_dim, K)
                bwtime = CostModel.Systolic_IS(mesh, self.in_dim, self.out_dim, self.bsize, K)
            else: #ws
                fwtime = CostModel.Systolic_IS(mesh, self.out_dim, self.bsize, self.in_dim, K)
                bdtime = CostModel.Systolic_WS(mesh, self.bsize, self.in_dim, self.out_dim, K)
                bwtime = CostModel.Systolic_OS(mesh, self.out_dim, self.in_dim, self.bsize, K)
        print("fw: {:.3f}, bd: {:.3f}, bw: {:.3f}")
        return fwtime + bdtime + bwtime
    
class ComputeGraph:
    def __init__(self):
        self.nodenames = [] #Name: node
        self.nodes = {}
        self.terminal = None
        
    def add_operand(self, op_type, inputs, shape):
        for inp in inputs:
            assert inp in self.nodenames
        nodename = op_type + ':' + str(len(self.nodenames))
        self.nodenames.append(nodename)
        node = {'op_name':nodename, 'op_type':op_type, 'inputs':inputs, 'shape': shape}
        self.nodes[nodename] = node
        self.terminal = node
        return node
    def insert_operand(self, op_type, inputs, shape, pos):
        for inp in inputs:
            assert inp in self.nodenames
        nodename = op_type + ':' + str(len(self.nodenames))
        self.nodenames = self.nodenames[:pos] + [nodename] + self.nodenames[pos:]
        node = {'op_name':nodename, 'op_type':op_type, 'inputs':inputs, 'shape': shape}
        self.nodes[nodename] = node
        self.terminal = node
        return node
    def Input(self, shape):
        return self.add_operand('Input', (), shape)
    def FeedForward(self, input, output_dim):
        input_name = input['op_name']
        newshape = copy.deepcopy(self.nodes[input_name]['shape'])
        newshape[-1] = output_dim
        return self.add_operand('FeedForward', (input_name,), newshape)
    def Attention(self, input, num_heads): #assume 3H
        input_name = input['op_name']
        output_shape = self.nodes[input_name]['shape']
        assert output_shape[-1]%3 == 0
        newshape = copy.deepcopy(output_shape)
        newshape[-1] = output_shape[-1]//3
        mynode = self.add_operand('Attention', (input_name,), newshape)
        mynode['num_heads'] = num_heads
        return mynode
    def Add(self, input1, input2): #assume 3H
        input_name1 = input1['op_name']
        input_name2 = input2['op_name']
        assert input1['shape'] == input2['shape']
        return self.add_operand('Add', (input_name1,input_name2), input1['shape'])
    def Gelu(self, input):
        return self.add_operand('Gelu', (input['op_name'],), input['shape'])
    def LayerNorm(self, input):
        return self.add_operand('LayerNorm', (input['op_name'],), input['shape'])
    def printGraph(self):
        for op_name in self.nodenames:
            node = self.nodes[op_name]
            print('{}:{} = {}{}'.format(op_name, node['shape'], node['op_type'], node['inputs']))

def TransformerBlock(B,S,num_heads, head_dim):
    graph = ComputeGraph()
    dim = num_heads*head_dim
    x_in = graph.Input([B,S,dim])
    norm_1 = graph.LayerNorm(x_in)
    ff_in = graph.FeedForward(norm_1, 3*dim)
    att = graph.Attention(ff_in, num_heads)
    ff_out = graph.FeedForward(att, dim)
    add_1 = graph.Add(x_in, ff_out)
    norm_2 = graph.LayerNorm(add_1)
    ff_1 = graph.FeedForward(norm_2, 4*dim)
    gelu_1 = graph.Gelu(ff_1)
    ff_2 = graph.FeedForward(gelu_1, dim)
    add_2 = graph.Add(add_1, ff_2)
    norm_3 = graph.LayerNorm(add_2)
    return graph

def possibleK(B,I,O,meshshape,dataflow):
    mlcm = math.lcm(meshshape[0], meshshape[1])
    if dataflow == 'os':
        kdim = I//mlcm//8
    elif dataflow == 'is':
        kdim = O//mlcm//8
    else:
        kdim = B//mlcm//8
    if kdim%48 == 0:
        return [3,4,6,8,12,16]
    elif kdim%24 == 0:
        return [3,4,6,8,12]
    elif kdim%16 == 0:
        return [4,8,16]
    elif kdim%12 == 0:
        return [3,4,12]
    elif kdim%8 == 0:
        return [4,8]
    elif kdim%6 == 0:
        return [3,6]
    elif kdim%4 == 0:
        return [4]
    else:
        return kdim

#shapes: list of available mesh shapes. Must be a same number of chips
def autotuneTransformer(graph:ComputeGraph, shapes):
    layers = copy.copy(graph.nodenames)
    shardings = {}
    transposition = {}
    dataflows = {}

    #step 1: the dataflow
    for lname in layers:
        layer = graph.nodes[lname]
        if layer['op_type'] == 'FeedForward':
            B = 1
            for sdim in layer['shape'][:-1]:
                B = B*sdim
            inputname = layer['inputs'][0]
            I = graph.nodes[inputname]['shape'][-1]
            O = layer['shape'][-1]
            if O>=I and B>=I:
                dataflows[lname] = 'os'
            elif I>=O and B>=O:
                dataflows[lname] = 'is'
            else: #weight is the biggest
                dataflows[lname] = 'ws'
    #step 2: transpositions
    for pos,lname in enumerate(layers):
        layer = graph.nodes[lname]
        if layer['op_type'] == 'Input':
            transposition[lname] = False
        elif layer['op_type'] == 'FeedForward':
            input_name = layer['inputs'][0]
            assert input_name in transposition
            if dataflows[lname] == 'ws':
                transposition[lname] = not transposition[input_name]
            else:
                transposition[lname] = transposition[input_name]
        elif layer['op_type'] == 'Add':
            input_1 = layer['inputs'][0]
            input_2 = layer['inputs'][1]
            assert input_1 in transposition
            assert input_2 in transposition
            if transposition[input_1] == transposition[input_2]:
                transposition[lname] = transposition[input_1]
            elif transposition[input_1]: #2 is transposed
                transpose_op = graph.insert_operand(
                    'Transpose', (input_2,), layer['shape'],pos)
                layer['inputs'] = (input_1, transpose_op['op_name'])
                transposition[lname] = False
                transposition[transpose_op['op_name']] = False
            else:
                transpose_op = graph.insert_operand(
                    'Transpose', (input_1,), layer['shape'],pos)
                layer['inputs'] = (input_2, transpose_op['op_name'])
                transposition[lname] = False
                transposition[transpose_op['op_name']] = False
        else:
            input_name = layer['inputs'][0]
            assert input_name in transposition
            transposition[lname] = transposition[input_name]
    
    #step 3: find the shape for minimal comm. cost
    bestTime = float('inf')
    bestMesh = shapes[0]
    bestksplits = {}
    for meshshape in shapes:
        ksplits = {}
        mesh = DeviceMesh(meshshape,
                          242*1024**4, bws_per_direction=bws,
                          link_latencies=latencies, base_overheads=base_overheads)
        totalTraffic = 0.0
        totalLatency = 0.0
        totalPipeline = 0.0
        for lname in graph.nodenames:
            layer = graph.nodes[lname]
            if layer['op_type'] == 'FeedForward':
                B = 1
                for sdim in layer['shape'][:-1]:
                    B = B*sdim
                inputname = layer['inputs'][0]
                I = graph.nodes[inputname]['shape'][-1]
                O = layer['shape'][-1]
                model = FFLayerModel(B,I,O,dataflows[lname],transposition[lname])
                totalTraffic += model.totalTrafficTime(mesh)
                klist = possibleK(B,I,O,meshshape, dataflows[lname])
                bestK = klist[0]
                bestLatency = model.totalLatency(mesh, klist[0])
                bestPipeline = model.pipelineOverhead(mesh,klist[0])
                for K in klist[1:]:
                    latency = model.totalLatency(mesh, K)
                    pipe = model.pipelineOverhead(mesh,K)
                    if  latency+ 2*pipe  < bestLatency + 2*bestPipeline:
                        bestK = K
                        bestLatency = latency
                        bestPipeline = pipe
                ksplits[lname] = bestK
                totalLatency += bestLatency
                totalPipeline += bestPipeline
        commTime = totalTraffic + totalLatency
        print("Mesh shape {}, traffic time {:.3f}, latency {:.3f}, pipeline: {:.3f} ms".format(
            meshshape, totalTraffic*1000, totalLatency*1000, totalPipeline*1000
        ))
        if commTime < bestTime:
            bestMesh = meshshape
            bestTime = commTime
            bestksplits = ksplits
    print("Best mesh shape is : {}".format(bestMesh))
    for flayer in dataflows:
        print("Layer: {}\t Dataflow: {}\t Transpose: {}\t ksplit: {}".format(
            flayer, dataflows[flayer], transposition[flayer], bestksplits[flayer]
        ))



if __name__ == '__main__':
    B=64
    S=2048
    H=128
    D=128
    K=8
    gpt3 = TransformerBlock(B,S,H,D)
    gpt3.printGraph()
    computetime = 0
    computetime += 3* CostModel.estimateMatmul(None,B*S//32,H*D//8,H*D//K)*K
    computetime += 3* CostModel.estimateMatmul(None,B*S//32,3*H*D//8,H*D//K)*K
    computetime += 6* CostModel.estimateMatmul(None,B*S//32,4*H*D//8,H*D//K)*K
    print("Emulated compute time: {}".format(computetime*1000))
    #shapes = [(4,96), (8,48), (12,32), (16,24), (24,16), (32,12), (48,8), (96,4)]
    shapes = [(4,64), (8,32), (16,16), (32,8), (64,4)]
    autotuneTransformer(gpt3, shapes)
