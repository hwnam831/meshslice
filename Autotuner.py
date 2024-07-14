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

latencies = {'allgather':5e-6, 'reducescatter':5e-6, 'sendrecv':5e-5}
bws = {'allgather':(27.472e9, 79.751e9), 'reducescatter':(37.506e9, 63.167e9), 'sendrecv':(38.801e9, 35.224e9)}
base_overheads = {'allgather':13e-6, 'reducescatter':26e-6, 'sendrecv':13e-6}

class FFLayerModel:
    def __init__(self, B,I,O, dataflow=None, transpose=False):
        self.bsize = B
        self.in_dim = I
        self.out_dim = O
        self.transpose = False
        if dataflow in ['os','rs','ls']:
            self.dataflow=dataflow
        elif O>I and B>=I:
            self.dataflow = 'os'
        elif I>=O and B>=O:
            self.dataflow = 'ls'
        else: #weight is the biggest
            self.dataflow = 'rs'
        self.transpose = transpose
    def totalTrafficTime(self, mesh:DeviceMesh):
        
        if self.dataflow == 'os':
            input_traffic = 2*self.bsize*self.in_dim*(mesh.shape[1]-1)
            weight_traffic = 2*self.in_dim*self.out_dim*(mesh.shape[0]-1)
            if self.transpose:
                input_traffic = 2*self.bsize*self.in_dim*(mesh.shape[0]-1)
                weight_traffic = 2*self.in_dim*self.out_dim*(mesh.shape[1]-1)
            fwtime = max(input_traffic/mesh.bws['allgather'][1], weight_traffic/mesh.bws['allgather'][0])
            bdtime = max(input_traffic/mesh.bws['reducescatter'][1], weight_traffic/mesh.bws['allgather'][0])
            bwtime = max(input_traffic/mesh.bws['allgather'][1], weight_traffic/mesh.bws['reducescatter'][0])
            return (fwtime + bdtime + bwtime)/(mesh.shape[0]*mesh.shape[1])
        elif self.dataflow == 'ls':
            output_traffic = 2*self.bsize*self.out_dim*(mesh.shape[1]-1)
            weight_traffic = 2*self.in_dim*self.out_dim*(mesh.shape[0]-1)
            if self.transpose:
                output_traffic = 2*self.bsize*self.out_dim*(mesh.shape[0]-1)
                weight_traffic = 2*self.in_dim*self.out_dim*(mesh.shape[1]-1)
            fwtime = max(weight_traffic/mesh.bws['allgather'][0], output_traffic/mesh.bws['reducescatter'][1])
            bdtime = max(weight_traffic/mesh.bws['allgather'][0], output_traffic/mesh.bws['allgather'][1])
            bwtime = max(weight_traffic/mesh.bws['reducescatter'][0], output_traffic/mesh.bws['allgather'][1])
            return (fwtime + bdtime + bwtime)/(mesh.shape[0]*mesh.shape[1])
        else: #ws
            input_traffic = 2*self.bsize*self.in_dim*(mesh.shape[1]-1)
            output_traffic = 2*self.bsize*self.out_dim*(mesh.shape[0]-1)
            if self.transpose:
                input_traffic = 2*self.bsize*self.in_dim*(mesh.shape[0]-1)
                output_traffic = 2*self.bsize*self.out_dim*(mesh.shape[1]-1)
            fwtime = max(input_traffic/mesh.bws['allgather'][0], output_traffic/mesh.bws['reducescatter'][1])
            bdtime = max(input_traffic/mesh.bws['reducescatter'][0], output_traffic/mesh.bws['allgather'][1])
            bwtime = max(input_traffic/mesh.bws['allgather'][0], output_traffic/mesh.bws['allgather'])
            return (fwtime + bdtime + bwtime)/(mesh.shape[0]*mesh.shape[1])
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
        elif self.dataflow == 'ls':
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
            fwtime = max(input_traffic/mesh.bws['allgather'][1], weight_traffic/mesh.bws['allgather'][0])
            bdtime = (input_traffic/mesh.bws['reducescatter'][1]+ weight_traffic/mesh.bws['allgather'][0])
            bwtime = (input_traffic/mesh.bws['allgather'][1]+ weight_traffic/mesh.bws['reducescatter'][0])
            return (fwtime + bdtime + bwtime)/(mesh.shape[0]*mesh.shape[1])/K
        elif self.dataflow == 'ls':
            output_traffic = 2*self.bsize*self.out_dim*(mesh.shape[1]-1)
            weight_traffic = 2*self.in_dim*self.out_dim*(mesh.shape[0]-1)
            if self.transpose:
                output_traffic = 2*self.bsize*self.out_dim*(mesh.shape[0]-1)
                weight_traffic = 2*self.in_dim*self.out_dim*(mesh.shape[1]-1)
            fwtime = (weight_traffic/mesh.bws['allgather'][0] + output_traffic/mesh.bws['reducescatter'][1])
            bdtime = max(weight_traffic/mesh.bws['allgather'][0], output_traffic/mesh.bws['allgather'][1])
            bwtime = (weight_traffic/mesh.bws['reducescatter'][0] + output_traffic/mesh.bws['allgather'][1])
            return (fwtime + bdtime + bwtime)/(mesh.shape[0]*mesh.shape[1])/K
        else: #ws
            input_traffic = 2*self.bsize*self.in_dim*(mesh.shape[1]-1)
            output_traffic = 2*self.bsize*self.out_dim*(mesh.shape[0]-1)
            if self.transpose:
                input_traffic = 2*self.bsize*self.in_dim*(mesh.shape[0]-1)
                output_traffic = 2*self.bsize*self.out_dim*(mesh.shape[1]-1)
            fwtime = (input_traffic/mesh.bws['allgather'][1] + output_traffic/mesh.bws['reducescatter'][0])
            bdtime = (input_traffic/mesh.bws['reducescatter'][1] + output_traffic/mesh.bws['allgather'][0])
            bwtime = max(input_traffic/mesh.bws['allgather'][1], output_traffic/mesh.bws['allgather'][0])
            return (fwtime + bdtime + bwtime)/(mesh.shape[0]*mesh.shape[1])/K
    def emulate(self, mesh:DeviceMesh, K):
        if not self.transpose:
            if self.dataflow == 'os':
                fwtime = CostModel.MeshFlow_OS(mesh, self.bsize, self.out_dim, self.in_dim, K)
                bdtime = CostModel.MeshFlow_LS(mesh, self.bsize, self.in_dim, self.out_dim, K)
                bwtime = CostModel.MeshFlow_RS(mesh, self.in_dim, self.out_dim, self.bsize, K)
            elif self.dataflow == 'ls':
                fwtime = CostModel.MeshFlow_LS(mesh, self.bsize, self.out_dim, self.in_dim, K)
                bdtime = CostModel.MeshFlow_OS(mesh, self.bsize, self.in_dim, self.out_dim, K)
                bwtime = CostModel.MeshFlow_RS(mesh, self.out_dim, self.in_dim, self.bsize, K)
            else: #ws
                fwtime = CostModel.MeshFlow_RS(mesh, self.bsize, self.out_dim, self.in_dim, K)
                bdtime = CostModel.MeshFlow_LS(mesh, self.in_dim, self.bsize, self.out_dim, K)
                bwtime = CostModel.MeshFlow_OS(mesh, self.in_dim, self.out_dim, self.bsize, K)
        else: #transposed versions
            if self.dataflow == 'os':
                fwtime = CostModel.MeshFlow_OS(mesh, self.out_dim, self.bsize, self.in_dim, K)
                bdtime = CostModel.MeshFlow_RS(mesh, self.in_dim, self.bsize, self.out_dim, K)
                bwtime = CostModel.MeshFlow_LS(mesh, self.out_dim, self.in_dim, self.bsize, K)
            elif self.dataflow == 'ls':
                fwtime = CostModel.MeshFlow_RS(mesh, self.out_dim, self.bsize, self.in_dim, K)
                bdtime = CostModel.MeshFlow_OS(mesh, self.in_dim, self.bsize, self.out_dim, K)
                bwtime = CostModel.MeshFlow_LS(mesh, self.in_dim, self.out_dim, self.bsize, K)
            else: #ws
                fwtime = CostModel.MeshFlow_LS(mesh, self.out_dim, self.bsize, self.in_dim, K)
                bdtime = CostModel.MeshFlow_RS(mesh, self.bsize, self.in_dim, self.out_dim, K)
                bwtime = CostModel.MeshFlow_OS(mesh, self.out_dim, self.in_dim, self.bsize, K)
        #print("fw: {}, bd: {}, bw: {}".format(fwtime, bdtime, bwtime))
        return (fwtime[0] + bdtime[0] + bwtime[0]+fwtime[1] + bdtime[1] + bwtime[1])
    
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

def build_transformerBlock(B,S,num_heads, head_dim):
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
    mlcm = meshshape[0]*meshshape[1] // math.gcd(meshshape[0], meshshape[1])
    if dataflow == 'os':
        kdim = I//mlcm//8
    elif dataflow == 'ls':
        kdim = O//mlcm//8
    else:
        kdim = B//mlcm//8
    if kdim%48 == 0:
        return [3,4,6,8,12,16,24]
    elif kdim%40 == 0:
        return [4,5,8,10,20]
    elif kdim%24 == 0:
        return [3,4,6,8,12,24]
    elif kdim%20 == 0:
        return [4,5,10,20]
    elif kdim%16 == 0:
        return [4,8,16]
    elif kdim%12 == 0:
        return [3,4,12]
    elif kdim%10 == 0:
        return [5,10]
    elif kdim%8 == 0:
        return [4,8]
    elif kdim%6 == 0:
        return [3,6]
    elif kdim%4 == 0:
        return [4]
    elif kdim%5 == 0:
        return [5]
    else:
        return [kdim]
    
   

#shapes: list of available mesh shapes. Must be a same number of chips
class Autotuner:
    def __init__(self, graph, shapes):
        self.graph = graph
        self.shapes = shapes
        self.layers = copy.copy(graph.nodenames)
        self.shardings = {}
        self.transposition = {}
        self.dataflows = {}

        #step 1: the dataflow
        for lname in self.layers:
            layer = graph.nodes[lname]
            if layer['op_type'] == 'FeedForward':
                B = 1
                for sdim in layer['shape'][:-1]:
                    B = B*sdim
                inputname = layer['inputs'][0]
                I = graph.nodes[inputname]['shape'][-1]
                O = layer['shape'][-1]
                if O>I and B>=I:
                    self.dataflows[lname] = 'os'
                elif I>=O and B>=O:
                    self.dataflows[lname] = 'ls'
                else: #weight is the biggest
                    self.dataflows[lname] = 'rs'
        #step 2: transpositions
        for pos,lname in enumerate(self.layers):
            layer = graph.nodes[lname]
            if layer['op_type'] == 'Input':
                self.transposition[lname] = False
            elif layer['op_type'] == 'FeedForward':
                input_name = layer['inputs'][0]
                assert input_name in self.transposition
                if self.dataflows[lname] == 'rs':
                    self.transposition[lname] = not self.transposition[input_name]
                else:
                    self.transposition[lname] = self.transposition[input_name]
            elif layer['op_type'] == 'Add':
                input_1 = layer['inputs'][0]
                input_2 = layer['inputs'][1]
                assert input_1 in self.transposition
                assert input_2 in self.transposition
                if self.transposition[input_1] == self.transposition[input_2]:
                    self.transposition[lname] = self.transposition[input_1]
                elif self.transposition[input_1]: #2 is transposed
                    transpose_op = graph.insert_operand(
                        'Transpose', (input_2,), layer['shape'],pos)
                    layer['inputs'] = (input_1, transpose_op['op_name'])
                    self.transposition[lname] = False
                    self.transposition[transpose_op['op_name']] = False
                else:
                    transpose_op = graph.insert_operand(
                        'Transpose', (input_1,), layer['shape'],pos)
                    layer['inputs'] = (input_2, transpose_op['op_name'])
                    self.transposition[lname] = False
                    self.transposition[transpose_op['op_name']] = False
            else:
                input_name = layer['inputs'][0]
                assert input_name in self.transposition
                self.transposition[lname] = self.transposition[input_name]
        self.bestshape = self.analyticalAutotune() #first find analytical optimal
    def analyticalAutotune(self):
        bestTime = float('inf')
        bestMesh = self.shapes[0]
        bestksplits = {}
        for meshshape in self.shapes:
            ksplits = {}
            mesh = DeviceMesh(meshshape,
                            242*1024**4, bws_per_direction=bws,
                            link_latencies=latencies, base_overheads=base_overheads)
            totalTraffic = 0.0
            totalLatency = 0.0
            totalPipeline = 0.0
            for lname in self.graph.nodenames:
                layer = self.graph.nodes[lname]
                if layer['op_type'] == 'FeedForward':
                    B = 1
                    for sdim in layer['shape'][:-1]:
                        B = B*sdim
                    inputname = layer['inputs'][0]
                    I = self.graph.nodes[inputname]['shape'][-1]
                    O = layer['shape'][-1]
                    model = FFLayerModel(B,I,O,self.dataflows[lname],self.transposition[lname])
                    totalTraffic += model.totalTrafficTime(mesh)
                    klist = possibleK(B,I,O,meshshape, self.dataflows[lname])
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
        for flayer in self.dataflows:
            print("Layer: {}\t Dataflow: {}\t Transpose: {}\t ksplit: {}".format(
                flayer, self.dataflows[flayer], self.transposition[flayer], bestksplits[flayer]
            ))
        self.ksplits = bestksplits
        return bestMesh
    def emulateFFTime(self, meshshape, sweep=False):
        totaltime = 0.0
        mesh = DeviceMesh(meshshape,
                        242*1024**4, bws_per_direction=bws,
                        link_latencies=latencies, base_overheads=base_overheads)
        ksplits = {}
        for lname in self.graph.nodenames:
            layer = self.graph.nodes[lname]
            if layer['op_type'] == 'FeedForward':
                B = 1
                for sdim in layer['shape'][:-1]:
                    B = B*sdim
                inputname = layer['inputs'][0]
                I = self.graph.nodes[inputname]['shape'][-1]
                O = layer['shape'][-1]
                model = FFLayerModel(B,I,O,self.dataflows[lname],self.transposition[lname])
                klist = possibleK(B,I,O,meshshape, self.dataflows[lname])
                bestTime = model.emulate(mesh,klist[0])
                bestK = klist[0]
                if sweep:
                    print(klist)
                    print("FFtime at k={} is {}".format(bestK,bestTime))
                for k in klist[1:]:
                    curtime = model.emulate(mesh,k)
                    if sweep:
                        print("FFtime at k={} is {}".format(k,curtime))
                    if curtime < bestTime:
                        bestK = K
                        bestTime = curtime
                    else:
                        if not sweep:
                            break
                totaltime += bestTime
                ksplits[lname] = bestK
        print("FFtime: {}ms".format(totaltime*1000))
        return totaltime, ksplits
    def emulatedFinetune(self, sweep=False):
        print("Start emulated finetuning: ")
        curidx = self.shapes.index(self.bestshape)
        print("Trying shape of {}".format(self.bestshape))
        curtime, cursplit = self.emulateFFTime(self.bestshape)
        if sweep:
            besttime = curtime
            bestsplit = cursplit
            for shp in self.shapes:
                print("Trying shape of {}".format(shp))
                curtime, cursplit = self.emulateFFTime(shp, sweep=True)
                if curtime < besttime:
                    besttime = curtime
                    self.bsetshape = shp
                    bestsplit = cursplit
            return
        def searchBackward(curidx,curtime,cursplit):
            if curidx == 0:
                return curidx, cursplit
            print("Trying shape of {}".format(self.shapes[curidx-1]))
            prevtime, prevsplit = self.emulateFFTime(self.shapes[curidx-1])
            if curtime <= prevtime:
                return curidx, cursplit
            else:
                return searchBackward(curidx-1, prevtime, prevsplit)
        def searchForward(curidx,curtime,cursplit):
            if curidx == len(self.shapes)-1:
                return curidx, cursplit
            print("Trying shape of {}".format(self.shapes[curidx+1]))
            nexttime, nextsplit = self.emulateFFTime(self.shapes[curidx+1])
            if curtime <= nexttime:
                return curidx, cursplit
            else:
                return searchForward(curidx+1, nexttime, nextsplit)

        if curidx == len(self.shapes)-1:
            bestidx, bestsplit = searchBackward(curidx, curtime, cursplit)
        elif curidx == 0:
            bestidx, bestsplit = searchForward(curidx, curtime, cursplit)
        else:
            print("Trying shape of {}".format(self.shapes[curidx-1]))
            prevtime, prevsplit = self.emulateFFTime(self.shapes[curidx-1])
            print("Trying shape of {}".format(self.shapes[curidx+1]))
            nexttime, nextsplit = self.emulateFFTime(self.shapes[curidx+1])
            if prevtime < curtime:
                bestidx, bestsplit = searchBackward(curidx-1, prevtime, prevsplit)
            elif nexttime < curtime:
                bestidx, bestsplit = searchForward(curidx+1, nexttime, nextsplit)
            else:
                bestidx, bestsplit = curidx, cursplit
        self.bestshape = self.shapes[bestidx]
        self.ksplits = bestsplit
        print("Best mesh shape is : {}".format(self.bestshape))
        for flayer in self.dataflows:
            print("Layer: {}\t Dataflow: {}\t Transpose: {}\t ksplit: {}".format(
                flayer, self.dataflows[flayer], self.transposition[flayer], self.ksplits[flayer]
            ))
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ksplit', nargs=4, type=int, default=[8,8,8,8])
    parser.add_argument('--batchsize', type=int, default=-1, help='Default is number of chips')
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--nheads', type=int, default=96)
    parser.add_argument('--headdim', type=int, default=128)
    parser.add_argument('--nrows', type=int, default=4, help='Number of rows in device mesh. Must be multiple of 4')
    parser.add_argument('--ncols', type=int, default=4, help='Number of cols in device mesh. Must be multiple of 4')
    parser.add_argument('--alg', type=str, default='noff', choices=['noff','collective', 'cannon', 'wang', 'systolic'])
    parser.add_argument('--sweep', action='store_true')
    args = parser.parse_args()

    B = args.nrows*args.ncols if args.batchsize <= 0 else args.batchsize
    S = args.seqlen
    H = args.nheads
    D = args.headdim
    alg = args.alg
    NROW = args.nrows
    NCOL = args.ncols
    K=8
    gpt3 = build_transformerBlock(B,S,H,D)
    gpt3.printGraph()
    computetime = 0
    computetime += 3* CostModel.estimateMatmul(None,B*S//NROW,H*D//NCOL,H*D//K)*K
    computetime += 3* CostModel.estimateMatmul(None,B*S//NROW,3*H*D//NCOL,H*D//K)*K
    computetime += 6* CostModel.estimateMatmul(None,B*S//NROW,4*H*D//NCOL,H*D//K)*K
    print("Emulated compute time: {}".format(computetime*1000))
    #shapes = [(4,96), (8,48), (12,32), (16,24), (24,16), (32,12), (48,8), (96,4)]
    shapes = []
    currow = 4
    curcol = NCOL*NROW//4
    while curcol >= 4:
        shapes.append((currow, curcol))
        currow = currow * 2
        curcol = curcol //2
    print(shapes)
    tuner = Autotuner(gpt3, shapes)
    mesh = DeviceMesh((NROW,NCOL),
                    242*1024**4, bws_per_direction=bws,
                    link_latencies=latencies, base_overheads=base_overheads)
    ffmodel = FFLayerModel(B*S,H*D,4*H*D,'os')
    ffmodel.emulate(mesh,4)
    tuner.emulatedFinetune(args.sweep)
