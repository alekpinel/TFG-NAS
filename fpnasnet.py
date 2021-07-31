# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 10:15:56 2021

@author: alekp
"""

import math
import cv2
import numpy as np
import keras
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation, Add
from keras.optimizers import SGD

from keras.datasets import mnist, cifar10
from keras.utils import np_utils

#This function generates all the possible block architectures
def GenerateBlockArchitectureList():
    VERTICES = ['None', 'Split', 'Concat', 'Add']
    EDGES = ['No', 'Id', 'Conv']
    RATIOS = [4, 6, 8]
    STRIDES = [1, 2]
    KERNEL_SIZE = [3, 5]
    
    def PossibleEdges(block):
        poss_edges = ['Id', 'No']
        if ('Conv' not in block and 'None' not in block):
            poss_edges.append('Conv')
        return poss_edges
    
    def PossibleVertices(block):
        poss_vertices = ['Add']
        if ('None' in block):
            return ['None']
        if ('Split' in block or 'Concat' in block or 'Add' in block):
            poss_vertices.append('None')
        if ('Split' in block and 'Concat' not in block):
            poss_vertices.append('Concat')
        return poss_vertices
    
    def AddEdge(block):
        poss_nodes = PossibleEdges(block)
        new_blocks = []
        for node in poss_nodes:
            new_block = block.copy()
            new_block.append(node)
            new_blocks.append(new_block)
        return new_blocks
    
    def AddVertex(block):
        poss_nodes = PossibleVertices(block)
        new_blocks = []
        for node in poss_nodes:
            new_block = block.copy()
            new_block.append(node)
            new_blocks.append(new_block)
        return new_blocks
    
    def IncorporateBlocks(all_blocks, block_type):
        if (block_type == 'Edge'):
            NodeFunction = AddEdge
        elif (block_type == 'Vertex'):
            NodeFunction = AddVertex
            
        new_blocks = []
        for block in all_blocks:
            for new_block in NodeFunction(block):
                new_blocks.append(new_block)
        return new_blocks
    
    def CheckNodeConsistency(node):
        if (node[0] == 'None'):
            if (node[1] == 'Id' and node[2] == 'No' and node[3] == 'No'):
                return True
        else:
            if (node[1] != 'No' and (node[2] == 'No' or node[3] == 'No') and not (node[2] == 'No' and node[3] == 'No')):
                return True
        return False
    
    def CheckBlockConsistency(all_blocks):
        new_blocks = []
        for block in all_blocks:
            if ('Conv' not in block):
                continue
            if ('Split' in block and 'Concat' not in block):
                continue
            
            if (block[1] == 'No' or block[-1] == 'No'):
                continue
            
            if (not (CheckNodeConsistency(block[2:6]) and CheckNodeConsistency(block[6:10]))):
                continue
            
            if (block[2] == 'Concat' and block[4] == 'No'):
                continue
            
            if (block[6] == 'Concat' and block[8] == 'No'):
                continue
            
            if (block[4] == 'Conv'):
                continue
            
            if (block[0] != 'Split' and (block[1] != 'Id' or block[3] == 'No' or block[4] != 'No' or block[8] != 'No')):
                continue
            
            new_blocks.append(block)
        return new_blocks
    
    def ExpandConvs(all_blocks):
        new_blocks = []
        for block in all_blocks:
            concat_index = block.index('Conv')
            for ratio in RATIOS:
                for strides in STRIDES:
                    for kernel in KERNEL_SIZE:
                        new_block = block.copy()
                        new_block[concat_index] = f"Conv_{ratio}_{strides}_{kernel}"
                        new_blocks.append(new_block)
            
        return new_blocks
    
    possible_blocks = []
    
    #Blocks with split
    blocks_2 = [['Split'], ['Add']]
    # Edge 1
    blocks_2 = IncorporateBlocks(blocks_2, 'Edge')
    # Vertex 1
    blocks_2 = IncorporateBlocks(blocks_2, 'Vertex')
    # Edge 2
    blocks_2 = IncorporateBlocks(blocks_2, 'Edge')
    # Edge 3
    blocks_2 = IncorporateBlocks(blocks_2, 'Edge')
    # Edge 3
    blocks_2 = IncorporateBlocks(blocks_2, 'Edge')
    # Vertex 2
    blocks_2 = IncorporateBlocks(blocks_2, 'Vertex')
    # Edge 4
    blocks_2 = IncorporateBlocks(blocks_2, 'Edge')
    # Edge 5
    blocks_2 = IncorporateBlocks(blocks_2, 'Edge')
    # Edge 5
    blocks_2 = IncorporateBlocks(blocks_2, 'Edge')
    # Edge 5
    blocks_2 = IncorporateBlocks(blocks_2, 'Edge')
    
    possible_blocks = possible_blocks + blocks_2
    
    possible_blocks = CheckBlockConsistency(possible_blocks)
    possible_blocks = ExpandConvs(possible_blocks)
    
    
    for block in possible_blocks:
        print(block)
    print(f"N blocks: {len(possible_blocks)}")
    
    return possible_blocks


class Block:
    def __init__(self, block):
        self.vertex1
        self.edge_I_1
        
        self.vertex2
        self.edge_1_2_1
        self.edge_1_2_2
        self.edge_I_2
        
        self.vertex3
        self.edge_2_3
        self.edge_1_3
        self.edge_I_3
        
        self.edge_3_O
        
        # #Case without split
        # if (len(block) == 3):
            
        
        # i = 0
        # while (i < len(block)):
        #     #Case Split
        #     if (block[i] == 'Split'):
                
                
        #         i += 2
        #         continue
            
        #     #Any other node
            
            
            
    

class FPANet:
    def __init__(self, input_shape = (28, 28, 1), output_shape = 10, blocks = None):
        self.input_shape = input_shape
        self.output_shape = output_shape
        
        if (blocks == None):
            blocks = [24, 24, 40, 40, 80, 80, 128, 128]
        
    
            
            


def GetData():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    # Normalize the images.
    X_train = (X_train / 255) - 0.5
    X_test = (X_test / 255) - 0.5
    
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    
    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_test.shape)
    
    return (X_train, y_train), (X_test, y_test)
    

def main():
    print("MAIN")
    # (X_train, y_train), (X_test, y_test) = GetData()
    # input_shape = X_train.shape[1:]
    # output_shape = y_train.shape[1]
    
    # childmodel = FPANet(input_shape=input_shape, output_shape=output_shape)
    
    GenerateBlockArchitectureList()

if __name__ == '__main__':
  main()