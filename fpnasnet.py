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
from keras.layers import Input, Dense, Dropout, Flatten, ReLU
from keras.layers import Conv2D, MaxPooling2D, Activation, Add, Concatenate, BatchNormalization, DepthwiseConv2D
from keras.optimizers import SGD
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, DepthwiseConv2D, BatchNormalization

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
            
            if (block[8] == 'Conv' or block[9] == 'Conv'):
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
    # possible_blocks = ExpandConvs(possible_blocks)
    
    
    for block in possible_blocks:
        print(block)
    print(f"N blocks: {len(possible_blocks)}")
    
    return possible_blocks

class InvertedResidual(Layer):
    def __init__(self, filters, strides, expansion_factor=6, trainable=True,
    	         name=None, **kwargs):
        super(InvertedResidual, self).__init__(trainable=trainable, name=name, **kwargs)
        self.filters = filters
        self.strides = strides
        self.expansion_factor = expansion_factor	# allowed to be decimal value

    def build(self, input_shape):
        input_channels = int(input_shape[3])
        self.ptwise_conv1 = Conv2D(filters=int(input_channels*self.expansion_factor),
        	                       kernel_size=1, use_bias=False)
        self.dwise = DepthwiseConv2D(kernel_size=3, strides=self.strides,
        	                         padding='same', use_bias=False)
        self.ptwise_conv2 = Conv2D(filters=self.filters, kernel_size=1, use_bias=False)

        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.bn3 = BatchNormalization()

    def call(self, input_x):
        # Expansion to high-dimensional space
        x = self.ptwise_conv1(input_x)
        x = self.bn1(x)
        x = tf.nn.relu6(x)

        # Spatial filtering
        x = self.dwise(x)
        x = self.bn2(x)
        x = tf.nn.relu6(x)

        # Projection back to low-dimensional space w/ linear activation
        x = self.ptwise_conv2(x)
        x = self.bn3(x)

        # Residual connection if i/o have same spatial and depth dims
        if input_x.shape[1:] == x.shape[1:]:
            x += input_x
        return x

    def get_config(self):
        cfg = super(InvertedResidual, self).get_config()
        cfg.update({'filters': self.filters,
        	        'strides': self.strides,
        	        'expansion_factor': self.expansion_factor})
        return cfg

#Implementation of a Split layer
# https://www.programmersought.com/article/76885404311/
class Split(Layer):
    def __init__(self, **kwargs):
        super(Split, self).__init__(**kwargs)

    def build(self, input_shape):
        # Call the build function of the parent class to build this layer
        super(Split, self).build(input_shape)
        # Save the shape, use other functions
        self.shape = input_shape

    def call(self, x, mask=None):
        # Split x into two tensors
        seq = [x[:, 0:self.shape[1] // 2, ...],
               x[:, self.shape[1] // 2:, ...]]
        return seq

    def compute_mask(self, inputs, input_mask=None):
        # This layer outputs two tensors and needs to return multiple masks, and the mask can be None
        return [None, None]

    def get_output_shape_for(self, input_shape):
        # If this layer returns two tensors, it will return the shape of the two tensors
        shape0 = list(self.shape)
        shape1 = list(self.shape)
        shape0[1] = self.shape[1] // 2
        shape1[1] = self.shape[1] - self.shape[1] // 2
        # print [shape0, shape1]
        return [shape0, shape1]

class Block:
    def __init__(self, block, n_filters):
        self.n_filters = n_filters
        
        self.vertex1 = block[0]
        self.edge_I_1 = block[1]
        
        self.vertex2 = block[2]
        self.edge_1_2_1 = block[3]
        self.edge_1_2_2 = block[4]
        self.edge_I_2 = block[5]
        
        self.vertex3 = block[6]
        self.edge_2_3 = block[7]
        self.edge_1_3 = block[8]
        self.edge_I_3 = block[9]
        
        self.edge_3_O = block[10]
        
        layers = []
    
    def CreateEdge(self, x, layer):
        if (layer == 'No'):
            return 'No'
        elif(layer == 'Id'):
            return x
        else:
            #TODO: inverted residual convolution
            filters = self.n_filters
            ratio = 4
            strides = 1
            return InvertedResidual(filters=filters, strides=strides, expansion_factor=ratio)(x)
            # return self.bottleneck_block(x, filters*ratio, filters)
            # return Conv2D(self.n_filters, (3, 3), activation='relu', padding='same')(x)
    
    def CreateVertex(self, x, layer):
        print(x)
        if (len(x) > 1 and isinstance(x[1], str) and x[1] == 'No'):
            x.pop(1)
        if (len(x) > 2 and isinstance(x[2], str) and x[2] == 'No'):
            x.pop(2)
        
        if (layer == 'None' or len(x) == 1):
            return x[0]
        elif(layer == 'Split'):
            #TODO
            return Split()(x)
        elif(layer == 'Concat'):
            return Concatenate(axis=1)(x)
        elif(layer == 'Add'):
            if x[0].shape[1:] == x[1].shape[1:]:
                return Add()(x)
            else:
                return x[0]
    
    def __call__(self, inputs):
        x_I_1 = self.CreateEdge(inputs, self.edge_I_1)
        
        x_v1 = self.CreateVertex([x_I_1], self.vertex1)
        
        if (self.vertex1 == 'Split'):
            x_1_2_1 = self.CreateEdge(x_v1[0], self.edge_1_2_1)
            x_1_2_2 = self.CreateEdge(x_v1[1], self.edge_1_2_2)
        else:
            x_1_2_1 = self.CreateEdge(x_v1, self.edge_1_2_1)
            x_1_2_2 = self.CreateEdge(x_v1, self.edge_1_2_2)
        
        x_I_2 = self.CreateEdge(inputs, self.edge_I_2)
        
        x_v2 = self.CreateVertex([x_1_2_1, x_1_2_2, x_I_2], self.vertex2)
        
        x_2_3 = self.CreateEdge(x_v2, self.edge_2_3)
        
        if (self.vertex1 == 'Split'):
            x_1_3 = self.CreateEdge(x_v1[1], self.edge_1_3)
        else:
            x_1_3 = self.CreateEdge(x_v1, self.edge_1_3)
            
        
        x_I_3 = self.CreateEdge(inputs, self.edge_I_3)
        
        x_v3 = self.CreateVertex([x_2_3, x_1_3, x_I_3], self.vertex3)
        
        x_3_O = self.CreateEdge(x_v3, self.edge_3_O)
        
        return x_3_O
        
    
    

class FPANet:
    def __init__(self, input_shape = (28, 28, 1), output_shape = 10, block_size = None):
        self.input_shape = input_shape
        self.output_shape = output_shape
        
        if (block_size == None):
            block_size = [24,24, 40,40,40, 80,80,80,80, 112,112,112, 192,192,192, 320]
    
        self.block_filters = block_size
    
    def Generate(self):
        self.input = Input(shape=self.input_shape)
        
        # block_code = ['Add', 'Id', 'Add', 'Id', 'No', 'Id', 'None', 'Id', 'No', 'No', 'Conv_8_2_5']
        block_code = ['Add', 'Id', 'Add', 'Conv_8_2_5', 'No', 'Id', 'None', 'Id', 'No', 'No', 'Id']
        block_code = ['Split', 'Conv', 'Concat', 'Id', 'Id', 'No', 'None', 'Id', 'No', 'No', 'Id']
        x = self.input
        for n_filters in self.block_filters:
            x = Block(block_code, n_filters)(x)
            
        x = Flatten()(x)
        self.output = Dense(self.output_shape, activation='softmax') (x)
        
        self.model = Model(inputs = self.input, outputs = self.output)
        
        print(self.model.summary())
    
    def Compile(self):
        loss = keras.losses.categorical_crossentropy
        
        self.model.compile(optimizer='adam',
                      loss=loss,
                      metrics=['accuracy'])
    def Train(self, X_train, Y_train, X_val, Y_val, epochs=2, batch_size=128):
        
        return self.model.fit(X_train, Y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(X_val, Y_val)
                            )
    
    def Predict(self, X_test):
        results = self.model.predict(X_test)
        return results


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
    (X_train, y_train), (X_test, y_test) = GetData()
    input_shape = X_train.shape[1:]
    output_shape = y_train.shape[1]
    
    # childmodel = FPANet(input_shape=input_shape, output_shape=output_shape)
    
    # GenerateBlockArchitectureList()
    
    model = FPANet(input_shape=input_shape, output_shape=output_shape, block_size=[24, 24])
    model.Generate()
    model.Compile()
    # model.Train(X_train, y_train, X_test, y_test)

if __name__ == '__main__':
  main()