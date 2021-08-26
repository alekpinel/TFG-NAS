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
from random import sample

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten, ReLU, Lambda, MaxPooling2D, AveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, Activation, Add, Concatenate, BatchNormalization, DepthwiseConv2D
from keras.optimizers import SGD
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, DepthwiseConv2D, BatchNormalization
from utils import SummaryString, ReadData, convertToBinary

from keras.datasets import mnist, cifar10
from keras.utils import np_utils
from sklearn.model_selection import train_test_split


from sklearn import metrics
from utils import CalculateAccuracy, extraerSP_SS, ResultsToFile, createConfusionMatrix, convertToBinary, SummaryString, PlotModelToFile

#Write something to file
def WriteFile(file_name, string, mode='a'):
    mainpath = "./"
    resultpath = mainpath + "ouput/"
    f = open(resultpath + file_name + ".txt", mode)
    f.write(string)
    f.close()

#This function generates all the possible block architectures
def GenerateBlockArchitectureList(verbose=0):
    VERTICES = ['None', 'Add']
    EDGES = ['No', 'Id', 'Conv', 'Avg', 'Max', 'Batch']
    KERNEL_SIZE = [3, 5]
    # STRIDES = [1, 2]
    STRIDES = [1]
    
    def PossibleEdges(block):
        poss_edges = ['Id', 'Conv', 'No', 'Batch']
        
        return poss_edges
    
    def PossibleVertices(block):
        poss_vertices = ['Add', 'None']
        
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
    
    def CheckVertexConsistency(nodes):
        if (nodes[1] == 'No'):
            return False
        
        if (nodes[0] == 'None'):
            if (nodes[2] != 'No'):
                return False
        elif (nodes[0] == 'Add'):
            if (nodes[1] == 'No' or nodes[2] == 'No'):
                return False
            if (nodes[1] == 'Id' and nodes[2] == 'Id'):
                return False
        
        
        return True
        
    
    def CheckBlockConsistency(all_blocks):
        new_blocks = []
        for block in all_blocks:
            #A block must have a conv
            if ('Conv' not in block):
                continue
            
            valid = True
            for i in range(0, len(block), 3):
                if(not CheckVertexConsistency(block[i:i+3])):
                    valid = False
            if (not valid):
                continue
            
            new_blocks.append(block)
        return new_blocks
    
    def ExpandConvs(all_blocks):
        new_blocks = []
        
        for kernel_size in KERNEL_SIZE:
            for strides in STRIDES:
                for block in all_blocks:
                    new_block = block.copy()
                    # if ('Conv' not in block):
                    #     new_blocks.append(new_block)
                    
                    for i in range(len(block)):
                        if (block[i] == 'Conv'):
                                    new_block[i] = f"Conv_{kernel_size}_{strides}"
                    new_blocks.append(new_block)
            
        return new_blocks
    
    possible_blocks = []
    blocks_2 = [[]]
    
    blocks_2 = IncorporateBlocks(blocks_2, 'Vertex')
    # Edge 2
    blocks_2 = IncorporateBlocks(blocks_2, 'Edge')
    # Edge 3
    blocks_2 = IncorporateBlocks(blocks_2, 'Edge')
    
    blocks_2 = IncorporateBlocks(blocks_2, 'Vertex')
    # Edge 2
    blocks_2 = IncorporateBlocks(blocks_2, 'Edge')
    # Edge 3
    blocks_2 = IncorporateBlocks(blocks_2, 'Edge')
    
    possible_blocks = possible_blocks + blocks_2
    
    possible_blocks = CheckBlockConsistency(possible_blocks)
    possible_blocks = ExpandConvs(possible_blocks)
    
    if(verbose >=1):
        for block in possible_blocks:
            print(block)
        print(f"N blocks: {len(possible_blocks)}")
    
    return possible_blocks

class Block:
    def __init__(self, block, n_filters, reduction=True):
        
        self.n_filters = n_filters
        self.description = block
        self.reduction = reduction
        self.trainable_layers = []
        
        self.layers = []
        
        for i in range(0, len(block), 3):
            self.layers.append('Building')
            self.layers.append(self.CreateEdge(i+1))
            self.layers.append(self.CreateEdge(i+2))
            self.layers[i] = self.CreateVertex(i)
            
        self.last_layer = self.AddLayer(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    
    def AddLayer(self, layer):
        self.trainable_layers.append(layer)
        return layer
    
    def CreateEdge(self, nlayer):
        layer = self.description[nlayer]
        if (layer == 'No'):
            return 'No'
        elif(layer == 'Id'):
            return 'Id'
        elif(layer == 'Avg'):
            return self.AddLayer(AveragePooling2D(pool_size=(2,2), strides=(1, 1), padding='same'))
        elif(layer == 'Max'):
            return self.AddLayer(MaxPooling2D(pool_size=(2,2), strides=(1, 1), padding='same'))
        elif(layer == 'Batch'):
            return self.AddLayer(BatchNormalization())
        else:
            elements_list = layer.split("_")
            
            filters = self.n_filters
            kernel_size = (int(elements_list[1]), int(elements_list[1]))
            strides = int(elements_list[2])
            return self.AddLayer(Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same'))
    
    def CreateVertex(self, nlayer):
        layer = self.description[nlayer]
        if (layer == 'None'):
            return 'None'
        elif(layer == 'Add'):
            self.CreateShortcout(nlayer+1)
            self.CreateShortcout(nlayer+2)
            return Add()
            
    def ConnectVertex(self, vertex_index, x1, x2):
        vertex = self.layers[vertex_index]
        if (vertex == 'None'):
            return self.ConnectEdge(vertex_index + 1, x1)
        else:
            return vertex([self.ConnectEdge(vertex_index + 1, x1), self.ConnectEdge(vertex_index + 2, x2)])
    
    def ConnectEdge(self, edge_index, x):
        edge = self.layers[edge_index]
        if (edge == 'Id'):
            return x
        else:
            return edge(x)
    
    def CreateShortcout(self, edge_index):
        print(self.layers)
        strides = 1
        filters = self.n_filters
        layer0 = self.layers[edge_index]
        layer1 = self.AddLayer(Conv2D(filters, (1, 1), strides=strides, padding='same', use_bias=False))
        layer2 = self.AddLayer(BatchNormalization())
        
        def Shortcout(x):
            if (layer0 != 'No' and layer0 != 'Id'):
                x=layer0(x)
            x=layer1(x)
            x=layer2(x)
            return x
        
        self.layers[edge_index] = Shortcout
    
    def CreateShortcoutIfNecessary(self, x):
        if (x[0].shape[1:] == x[1].shape[1:]):
            return x
        else:
            #Calculate the strides
            shortcout = -1
            strides = 1
            if (x[0].shape[1] < x[1].shape[1]):
                shortcout = 1
                strides = (x[1].shape[1] // x[0].shape[1], x[1].shape[2] // x[0].shape[2])
            elif (x[1].shape[1] < x[0].shape[1]):
                shortcout = 0
                strides = (x[0].shape[1] // x[1].shape[1], x[0].shape[2] // x[1].shape[2])
            
            #Calculate the filters
            if (x[0].shape[3] == x[1].shape[3]):
                filters = x[0].shape[3]
            elif (x[0].shape[3] == self.n_filters):
                shortcout = 1
                filters = self.n_filters
            elif (x[1].shape[3] == self.n_filters):
                shortcout = 0
                filters = self.n_filters
            
            # print(f"shortcout {shortcout} strides: {strides} filters: {filters}")
            x[shortcout] = self.AddLayer(Conv2D(filters, (1, 1), strides=strides, padding='same', use_bias=False))(x[shortcout])
            x[shortcout] = self.AddLayer(BatchNormalization())(x[shortcout])
            
            return x
    
    def __call__(self, inputs):
        
        x_v1 = self.ConnectVertex(0, inputs, inputs)
        
        x_v2 = self.ConnectVertex(3, x_v1, inputs)
        
        if (self.reduction):
            output = self.last_layer(x_v2)
        else:
            output = x_v2
        
        return output
    
    def SetTrainable(self, trainable):
        for layer in self.trainable_layers:
            layer.trainable = trainable

class FPANet:
    def __init__(self, input_shape = (28, 28, 1), output_shape = 10, block_size = None):
        self.input_shape = input_shape
        self.output_shape = output_shape
        
        if (block_size == None):
            block_size = [24,24, 40,40,40, 80,80,80,80, 112,112,112, 192,192,192, 320]
    
        self.block_filters = block_size
        self.block_architectures = GenerateBlockArchitectureList()
        
        self.GenerateInitialState()
    
    def SetParameters(self, X_train, Y_train, X_val, Y_val):
        self.data = [X_train, Y_train, X_val, Y_val]
    
    def SetPreMadeBuild(self, blocks):
        self.blocks = []
        
        for nfilter, block in zip(self.block_filters, blocks):
            self.blocks.append(Block(block, nfilter, reduction=True))
        
        if (self.output_shape > 1):
            last_activation = 'softmax'
        else:
            last_activation = 'sigmoid'
        
        self.dense = Dense(self.output_shape, activation=last_activation)
        
        self.Ensamble(verbose=1)
        self.Compile()
    
    def GenerateInitialState(self):
        self.input = Input(shape=self.input_shape)
        self.blocks = []
        
        i = 0
        for n_filters in self.block_filters:
            self.blocks.append(Block(self.block_architectures[np.random.randint(0, len(self.block_architectures))], n_filters, reduction=True))
            i += 1
        
        if (self.output_shape > 1):
            last_activation = 'softmax'
        else:
            last_activation = 'sigmoid'
        
        self.dense = Dense(self.output_shape, activation=last_activation)
        
        self.Ensamble(verbose=1)
        
        print("Initial State")
        print(self.model.summary())
    
    def Ensamble(self, trainable = True, verbose=0):
        x = self.input
        for block in self.blocks:
            if (verbose):
                print(block.description)
            x = block(x)
            block.SetTrainable(trainable)
            
        x = Flatten()(x)
        x = Dropout(rate=0.2)(x)
        
        x = self.dense(x)
        # self.dense.trainable = trainable
        
        self.model = Model(inputs = self.input, outputs = x)
    
    def OptimizeBlock(self, block_index, epochs=2, n_best_models = 3, best_epochs = 4, batch_size=32, verbose=0):
        if verbose:
            print(f"\n\nOPTIMIZING BLOCK {block_index}\n") 
        results = []
        all_blocks = []
        i = 0
        for block_architecture in self.block_architectures:
            all_blocks.append(Block(block_architecture, self.block_filters[block_index]))
            self.blocks[block_index] = all_blocks[-1]
            
            if verbose:
                print(f"Block Architecture {i}/{len(self.block_architectures)}")
                print(all_blocks[-1].description)
            
            #Ensamble with all blocks frozen except one
            self.Ensamble(False, verbose=0)
            self.blocks[block_index].SetTrainable(True)
            
            # if verbose:
            #     print(SummaryString(self.model))
            
            # if verbose:
            #     print(self.model.summary())
                
            self.Compile()
            res = self.Train(self.data[0], self.data[1], self.data[2], self.data[3], epochs=epochs, batch_size=batch_size)
            results.append(res.history['val_loss'][-1])
            
            
            
            i+=1
        
        selected_index = sorted(range(len(results)),key=results.__getitem__)[:n_best_models]
        if verbose:
            print(f"Selected models: {selected_index}") 
        
        best_result = 100000
        best_block = None
        for index in selected_index:
            self.blocks[block_index] = all_blocks[index]
            
            #Ensamble with all block unfrozen
            self.Ensamble(True, verbose)
            ClearWeights(self.model)
            self.Compile()
            res = self.Train(self.data[0], self.data[1], self.data[2], self.data[3], epochs=best_epochs, batch_size=batch_size)
            result = res.history['val_loss'][-1]
            if (result < best_result):
                best_result = result
                best_block = self.blocks[block_index]
        
        self.blocks[block_index] = best_block
        self.Ensamble(True)
        ClearWeights(self.model)
        self.Compile()
        self.Train(self.data[0], self.data[1], self.data[2], self.data[3], epochs=best_epochs, batch_size=batch_size)
        
        print(f"Best block {block_index}: {best_block.description}")
        print(f"{SummaryString(self.model)}")
        
        WriteFile("fpnas_log", f"Block {block_index} {self.blocks[block_index].description} Val Loss: {best_result}\n")
        
        
        
        return best_block
        
    def OptimizeArchitecture(self, P=4, Q=4, E=10, T=1, DEBUG=None, batch_size=32):
        if (DEBUG is not None):
            self.block_architectures = sample(self.block_architectures, DEBUG)
        
        
        WriteFile("fpnas_log", "FPNAS Optimizing Architecture\n")
        
        for i in range(T):
            for block_index in range(len(self.block_filters)):
                self.OptimizeBlock(block_index, epochs=P, n_best_models=E, best_epochs=Q, verbose=1, batch_size=batch_size)
                
        print("Final Architecture")
        for block in self.blocks:
            print(block.description)
        
        WriteFile("fpnas_log", f"Final Model \n{SummaryString(self.model)}\n")
    
    def Compile(self):
        if (self.output_shape > 1):
            loss = keras.losses.categorical_crossentropy
            self.model.compile(optimizer='adam',
                          loss=loss,
                          metrics=['accuracy'])
        else:
            loss = keras.losses.binary_crossentropy
            self.model.compile(loss=loss,
              optimizer=keras.optimizers.Adam(),
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

def fpnasModel(X, Y, P, Q, E, T, D=None, validation_split=0.15, blocks_size=None, batch_size=1, blocks=None):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=validation_split, stratify=Y)
    input_shape = X_train.shape[1:]
    if (len(y_train.shape)>1):
        output_shape = y_train.shape[1]
    else:
        output_shape = 1
    
    if (blocks_size == None):
        blocks_size = [24, 24,40,40,128,128]
    
    print(f"Train: {X_train.shape}")
    print(f"Test: {X_test.shape}")
    
    model = FPANet(input_shape=input_shape, output_shape=output_shape, block_size=blocks_size)
    model.SetParameters(X_train, y_train, X_test, y_test)
    
    # model.Compile()
    # model.model.summary()
    # model.Train(X_train, y_train, X_test, y_test, epochs=4, batch_size=8)
    # predictions = model.Predict(X_train)
    # print(predictions)
    # print(y_test)
    
    if (blocks == None):
        model.OptimizeArchitecture(P=P, Q=Q, E=E, T=T, DEBUG=D, batch_size=batch_size)
    else:
        model.SetPreMadeBuild(blocks);
    
    
    return model.model

def GetCIFARData():
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
    np.random.seed(1)
    
    X, Y = ReadData(light='NBI')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y)
    Y_train = convertToBinary(Y_train)
    Y_test = convertToBinary(Y_test)
    
    # GenerateBlockArchitectureList(verbose=1)
    # return 0
    
    batchsize = 32
    
    blocks = [['Add', 'Conv_5_1', 'Conv_5_1', 'None', 'Batch', 'No'],
              ['Add', 'Id', 'Conv_3_1', 'Add', 'Id', 'Batch']]
    
    blocks_size=[32, 64]
    model = fpnasModel(X_train, Y_train, validation_split=0.15, P=4, Q=10, E=10, T=1, D=None, batch_size=batchsize, blocks_size=blocks_size, blocks=blocks)
    
    #Test
    ClearWeights(model)
    model.fit(X_train, Y_train, batch_size=batchsize, epochs=50)
    
    
    y_predict = model(X_test)
    y_predict = [1 if val > 0.5 else 0 for val in y_predict]
    cm = metrics.confusion_matrix(Y_test, y_predict)
    accuracy, specificity, sensitivity, precision, f1score = extraerSP_SS(cm)
    statistics_s = f"\nRESULTS: \nACC:{accuracy:.3f} \nSP:{specificity:.3f} \nSS:{sensitivity:.3f} \nPr:{precision:.3f} \nScore:{f1score:.3f}"
    print(statistics_s)
    createConfusionMatrix(cm, "Test", save=False)
    
    return 0
    
    
    
    (X_train, y_train), (X_test, y_test) = GetCIFARData()
    input_shape = X_train.shape[1:]
    output_shape = y_train.shape[1]
    
    # childmodel = FPANet(input_shape=input_shape, output_shape=output_shape)
    
    # GenerateBlockArchitectureList()
    
    model = FPANet(input_shape=input_shape, output_shape=output_shape, block_size=[24,40, 40])
    model.SetParameters(X_train, y_train, X_test, y_test)
    
    D=5
    P=2
    Q=4
    E=3
    T=1
    fpnasModel(X_train, y_train, validation_split=0.15, P=P, Q=Q, E=E, T=T, batch_size=8)
    # model.OptimizeArchitecture(P=P, Q=Q, E=E, T=T, DEBUG=D)
    
    return 0
    
    
    


if __name__ == '__main__':
  main()