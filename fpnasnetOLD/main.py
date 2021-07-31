# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 09:49:23 2020

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

class FPANet:
    def __init__(self, input_shape = (28, 28, 1), output_shape = 10):
        self.input_shape = input_shape
        self.output_shape = output_shape
        
        block1 = Block(16)
        block2 = Block(32)
        block3 = Block(64)
        
        self.blocks= [block1, block2, block3]
        
        self.EnsambleNetwork()
        
        print(self.model.summary())
        
    def EnsambleNetwork(self):
        self.input = Input(shape=self.input_shape)
        x = self.input
        for block in self.blocks:
            x = block(x)
            
        x = Flatten()(x)
        self.output = Dense(self.output_shape, activation='softmax') (x)
        
        self.model = Model(inputs = self.input, outputs = self.output)
        
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
    
    def Test(self, X_test, Y_test, verbose=True):
        preds = self.Predict(X_test)
        labels = np.argmax(Y_test, axis = 1)
        preds = np.argmax(preds, axis = 1)
        accuracy = sum(labels == preds)/len(labels)
        
        if (verbose):
            print(f"Test accuracy is: {accuracy}")
        
        return accuracy
    
    def Evolve(self):
        block2 = self.blocks[1]
        block2.Evolve()
        self.EnsambleNetwork()
        print(self.model.summary())

EDGES = ['Conv2D', 'BatchNorm', 'Dropout']

class BlockEdge():
    def __init__(self):
        self.type = EDGES[0]
    

class Block():
    def __init__(self, nfilters, reduction=True):
        self.nfilters = nfilters
        self.reduction = True
        self.Generate()
        
    def Generate(self):
        nfilters = self.nfilters
        self.layer1 = Conv2D(nfilters / 2, (3, 3), activation='relu', padding='same')
        self.layer2 = Conv2D(nfilters, (3, 3), activation='relu', padding='same')
        self.layer3 = Conv2D(nfilters * 2, (3, 3), activation='relu', padding='same')
        
        self.final = []
        if (self.reduction):
            self.final.append(MaxPooling2D())
        self.final.append(Conv2D(nfilters, (3, 3), activation='relu', padding='same'))
        
    def __call__(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)
        
        for layer in self.final:
            x = layer(x)
        return x
    
    def SetTrainable(self, trainable):
        self.layer1.trainable = trainable
        self.layer2.trainable = trainable
        self.layer3.trainable = trainable
    
    def Evolve(self):
        self.Generate()
        self.layer1 = Conv2D(11, (3, 3), activation='relu', padding='same')
        self.layer2 = Conv2D(7, (3, 3), activation='relu', padding='same')
        
        
def GetData():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    # Normalize the images.
    X_train = (X_train / 255) - 0.5
    X_test = (X_test / 255) - 0.5
    
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    
    return (X_train, y_train), (X_test, y_test)
    


def main():
    print("MAIN")
    (X_train, y_train), (X_test, y_test) = GetData()
    input_shape = X_train.shape[1:]
    output_shape = y_train.shape[1]
    
    childmodel = FPANet(input_shape=input_shape, output_shape=output_shape)
    
    childmodel.Test(X_test, y_test)
    
    # childmodel.Compile()
    # childmodel.Train(X_train, y_train, X_test, y_test)
    
    # childmodel.Test(X_test, y_test)
    
    childmodel.Evolve()
    childmodel.Compile()
    childmodel.Train(X_train, y_train, X_test, y_test)
    
    childmodel.Test(X_test, y_test)

if __name__ == '__main__':
  main()