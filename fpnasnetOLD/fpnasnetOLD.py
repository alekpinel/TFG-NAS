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

class ChildModel:
    def __init__(self, input_shape = (28, 28, 1), output_shape = 10):
        self.input_shape = input_shape
        self.output_shape = output_shape
        
        block1 = Block()
        block2 = Block()
        block3 = Block()
        
        self.blocks= [block1, block2, block3]
        
        self.EnsambleNetwork()
        
        print(self.model.summary())
        
        #block1.SetTrainable(False)
        # block2.Evolve()
        
        # self.EnsambleNetwork()
        # self.ReesensambleNetwork()
        
        # print(self.model.summary())
        
    def EnsambleNetwork(self):
        self.input = Input(shape=self.input_shape)
        x = self.input
        for block in self.blocks:
            x = block(x)
        self.output = x
        x = Flatten()(x)
        self.output = Dense(self.output_shape, activation='softmax') (x)
        
        self.model = Model(inputs = self.input, outputs = self.output)
    
    def ReesensambleNetwork(self):
        self.model.layers.pop()
        
        self.output = Dense(self.output_shape, activation='softmax') (self.model.layers[-1].output)
        self.model = Model(inputs = self.input, outputs = self.output)
        
    def Compile(self):
        optimizer = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
        loss = keras.losses.categorical_crossentropy
        
        self.model.compile(optimizer=optimizer,
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


class Block():
    def __init__(self):
        self.input_shape = None
        self.layer1 = Conv2D(10, (3, 3), activation='relu', padding='same')
        self.layer2 = Conv2D(10, (3, 3), activation='relu', padding='same')
        
    def __call__(self, inputs):
        #If the input shape has changed, change the first layer
        if (self.input_shape == None):
            self.input_shape = inputs.shape
        elif (not np.array_equal(self.input_shape, inputs.shape)):
            self.input_shape = inputs.shape
            #First layer (temporal)
            self.ResetFirstLayer()
            
        x = self.layer1(inputs)
        x = self.layer2(x)
        return x
    
    def ResetFirstLayer(self):
        self.layer1 = Conv2D(10, (3, 3), activation='relu', padding='same')
    
    def SetTrainable(self, trainable):
        self.layer1.trainable = trainable
        self.layer2.trainable = trainable
    
    def Evolve(self):
        self.layer1 = Conv2D(11, (3, 3), activation='relu', padding='same')
        self.layer2 = Conv2D(7, (3, 3), activation='relu', padding='same')
        
        #x = self.layer1(previous.output)
        #x = self.layer2(x)
        #next(x)

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
    
    childmodel = ChildModel(input_shape=input_shape, output_shape=output_shape)
    
    childmodel.Test(X_test, y_test)
    
    childmodel.Compile()
    childmodel.Train(X_train, y_train, X_test, y_test)
    
    childmodel.Test(X_test, y_test)
    
    childmodel.Evolve()
    
    childmodel.Test(X_test, y_test)

if __name__ == '__main__':
  main()