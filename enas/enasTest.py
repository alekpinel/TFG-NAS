# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tensorflow as tf
from tensorflow.keras.losses import Reduction, SparseCategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.optimizers import SGD

from nni.algorithms.nas.tensorflow import enas

from macro import GeneralNetwork
from micro import MicroNetwork
from utils import accuracy, accuracy_metrics

import numpy as np
from sklearn.model_selection import train_test_split
import sys, os
from keras.preprocessing.image import load_img, img_to_array



def enasModel(X, Y, validation_split=0.3):
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=validation_split, stratify=Y)
    train_set = (X_train, Y_train)
    valid_set = (X_val, Y_val)
    
    # train_set, valid_set = get_dataset()
    
    model = MicroNetwork(num_layers=3, num_nodes=5, out_channels=20, num_classes=1, dropout_rate=0.1)
    
    batchsize = 8
    epochs = 4
    child_steps = 100
    mutator_steps = 10
    
    loss = BinaryCrossentropy(from_logits=True, reduction=Reduction.NONE)
    # loss = SparseCategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE)
    
    optimizer = SGD(learning_rate=0.05, momentum=0.9)
    
    print(f"X: {train_set[0].shape} Y: {train_set[1].shape}")
    
    trainer = enas.EnasTrainer(model,
                               loss=loss,
                               metrics=accuracy_metrics,
                               reward_function=accuracy,
                               optimizer=optimizer,
                               batch_size=batchsize,
                               num_epochs=epochs,
                               child_steps=child_steps,
                               mutator_steps=mutator_steps,
                               dataset_train=train_set,
                               dataset_valid=valid_set)
    trainer.train()
    
    return trainer.model
    
def get_dataset():
    (x_train, y_train), (x_valid, y_valid) = tf.keras.datasets.cifar10.load_data()
    x_train, x_valid = x_train / 255.0, x_valid / 255.0
    train_set = (x_train, y_train)
    valid_set = (x_valid, y_valid)
    return train_set, valid_set   

def ReadData(light='WL', input_size = (224,224)):
    
    # Shuffle the data and the labels
    def ShuffleData(X, Y):
        randomize = np.arange(len(X))
        np.random.shuffle(randomize)
        X = X[randomize]
        Y = Y[randomize]
        return X, Y
        
    mainpath = "./" #Local
    imagepath = mainpath + "data/" + light + '/'
    
    paths = [imagepath + 'adenoma', imagepath + 'hyperplasic', imagepath + 'serrated']
    
    labels = range(len(paths))

    valid_images = [".jpg"]
    X = []
    X_names = []
    Y = []
    
    for p, y in zip(paths, labels):
        lst = os.listdir(p)
        lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        
        for f in lst:
            ext = os.path.splitext(f)[1]
    
            if ext.lower() not in valid_images:
                continue
                
            img = load_img(os.path.join(p,f),target_size=input_size)
            X_names.append(f.split(".")[0])
            img_array = img_to_array(img)
            X.append(img_array)
            Y.append(y)
    
    
    X = np.asarray(X)
    X /= 255
    Y = np.array(Y)
    
    X, Y = ShuffleData(X, Y)
    # Y = convertToBinary(Y)
    
    return X, Y

def main():
    print("MAIN")
    np.random.seed(1)
    
    # X, Y = ReadData(light='NBI')
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y)
    # Y_train = convertToBinary(Y_train)
    # Y_test = convertToBinary(Y_test)
    
    dataset_train, dataset_valid = get_dataset()
    #model = GeneralNetwork()
    
    print (dataset_train[0].shape)
    print (dataset_train[1].shape)
    
    # enasModel(dataset_train[0], dataset_train[1])
    
    

if __name__ == '__main__':
  main()