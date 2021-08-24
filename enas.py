# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tensorflow as tf
from tensorflow.keras.losses import Reduction, SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD

from nni.algorithms.nas.tensorflow import enas

from macro import GeneralNetwork
from micro import MicroNetwork
from utils import accuracy, accuracy_metrics

import numpy as np
from sklearn.model_selection import train_test_split
from data_reading_visualization import SummaryString, ReadData, convertToBinary


def enasTest(dataset_train, dataset_valid):
    model = MicroNetwork()
    
    loss = SparseCategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE)
    optimizer = SGD(learning_rate=0.05, momentum=0.9)
    
    trainer = enas.EnasTrainer(model,
                               loss=loss,
                               metrics=accuracy_metrics,
                               reward_function=accuracy,
                               optimizer=optimizer,
                               batch_size=64,
                               num_epochs=310,
                               dataset_train=dataset_train,
                               dataset_valid=dataset_valid)
    trainer.train()
    
def get_dataset():
    (x_train, y_train), (x_valid, y_valid) = tf.keras.datasets.cifar10.load_data()
    x_train, x_valid = x_train / 255.0, x_valid / 255.0
    train_set = (x_train, y_train)
    valid_set = (x_valid, y_valid)
    return train_set, valid_set   

def main():
    print("MAIN")
    np.random.seed(1)
    
    X, Y = ReadData(light='NBI')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y)
    Y_train = convertToBinary(Y_train)
    Y_test = convertToBinary(Y_test)
    
    dataset_train, dataset_valid = get_dataset()
    #model = GeneralNetwork()
    
    print (dataset_train.shape)
    
    

if __name__ == '__main__':
  main()