# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tensorflow as tf
import logging
import time
from argparse import ArgumentParser

import torch
import torch.nn as nn

import datasets
from macro import GeneralNetwork
from micro import MicroNetwork
from nni.algorithms.nas.pytorch import enas
from nni.nas.pytorch.callbacks import (ArchitectureCheckpoint,
                                       LRSchedulerCallback)
from utils import accuracy, reward_accuracy

import numpy as np
from sklearn.model_selection import train_test_split
import sys, os
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical

from data_reading_visualization import ReadData, CalculateAccuracy, extraerSP_SS, ResultsToFile, createConfusionMatrix, convertToBinary, SummaryString, PlotModelToFile, ClearWeights
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from utils import accuracy, reward_accuracy



def enasModel(X, Y, validation_split=0.3):
    # X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=validation_split, stratify=Y)
    
    # train_set = (X_train, Y_train)
    # valid_set = (X_val, Y_val)
    
    train_set = (X, Y)
    
    
    # train_set, valid_set = get_dataset()
    
    model = MicroNetwork(num_layers=3, num_nodes=5, out_channels=20, num_classes=10, dropout_rate=0.1)
    
    batchsize = 4
    # epochs = 10
    # child_steps = 500
    # mutator_steps = 50
    
    epochs = 2
    child_steps = 400
    mutator_steps = 10
    
    # epochs = 1
    # child_steps = 1
    # mutator_steps = 1
    
    # loss = BinaryCrossentropy(from_logits=True, reduction=Reduction.NONE)
    # loss = SparseCategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE)
    # optimizer = SGD(learning_rate=0.05, momentum=0.9)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 0.05, momentum=0.9, weight_decay=1.0E-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.001)
    log_frequency = 10
    ctrl_kwargs = {"tanh_constant": 1.1}
    
    
    
    print(f"X_train: {train_set[0].shape} Y_train: {train_set[1].shape}")
    # print(f"X_val: {valid_set[0].shape} Y_val: {valid_set[1].shape}")
    
    # trainer = enas.EnasTrainer(model,
    #                            loss=loss,
    #                            metrics=accuracy_metrics,
    #                            reward_function=accuracy,
    #                            optimizer=optimizer,
    #                            batch_size=batchsize,
    #                            num_epochs=epochs,
    #                            child_steps=child_steps,
    #                            mutator_steps=mutator_steps,
    #                            dataset_train=train_set,
    #                            dataset_valid=valid_set)
    
    from nni.retiarii.oneshot.pytorch.enas import EnasTrainer
    trainer = EnasTrainer(model,
                          loss=criterion,
                          metrics=accuracy,
                          reward_function=reward_accuracy,
                          optimizer=optimizer,
                          batch_size=batchsize,
                          num_epochs=epochs,
                          dataset=train_set,
                          log_frequency=log_frequency,
                          ctrl_kwargs=ctrl_kwargs)
    
    trainer.fit()
    
    return trainer.model

def extract0_1(data_x, data_y):
    y = []
    x = []	
	# Solo guardamos los datos cuya clase sea la 1 o la 5
    for i in range(0,data_y.shape[0]):
        if data_y[i] == 0 or data_y[i] == 1:
            if data_y[i] == 0:
                y.append(0)
            else:
                y.append(1)
            x.append(data_x[i])
			
    x = np.array(x, np.float64)
    y = np.array(y, np.int64).reshape((len(y), 1))
    return x, y

def get_dataset():
    (x_train, y_train), (x_valid, y_valid) = tf.keras.datasets.cifar10.load_data()
    x_train, x_valid = x_train / 255.0, x_valid / 255.0
    
    # x_train, y_train = extract0_1(x_train, y_train)
    # x_valid, y_valid = extract0_1(x_valid, y_valid)
    
    train_set = (x_train, y_train)
    valid_set = (x_valid, y_valid)
    return train_set, valid_set   

def CalculateAccuracy(Y, Pred):
    booleans = np.equal(Y, Pred)
    n = np.sum(booleans)
    return n/Y.shape[0]

def main():
    print("MAIN")
    np.random.seed(1)
    
    dataset_train, dataset_valid = get_dataset()
    #model = GeneralNetwork()
    
    print(f"X_train: {dataset_train[0].shape} Y_train: {dataset_train[1].shape}")
    print(f"X_val: {dataset_valid[0].shape} Y_val: {dataset_valid[1].shape}")
    
    print (dataset_train[0].shape)
    print (dataset_train[1].shape)
    print (dataset_train[1][:15])
    
    model = enasModel(dataset_train[0][:100], dataset_train[1][:100])
    
    X_test = dataset_valid[0][:100]
    Y_test = dataset_valid[1][:100]
    
    print(f"X_test: {X_test.shape} Y_test: {Y_test.shape}")
    
    y_predict = model(X_test)
    print(y_predict.shape)
    print(y_predict)
    # y_predict = [1 if val > 0.5 else 0 for val in y_predict]
    y_predict = np.argmax(y_predict, axis=1)
    print(y_predict)
    
    print(np.unique(y_predict))
    
    accuracy = CalculateAccuracy(Y_test, y_predict)
    print(f"Accuracy: {accuracy}")
    
    # cm = metrics.confusion_matrix(Y_test, y_predict)
    # accuracy, specificity, sensitivity, precision, f1score = extraerSP_SS(cm)
    # cm = metrics.confusion_matrix(Y_test, y_predict)
    
    # con_mat_df = pd.DataFrame(cm)

    # figure = plt.figure(figsize=(5, 5))
    # sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
    # plt.title("Confusion Matrix: ")
    # plt.tight_layout()
    # plt.xlabel('Predicted label')
    # plt.ylabel('True label')
    
    # plt.show()
    # plt.close()
    


if __name__ == '__main__':
  main()