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
from keras.utils import to_categorical

from data_reading_visualization import ReadData, CalculateAccuracy, extraerSP_SS, ResultsToFile, createConfusionMatrix, convertToBinary, SummaryString, PlotModelToFile, ClearWeights
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def enasModel(X, Y, validation_split=0.3):
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=validation_split, stratify=Y)
    
    train_set = (X_train, Y_train)
    valid_set = (X_val, Y_val)
    
    
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
    loss = SparseCategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE)
    
    optimizer = SGD(learning_rate=0.05, momentum=0.9)
    
    print(f"X_train: {train_set[0].shape} Y_train: {train_set[1].shape}")
    print(f"X_val: {valid_set[0].shape} Y_val: {valid_set[1].shape}")
    
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

def ReadData(light='WL', input_size = (224,224)):
    
    # Shuffle the data and the labels
    def ShuffleData(X, Y):
        randomize = np.arange(len(X))
        np.random.shuffle(randomize)
        X = X[randomize]
        Y = Y[randomize]
        return X, Y
        
    mainpath = "../" #Local
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

def CalculateAccuracy(Y, Pred):
    booleans = np.equal(Y, Pred)
    n = np.sum(booleans)
    return n/Y.shape[0]

def main():
    print("MAIN")
    np.random.seed(1)
    
    # X, Y = ReadData(light='NBI')
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y)
    # Y_train = convertToBinary(Y_train)
    # Y_test = convertToBinary(Y_test)
    
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