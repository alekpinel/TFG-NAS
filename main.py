# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 09:49:23 2020

@author: alekp
"""

import math
import numpy as np
import keras
from sklearn.model_selection import train_test_split, LeaveOneOut
from numpy.random import seed
import time
from tensorflow.keras.utils import to_categorical
import statistics
from sklearn import metrics
from keras.models import clone_model

from data_reading_visualization import ReadData, CalculateAccuracy, extraerSP_SS, ResultsToFile, createConfusionMatrix, convertToBinary, SummaryString
from original_nn import OriginalNN 
from autokeras_model import autokerasModel

def Compile(model):
    model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

def NASExperiment(X, Y, model_name, NAS_function, NAS_parameters, test_percent=0.3, epochs=50, batch_size=32, verbose=1, save_results=True):
    
    # Split the data and prepare the binary
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_percent, stratify=Y)
    Y_train = convertToBinary(Y_train)
    Y_test = convertToBinary(Y_test)
    
    if (verbose):
        print(f"X_train: {X_train.shape} X_test: {X_test.shape}")
    
    
    # Apply NAS to search the architecture
    start_time = time.time()
    
    NAS_parameters['X']=X_train
    NAS_parameters['Y']=Y_train
    NAS_model = NAS_function(**NAS_parameters)
    
    end_time = time.time()
    seconds = end_time - start_time
    searching_time = seconds
    
    if (verbose>=1):
        print(f"{model_name}:")
        print(SummaryString(NAS_model))
        
    result_s = f"{model_name}:"
    result_s += SummaryString(NAS_model)
    
    # With the final model, apply leave one out
    cm, fit_time = leaveOneOut(X_train, X_test, Y_train, Y_test, NAS_model, epochs=epochs, batch_size=batch_size, verbose=1)
    
    # Extract the results and show them
    accuracy, specificity, sensitivity, precision, f1score = extraerSP_SS(cm)
    statistics_s = f"\nRESULTS: \nACC:{accuracy:.3f} \nSP:{specificity:.3f} \nSS:{sensitivity:.3f} \nPr:{precision:.3f} \nScore:{f1score:.3f} \nSearching time:{searching_time:.3f} \nFitting time:{fit_time:.3f}"
    result_s += statistics_s
    
    if (verbose):
        print(statistics_s)
    
    if (save_results):
        ResultsToFile(model_name, result_s)
    
    createConfusionMatrix(cm, model_name, save=save_results)

# Realizes Leave One Out only in the test set, while the training is always used
def leaveOneOut(X_train, X_test, Y_train, Y_test, original_model, epochs=50, batch_size=32, verbose=1): 
    loo = LeaveOneOut()
    loo.get_n_splits(X_test)
    y_predict = []
    times = []
    
    labels_test = Y_test
  
    contador = 0
    for t_v_i, test_i in loo.split(X_test):
        if (verbose):
            print(f"LOOP {contador}:")
        
        # print(f"X_train size: {X_train.shape} X_test[t_v_i] {X_test[t_v_i].shape}")
        X_fold_train = np.concatenate((X_train, X_test[t_v_i]))
        Y_fold_train = np.concatenate((Y_train, Y_train[t_v_i]))
  
        X_fold_test = X_test[test_i]
        Y_fold_test = Y_test[test_i]
        
        new_model = clone_model(original_model)
        Compile(new_model)
        
        start_time = time.time()
        
        h = new_model.fit(X_fold_train, Y_fold_train, epochs=epochs, batch_size=batch_size, verbose=0)
        
        end_time = time.time()
        seconds = end_time - start_time
        times.append(seconds)
        
        if (verbose):
            print("Train Loss ",round(statistics.mean(h.history['loss']),3))
            print("Train Accuracy ",round(statistics.mean(h.history['accuracy']),3))
            print(f"Train Time {seconds}")
  
        y_pre = new_model(X_fold_test)
        y_pre = [np.argmax(values) for values in y_pre]
        # print(f"Y_true: {np.argmax(y_test, axis=1)} Y_pre: {y_pre}")
        y_predict.append(y_pre)
        
        if (verbose):
            print(f"Eval Accuracy: {CalculateAccuracy(Y_fold_test, y_pre)}")
        
        contador +=1
  
    cm = metrics.confusion_matrix(labels_test, y_predict)
    mean_time = sum(times) / len(times)
    
    return cm, mean_time

def main():
    print("MAIN")
    
    #Set seed for reproducible results
    seed(1)
    
    X, Y = ReadData(light='WL')
    # print(X.shape)
    # print(Y.shape)
    
    # print(Y)
    # for x in X[0:5]:
    #     ShowImage(x)
    
    
    originalNN_parameters = {'input_size_net':(224,224,3), 'output_size':1}
    NASExperiment(X, Y, "OriginalNN", OriginalNN, originalNN_parameters)
    
    autokeras_parameters = {'validation_split':0.15, 'epochs':50, 'max_trials':100}
    NASExperiment(X, Y, "Autokeras", autokerasModel, autokeras_parameters)

if __name__ == '__main__':
  main()