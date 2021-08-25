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

from utils import ReadData, CalculateAccuracy, extraerSP_SS, ResultsToFile, createConfusionMatrix, set_tf_loglevel
from utils import convertToBinary, SummaryString, PlotModelToFile, ClearWeights, NumpyDataset, predict_pytorch
from original_nn import OriginalNN 
from autokeras_model import autokerasModel
from fpnasnet2 import fpnasModel

import sys
sys.path.insert(0, './enas')
from enasTest import enasModel, enasModelFromNumpy

import logging

set_tf_loglevel(logging.FATAL)


def Compile(model):
    model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

def NASExperiment(X, Y, model_name, NAS_function, NAS_parameters, test_percent=0.3, epochs=50, batch_size=32, verbose=1, save_results=True, clearModel=True, api='tensorflow'):
    leave_one_out=False
    
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
    
    if (verbose):
        print(f"{model_name}:")
        print(SummaryString(NAS_model, api))
        
    result_s = f"{model_name}:"
    result_s += SummaryString(NAS_model, api)
    
    # if (save_results):
    #     PlotModelToFile(NAS_model, model_name)
    
    if (leave_one_out):
        # With the final model, apply leave one out
        cm, fit_time = leaveOneOut(X_train, X_test, Y_train, Y_test, NAS_model, epochs=epochs, batch_size=batch_size, verbose=1, api=api)
    else:
        cm, fit_time = holdOut(X_train, X_test, Y_train, Y_test, NAS_model, clearModel=clearModel, epochs=epochs, batch_size=batch_size, verbose=1, api=api)
    
    # Extract the results and show them
    accuracy, specificity, sensitivity, precision, f1score = extraerSP_SS(cm)
    statistics_s = f"\nRESULTS: \nACC:{accuracy:.3f} \nSS:{sensitivity:.3f} \nSP:{specificity:.3f} \nPr:{precision:.3f} \nScore:{f1score:.3f} \nSearching time:{searching_time:.3f} \nFitting time:{fit_time:.3f}"
    result_s += statistics_s
    
    if (verbose):
        print(statistics_s)
    
    if (save_results):
        ResultsToFile(model_name, result_s)
    
    createConfusionMatrix(cm, model_name, save=save_results)

# Realizes Leave One Out only in the test set, while the training is always used
def leaveOneOut(X_train, X_test, Y_train, Y_test, original_model, epochs=50, batch_size=32, verbose=1, clear=True, api='tensorflow'): 
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
        # new_model = ClearWeights(original_model)
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
        y_pre = [1 if val > 0.5 else 0 for val in y_pre]
        # print(f"Y_true: {Y_fold_test} Y_pre: {y_pre}")
        y_predict.append(y_pre)
        
        if (verbose):
            print(f"Eval Accuracy: {CalculateAccuracy(Y_fold_test, y_pre)}")
        
        contador +=1
  
    cm = metrics.confusion_matrix(labels_test, y_predict)
    mean_time = sum(times) / len(times)
    
    return cm, mean_time

# Realizes Leave One Out only in the test set, while the training is always used
def holdOut(X_train, X_test, Y_train, Y_test, original_model, epochs=50, batch_size=32, verbose=1, clearModel=True, api='tensorflow'): 
    new_model = original_model
    if (clearModel):
        new_model = ClearWeights(original_model)
        Compile(new_model)
    
    start_time = time.time()
    
    if (clearModel):
        h = new_model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    end_time = time.time()
    seconds = end_time - start_time
    
    mean_time = seconds
    cm = EvaluateModel(new_model, X_test, Y_test, api=api, printResults=True)
    EvaluateModel(new_model, X_train, Y_train, api=api, printResults=True)
    
    return cm, mean_time

def EvaluateModel(model, X_test, Y_test, api='tensorflow', printResults=False):
    if (api=='tensorflow'):
        y_predict = model(X_test)
    elif(api=='pytorch'):
        y_predict = predict_pytorch(X_test, model)
        
    print(y_predict.shape)
    y_predict = [1 if val > 0.5 else 0 for val in y_predict]
    cm = metrics.confusion_matrix(Y_test, y_predict)
    accuracy, specificity, sensitivity, precision, f1score = extraerSP_SS(cm)
  
    cm = metrics.confusion_matrix(Y_test, y_predict)
    
    if (printResults):
        # Extract the results and show them
        accuracy, specificity, sensitivity, precision, f1score = extraerSP_SS(cm)
        statistics_s = f"\nRESULTS: \nACC:{accuracy:.3f} \nSS:{sensitivity:.3f} \nSP:{specificity:.3f} \nPr:{precision:.3f} \nScore:{f1score:.3f}"
        print(statistics_s)
        createConfusionMatrix(cm, "Model", save=False)
    return cm

def main():
    print("MAIN")
    
    #Set seed for reproducible results
    seed(2)
    
    X, Y = ReadData(light='NBI')
    
    originalNN_parameters = {'input_size_net':(224,224,3), 'output_size':1}
    # NASExperiment(X, Y, "OriginalNN 3", OriginalNN, originalNN_parameters, batch_size=32)
    
    autokeras_parameters = {'validation_split':0.15, 'epochs':50, 'max_trials':20}
    # NASExperiment(X, Y, "Autokeras", autokerasModel, autokeras_parameters)
    
    fpnas_parameters = {'validation_split':0.30, 'P':4, 'Q':10, 'E':10, 'T':1, 'D':None, 'batch_size':32,
                        'blocks_size':[32, 64]}
    # NASExperiment(X, Y, "FPNAS2-B2 T1", fpnasModel, fpnas_parameters, batch_size=32, leave_one_out=False)
    
    enas_parameters = {'epochs':10}
    NASExperiment(X, Y, "ENAS 3", enasModelFromNumpy, enas_parameters, batch_size=32, clearModel=False, api='pytorch')

if __name__ == '__main__':
  main()